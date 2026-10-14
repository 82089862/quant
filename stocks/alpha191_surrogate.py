"""Alpha191 筛选第四关：surrogate 噪声底（置换收益侧）。

要回答什么
----------
观测到：147 个已算因子里 90 个过 Bonferroni、14 个「部分独立」、13 个候选
（Bonferroni ∧ 部分独立 ∧ 训练测试同号）。问：这些留存数能否与噪声区分？

做法：同一套筛选链，在 surrogate 数据上重跑 N 次，看留存数分布；
判据照 `leadlag.py` docstring：「留存率不高于对照即无发现」。

口径（2026-10-13 拍板；写死在这里，是唯一 provenance）
---------------------------------------------------
1. surrogate = 逐期打乱**收益侧的秩配对**，**不置换因子侧**。
   理由：正交化是拿因子值对 11 个控制集回归取残差。置换因子侧会同时打断
   「因子↔控制集」→ 残差≈原始因子 → 「部分独立」更容易成立 → 噪声底虚高。
   置换收益侧则因子侧的秩、正交残差秩、R² 全部是**迭代不变量**，只算一次。
2. universe 与观测 IC 同口径：`factors.ic_series` 的「前向收益非空数 > 5」，
   不加涨跌停 mask；正交块仍照 `orth_scan` 的「控制变量齐全 / len >= 30」。
3. Bonferroni 的 m = 实际算出来的因子数（与观测同链路，本次 147）。
4. N = 300，seed = 20261013。

秩的等价性
----------
打乱收益侧的**值**，等价于打乱该截面上的**秩向量**（秩的多重集不变，只换位置）。
所以每期只需对预算好的秩向量做 `rng.permutation`：

    IC   段: ic   = (ra · perm(rb))      / (|ra||rb|)
    正交 段: ic_m = (ra_res[m] · perm(rrank)) / (|ra_res[m]||rrank|)

三个模长都只与集合大小和秩有关，置换不变 → 全部预计算。

已知局限（读数必须一起看）
------------------------
- **因子间置换流独立**：让 147 个因子共享同一套置换，就得在每期子集上重算秩
  （算量 ×147）。现在各因子独立，等于抹掉「因子间在 surrogate 下的相关」，
  会让留存数分布的**方差偏小**。对头号读数不构成威胁：每个因子的边际零分布仍均匀，
  147 次检验在 Bonferroni 阈值下期望留存 0.05 个，观测到的是 90 —— 差三个数量级。
- 本方法只检验「因子秩 ↔ 下期收益的配对」，**验不了 IC 序列的时间序列依赖**
  （协议 §4 自己写明它「必要但不充分」）。

用法
----
    uv run stocks/alpha191_surrogate.py                    # 全量：147 因子 × N=300
    uv run stocks/alpha191_surrogate.py --limit 3 --n 5 --check   # 冒烟：验对账，不写文件
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import dominance as dom
import factors
from alpha191 import (ALPHAS, build_orth_inputs, gate, ic_stats, load_wide,
                      log, orth_verdict, p_value, self_check)

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
REF_CSV = OUT_DIR / "alpha191_ic.csv"

# 不钉 memory_limit：实测钉 1400MB 会改变 capacity.build_features 的末位，经 11 个共线
# 控制变量的病态回归放大到 1e-8 量级（2026-10-14 实测：alpha015 全 11 残差 IC 不钉 vs 钉
# 差 5.3e-8），与本脚本对账所用的 alpha191_ic.csv 不同源。而且它挡不住真正的 OOM ——
# 2026-10-14 那次失败死在 pandas 的 load_wide（119MiB 都拿不到），不是 DuckDB。
# 只留 spill 目录。
_DUCK_TMP = DATA_DIR / ".duckdb_tmp"
_DUCK_TMP.mkdir(parents=True, exist_ok=True)
duckdb.execute(f"SET temp_directory = '{_DUCK_TMP.as_posix()}'")
log(f"[surrogate] DuckDB 未钉 memory_limit（对账要求同链路），spill -> {_DUCK_TMP}")

SPLIT = "2020-12-31"
SEED = 20261013
N_SURROGATE = 300
MIN_IC_N = 5        # factors.ic_series 的守卫
MIN_ORTH_N = 30     # orth_scan 的守卫
COLS = list(dom.CONTROLS)


# ---------- 预计算件 ----------

def _demean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x - x.mean()


def ic_pair(y: pd.Series, f: pd.Series):
    """一期的 (因子秩, 收益秩, 1/(|ra||rb|))，与 factors.spearman 逐字等价。

    `factors.spearman(a, b)` = `a.rank().corr(b.rank())`：两边先各自在**非空集**上取秩，
    再在**交集**上算 Pearson。所以这里也必须「先各自取秩、再取交集」——
    直接在交集上重排秩会与观测值差出量级。
    """
    if f.notna().sum() <= MIN_IC_N:
        return None
    ra_f, rb_f = y.rank(), f.rank()
    common = ra_f.dropna().index.intersection(rb_f.dropna().index)
    if len(common) < 2:
        return None
    ra, rb = _demean(ra_f.reindex(common).to_numpy()), _demean(rb_f.reindex(common).to_numpy())
    den = math.sqrt(float(ra @ ra) * float(rb @ rb))
    if not den > 0:
        return None
    return ra, rb, 1.0 / den


def orth_arrays(block: pd.DataFrame, masks: np.ndarray):
    """一个截面的 (残差秩矩阵, R², 缩放, 收益秩)。

    数学与 `dominance.subset_scan` 完全相同，只是把「残差秩」留下来给 surrogate 复用
    —— 观测值就是那份置换不变的残差秩本身。
    """
    y = block["y"].to_numpy(dtype=float)
    rrank = dom._ranks(block["fwd"].to_numpy(dtype=float))
    if not np.any(rrank):
        return None
    x = block[COLS].to_numpy(dtype=float)
    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    if not yc.any():
        return None
    sx = xc.std(axis=0)
    xc = xc / np.where(sx > 0, sx, 1.0)
    yc = yc / yc.std()
    sst = float(yc @ yc)
    if sst <= 0:
        return None
    nrm_b = float(rrank @ rrank)
    if not nrm_b > 0:
        return None
    gram = xc.T @ xc
    ra_mat = np.empty((len(masks), len(y)))
    r2 = np.empty(len(masks))
    scale = np.empty(len(masks))
    for m, bits in enumerate(masks):
        idx = np.flatnonzero(bits)
        if len(idx) == 0:
            resid = yc
        else:
            c = xc[:, idx].T @ yc
            sub = gram[np.ix_(idx, idx)]
            try:
                beta = np.linalg.solve(sub, c)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(sub, c, rcond=None)[0]
            resid = yc - xc[:, idx] @ beta
        r2[m] = 1.0 - float(resid @ resid) / sst
        ra = dom._ranks(resid)
        ra_mat[m] = ra
        nrm_a = float(ra @ ra)
        scale[m] = 1.0 / math.sqrt(nrm_a * nrm_b) if nrm_a > 0 else np.nan
    return ra_mat, r2, scale, rrank


def build_precomp(fac_me: pd.DataFrame, fwd: pd.DataFrame, ctrl: dict,
                  masks: np.ndarray, orth_dates: pd.DatetimeIndex) -> dict:
    """一个因子的全部迭代不变量。逐期按 orth_scan 的守卫条件筛。"""
    ics, ics_dates = [], []
    for dt in fwd.index:
        if dt not in fac_me.index:
            continue
        b = ic_pair(fac_me.loc[dt], fwd.loc[dt])
        if b is not None:
            ics.append(b)
            ics_dates.append(dt)

    orth, raw_dates = [], []
    for dt in orth_dates:
        if dt not in fac_me.index or dt not in fwd.index:
            continue
        if fwd.loc[dt].notna().sum() > MIN_IC_N:
            raw_dates.append(dt)          # orth_scan 的 ic0 守卫：只看前向收益非空数
        block = pd.DataFrame({c: ctrl[c].loc[dt] for c in COLS})
        block["y"] = fac_me.loc[dt].reindex(block.index)
        block["fwd"] = fwd.loc[dt].reindex(block.index)
        block = block.dropna()
        if len(block) < MIN_ORTH_N:
            continue
        got = orth_arrays(block, masks)
        if got is not None:
            orth.append((dt, *got))

    return {"ics": ics, "ics_dates": pd.DatetimeIndex(ics_dates),
            "orth": orth, "raw_dates": pd.DatetimeIndex(raw_dates)}


# ---------- 观测值 / 一次 surrogate ----------

def ic_series_of(pre: dict, rng: np.random.Generator | None = None) -> pd.Series:
    vals = np.empty(len(pre["ics"]))
    for i, (ra, rb, inv) in enumerate(pre["ics"]):
        vals[i] = ra @ (rb[rng.permutation(len(rb))] if rng is not None else rb) * inv
    return pd.Series(vals, index=pre["ics_dates"])


def orth_ics_of(pre: dict, rng: np.random.Generator | None = None) -> np.ndarray:
    """(n_date, n_mask) 的残差 IC 矩阵；rng 为空即观测值。"""
    out = []
    for _, ra_mat, _r2, scale, rrank in pre["orth"]:
        rb = rrank[rng.permutation(len(rrank))] if rng is not None else rrank
        out.append(ra_mat @ rb * scale)
    return np.array(out) if out else np.empty((0, 0))


def verdict_of(ic: pd.Series, pre: dict, ic_m: np.ndarray) -> str:
    """用 alpha191 自己的 orth_verdict 判读，不自造阈值。"""
    if ic_m.size == 0:
        return "F5 本来就无信号"
    ic_raw = float(ic.reindex(pre["raw_dates"]).mean())
    by_mask = np.nanmean(ic_m, axis=0)
    r2_all = float(np.nanmean([p[2][-1] for p in pre["orth"]]))
    tag, _why, _ = orth_verdict(ic_raw, by_mask, r2_all)
    return tag


def stats_of(ic: pd.Series) -> dict:
    full = ic_stats(ic, SPLIT)
    tr = ic_stats(ic[ic.index <= pd.Timestamp(SPLIT)], SPLIT)
    te = ic_stats(ic[ic.index > pd.Timestamp(SPLIT)], SPLIT)
    sign_same = int(np.isfinite(tr["ic_mean"]) and np.isfinite(te["ic_mean"])
                    and np.sign(tr["ic_mean"]) == np.sign(te["ic_mean"]))
    return {"ic_mean": full["ic_mean"], "icir": full["icir"], "t": full["t"],
            "p": p_value(full["t"]), "sign_same": sign_same}


# ---------- 主流程 ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_SURROGATE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个因子（冒烟用）")
    ap.add_argument("--check", action="store_true", help="只打印诊断，不写文件")
    args = ap.parse_args()

    t0 = time.time()
    self_check()

    ref = pd.read_csv(REF_CSV)
    names = ref.loc[ref["status"] != "跳过", "factor"].tolist()
    total = len(names)
    if args.limit:
        names = names[:args.limit]
    log(f"[surrogate] 因子集取自 {REF_CSV.name}：已算 {total} 个，本次跑 {len(names)} 个")

    panel_path = DATA_DIR / "panel_2015_2026.parquet"
    log("[surrogate] 控制集（11 个价格因子）")
    all_dates = pd.to_datetime(duckdb.sql(
        f"select distinct date from read_parquet('{panel_path.as_posix()}') order by 1").df()["date"])
    me = factors.month_end_dates(pd.DatetimeIndex(all_dates))
    ctrl, masks = build_orth_inputs(panel_path, me)
    orth_dates = me[me > pd.Timestamp(SPLIT)]
    log(f"[surrogate] 月末 {len(me)} 期；正交期 {len(orth_dates)} 期；mask {len(masks)} 个")

    wide = load_wide(panel_path, None)
    closes = wide["close"]
    dates = closes.index
    o, h, l, c, v, a, r = (wide["open"], wide["high"], wide["low"], wide["close"],
                           wide["volume"], wide["amount"], wide["returns"])
    fwd = closes.reindex(me).pct_change().shift(-1)
    log(f"[surrogate] 面板 {closes.shape[0]} 交易日 × {closes.shape[1]} 只")

    F, N = len(names), args.n
    p_arr = np.full((F, N), np.nan)
    sign_arr = np.zeros((F, N), dtype=bool)
    part_arr = np.zeros((F, N), dtype=bool)
    icm_arr = np.full((F, N), np.nan)
    icr_arr = np.full((F, N), np.nan)
    observed: list[dict] = []

    for i, name in enumerate(names):
        fac = ALPHAS[name](o, h, l, c, v, a, r).replace([np.inf, -np.inf], np.nan)
        pre = build_precomp(fac.loc[me], fwd, ctrl, masks, orth_dates)
        del fac

        ic_obs = ic_series_of(pre)
        row = stats_of(ic_obs)
        tag = verdict_of(ic_obs, pre, orth_ics_of(pre))
        observed.append({"factor": name, "ic_mean": row["ic_mean"], "icir": row["icir"],
                         "t": row["t"], "p": row["p"], "sign_same": row["sign_same"],
                         "verdict": tag})
        del ic_obs

        for it in range(N):
            rng_ic = np.random.default_rng([args.seed, i, it, 0])
            ic = ic_series_of(pre, rng_ic)
            st = stats_of(ic)
            p_arr[i, it] = st["p"]
            sign_arr[i, it] = bool(st["sign_same"])
            icm_arr[i, it] = st["ic_mean"]
            icr_arr[i, it] = st["icir"]
            rng_or = np.random.default_rng([args.seed, i, it, 1])
            part_arr[i, it] = verdict_of(ic, pre, orth_ics_of(pre, rng_or)) == "部分独立"
            del ic
        del pre
        log(f"[surrogate] {i + 1}/{F} {name} 完成（{time.time() - t0:.0f}s）")

    gate(observed)                                  # 观测侧的 Bonferroni / FDR
    thr = 0.05 / total                              # 与 gate 同口径：m = 实际算出来的因子数
    bonf = p_arr < thr
    cand = bonf & sign_arr & part_arr
    n_pass = bonf.sum(axis=0)
    n_cand = cand.sum(axis=0)

    # ---- 对账：观测值与已归档产物逐值比 ----
    ref_ok = ref[ref["status"] != "跳过"].set_index("factor")
    bad = []
    for row in observed:
        f = row["factor"]
        for col, atol in (("ic_mean", 1e-9), ("icir", 1e-7), ("t", 1e-7)):
            if not np.isclose(row[col], ref_ok.at[f, col], rtol=0, atol=atol, equal_nan=True):
                bad.append(f"{f}.{col}: {row[col]!r} vs 归档 {ref_ok.at[f, col]!r}")
        if row["verdict"] != ref_ok.at[f, "verdict"]:
            bad.append(f"{f}.verdict: {row['verdict']} vs 归档 {ref_ok.at[f, 'verdict']}")
    if bad:
        raise SystemExit("[surrogate] 对账失败（前 8 条）：\n  " + "\n  ".join(bad[:8]))
    n_b = sum(r["bonferroni"] for r in observed)
    n_p = sum(1 for r in observed if r["verdict"] == "部分独立")
    n_c = sum(1 for r in observed if r["bonferroni"] and r["verdict"] == "部分独立"
              and r["sign_same"])
    log(f"[surrogate] 对账通过（{len(observed)} 因子）：本脚本 过Bonferroni {n_b} / "
        f"部分独立 {n_p} / 候选 {n_c}")
    if not args.limit:
        log(f"[surrogate] 归档 90 / 14 / 13 —— 两边须一致（对账已逐值通过）")

    summary = (f"因子 {F}（归档 {total}）N={N} seed={args.seed}\n"
               f"观测：过Bonferroni {n_b}，部分独立 {n_p}，候选 {n_c}\n"
               f"surrogate 过Bonferroni：中位 {np.nanmedian(n_pass):.0f}，"
               f"95 分位 {np.percentile(n_pass, 95):.0f}，最大 {n_pass.max():.0f}\n"
               f"surrogate 候选：中位 {np.median(n_cand):.0f}，最大 {n_cand.max():.0f}\n"
               f"surrogate 下『候选 ≥ 观测候选』的次数：{int((n_cand >= n_c).sum())}/{N}\n"
               f"耗时 {time.time() - t0:.0f}s")
    log("[surrogate] " + summary.replace("\n", "\n[surrogate] "))

    if args.check:
        log("[surrogate] --check：未写文件")
        return

    rows = [{"factor": names[i],
             "observed_ic_mean": observed[i]["ic_mean"],
             "observed_icir": observed[i]["icir"],
             "observed_verdict": observed[i]["verdict"],
             "observed_bonferroni": observed[i]["bonferroni"],
             "observed_sign_same": observed[i]["sign_same"],
             "surrogate_ic_mean_q05": float(np.nanpercentile(icm_arr[i], 5)),
             "surrogate_ic_mean_q95": float(np.nanpercentile(icm_arr[i], 95)),
             "surrogate_icir_q95": float(np.nanpercentile(np.abs(icr_arr[i]), 95)),
             "surrogate_bonf_rate": float(np.nanmean(bonf[i])),
             "surrogate_candidate_rate": float(np.mean(cand[i])),
             "n_surrogate": N, "seed": args.seed} for i in range(F)]
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "alpha191_surrogate.csv"
    out.to_csv(out_path, index=False)
    (OUT_DIR / "alpha191_surrogate.meta.txt").write_text(summary + "\n", encoding="utf-8")
    log(f"[surrogate] {out.shape[1]} 列 × {out.shape[0]} 行 -> {out_path}"
        f"（汇总另落 alpha191_surrogate.meta.txt）")
    if F != total:
        log(f"[!] 本次只跑了 {F}/{total} 个因子（--limit），产物不可当结论引用")


if __name__ == "__main__":
    main()
