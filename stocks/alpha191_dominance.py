"""Alpha191 筛选 A1 关：控制集 2^11 子集穷举。

要回答什么
----------
`alpha191.py` 的正交化只看了 **13 个控制集**（11 个单变量 + 无控制 + 全 11 个），
所以「部分独立」的 14 个候选只排除了两种解释：单个变量吃掉、全 11 个联手吃掉。
剩下的一种没排除：**中间大小的子集联手吃掉**（协议 §3.3 规则③「过控制嫌疑」）。
协议 §3.4 的注脚已实测过这个漏洞：两融 `m_shortsell_int` 在 7 档手挑集里看着独立，
占满 2^11 子集后 `log_amt`/`illiq20` **单个变量**就能吃掉它 —— 手挑集粒度不够。

本脚本对这 14 个候选占满 2^11 = 2048 个子集，判读用 `dominance.verdict`
（协议 §3.3 三条规则 + §11.4 优势分析），**阈值仍取 §3.5 的 KILL = 0.0704/5 = 0.0141，
不自造阈值**。

口径（写死在这里，是唯一 provenance）
------------------------------------
1. **复用 `alpha191.orth_scan`，只换 mask 表**。行集守卫（控制变量齐全 / len >= 30）、
   秩在因子实际行集上算、不剔涨跌停 —— 全部与出 `alpha191_ic.csv` 的那条链路逐字相同。
   自造一份扫描器会让读数与归档产物不可比。
2. 候选集 = `out/alpha191_ic.csv` 里 `verdict == "部分独立"` 的 14 个，**按名字排序**取。
   不在脚本里写死名单，避免与归档漂移。
3. 测试段 = `me[me > 2020-12-31]`（与 `alpha191.py` 的 `orth_dates` 同）。
4. 判读只看 **残差 IC**，不看 p 值（协议 §3.5）。
5. 本脚本**只读** `alpha191_ic.csv` / `alpha191_ic_series.csv`，不覆盖任何归档产物。

自检（跑完自动核，任何一条不过即非零退出）
---------------------------------------
- 子集数合计 = 2^11 = 2048
- 每个候选：k=0 的残差 IC 均值 == `alpha191_ic.csv` 的 `ic_nocontrol`（1e-12）；
  k=11 的残差 IC 均值 == `ic_allctrl`（1e-9）；单控制最弱值 == `ic_single_min`（1e-9）；
  k=11 的 R^2 均值 == `r2_allctrl`（1e-9）
  第一条 1e-12 是「同一根链路」的证明；后三条是「同一条读法」的证明。

用法
----
    uv run stocks/alpha191_dominance.py --limit 2     # 冒烟：只跑前 2 个候选
    uv run stocks/alpha191_dominance.py               # 全量：14 候选 × 69 期 × 2048 子集
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import dominance as dom
import factors
from alpha191 import (ALPHAS, build_orth_inputs, load_wide, log, orth_scan,
                      self_check)

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
REF_CSV = OUT_DIR / "alpha191_ic.csv"

# 不钉 memory_limit：实测钉 1400MB 会改变 capacity.build_features 的末位，经 11 个共线
# 控制变量的病态回归放大到 1e-8 量级（2026-10-14 实测：alpha015 全 11 残差 IC 不钉 vs 钉
# 差 5.3e-8），会让本脚本与归档产物对不上 —— 对账要求同链路，就不许钉。只给 spill 目录。
_DUCK_TMP = DATA_DIR / ".duckdb_tmp"
_DUCK_TMP.mkdir(parents=True, exist_ok=True)
duckdb.execute(f"SET temp_directory = '{_DUCK_TMP.as_posix()}'")
log(f"[a191dom] DuckDB 未钉 memory_limit（对账要求同链路），spill -> {_DUCK_TMP}")

SPLIT = "2020-12-31"
COLS = list(dom.CONTROLS)
N_K1K2 = len(COLS) + len(COLS) * (len(COLS) - 1) // 2      # C(11,1)+C(11,2) = 66
MASKS, _SIZES = dom.build_masks(len(COLS))                 # 全部 2^11 个子集


def by_size_table(mats: list[tuple[str, np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """按子集大小聚合的残差 IC 带（列名与 dominance_ic_by_size.csv 对齐，便于互看）。"""
    rows = []
    for f, ic_m, r2_m in mats:
        ic_mean = np.nanmean(ic_m, axis=0)
        icir = ic_mean / np.nanstd(ic_m, axis=0)
        r2_mean = np.nanmean(r2_m, axis=0)
        for k in range(len(COLS) + 1):
            sel = _SIZES == k
            ic = ic_mean[sel]
            dead = (np.abs(ic) < dom.KILL) | (np.abs(icir[sel]) < dom.DEAD_ICIR)
            rows.append({"factor": f, "k": k, "n_subsets": int(sel.sum()),
                         "ic_mean": float(np.nanmean(ic)),
                         "ic_median": float(np.nanmedian(ic)),
                         "ic_min": float(np.nanmin(ic)), "ic_max": float(np.nanmax(ic)),
                         "abs_ic_median": float(np.nanmedian(np.abs(ic))),
                         "abs_ic_max": float(np.nanmax(np.abs(ic))),
                         "n_flip": int((np.sign(ic) != np.sign(ic_mean[0])).sum()),
                         "n_eaten": int((np.abs(ic) < dom.KILL).sum()),
                         "dead_frac": float(dead.mean()),
                         "r2_mean": float(np.nanmean(r2_mean[sel]))})
    return pd.DataFrame(rows)


def judge(by_size: pd.DataFrame, ic_mean: dict[str, np.ndarray],
          r2_mean: dict[str, np.ndarray]) -> pd.DataFrame:
    """按 §3.3 三条规则逐候选判读，并附优势分析里最有解释力的控制变量。"""
    rows = []
    for f in ic_mean:
        s = by_size[(by_size["factor"] == f) & (by_size["k"] > 0)]
        med = s[s["abs_ic_median"] < dom.KILL]["k"]
        allk = s[s["abs_ic_max"] < dom.KILL]["k"]
        singles = np.array([ic_mean[f][1 << j] for j in range(len(COLS))])
        j = int(np.nanargmin(np.abs(singles)))
        eaten1 = [COLS[t] for t in range(len(COLS)) if abs(singles[t]) < dom.KILL]
        eaten12 = float(by_size[(by_size["factor"] == f)
                                & (by_size["k"].isin([1, 2]))]["n_eaten"].sum()) / N_K1K2
        k_med = int(med.iloc[0]) if len(med) else None
        k_all = int(allk.iloc[0]) if len(allk) else None
        inc = dom.dominance(r2_mean[f], MASKS)
        top = int(np.nanargmax(inc))
        rows.append({"factor": f, "ic_nocontrol": float(ic_mean[f][0]),
                     "ic_allctrl": float(ic_mean[f][-1]),
                     "k_med": k_med, "k_all": k_all,
                     "n_eaten1": len(eaten1), "eaten1": "/".join(eaten1),
                     "eaten_frac_k_le2": eaten12,
                     "n_flip_k1": int(by_size[(by_size["factor"] == f)
                                              & (by_size["k"] == 1)]["n_flip"].iloc[0]),
                     "r2_allctrl": float(r2_mean[f][-1]),
                     "top_control": COLS[top],
                     "top_avg_increment": float(inc[top]),
                     "verdict": dom.verdict(float(ic_mean[f][0]), eaten1, k_med, k_all, eaten12)})
    return pd.DataFrame(rows).sort_values("factor")


def reconcile(observed: dict[str, dict], ref: pd.DataFrame) -> None:
    """对账：本脚本的读数必须与归档的 alpha191_ic.csv 逐值一致，否则停。"""
    bad = []
    for f, obs in observed.items():
        r = ref.loc[f]
        for name, got, want, atol in (("ic_nocontrol", obs["ic_raw"],
                                       float(r["ic_nocontrol"]), 1e-12),
                                      ("ic_allctrl", obs["ic_mean"][-1],
                                       float(r["ic_allctrl"]), 1e-9),
                                      ("ic_single_min", obs["ic_single_min"],
                                       float(r["ic_single_min"]), 1e-9),
                                      ("r2_allctrl", obs["r2_mean"][-1],
                                       float(r["r2_allctrl"]), 1e-9)):
            if not np.isclose(got, want, rtol=0, atol=atol, equal_nan=True):
                bad.append(f"{f}.{name}: 本脚本 {got!r} vs 归档 {want!r}")
    if bad:
        raise SystemExit("[a191dom] 对账失败（前 8 条）：\n  " + "\n  ".join(bad[:8]))
    log(f"[a191dom] 对账通过（{len(observed)} 候选）：ic_nocontrol(1e-12)/ic_allctrl/"
        f"ic_single_min/r2_allctrl(1e-9) 与 alpha191_ic.csv 逐值一致")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个候选（冒烟用）")
    args = ap.parse_args()

    t0 = time.time()
    self_check()

    ref_all = pd.read_csv(REF_CSV).set_index("factor")
    names = sorted(ref_all.index[ref_all["verdict"] == "部分独立"].tolist())
    total = len(names)
    if args.limit:
        names = names[:args.limit]
    log(f"[a191dom] 候选集取自 {REF_CSV.name} 的「部分独立」：共 {total} 个，本次跑 {len(names)} 个")
    assert len(MASKS) == 2 ** len(COLS), f"子集表 {len(MASKS)} != 2^{len(COLS)}"
    log(f"[a191dom] 子集表 {len(MASKS)} 个 = 2^{len(COLS)}")

    panel_path = DATA_DIR / "panel_2015_2026.parquet"
    all_dates = pd.to_datetime(duckdb.sql(
        f"select distinct date from read_parquet('{panel_path.as_posix()}') order by 1").df()["date"])
    me = factors.month_end_dates(pd.DatetimeIndex(all_dates))
    ctrl, _ = build_orth_inputs(panel_path, me)
    orth_dates = me[me > pd.Timestamp(SPLIT)]
    log(f"[a191dom] 月末 {len(me)} 期；正交期 {len(orth_dates)} 期")

    wide = load_wide(panel_path, None)
    closes = wide["close"]
    o, h, l, c, v, a, r = (wide["open"], wide["high"], wide["low"], wide["close"],
                           wide["volume"], wide["amount"], wide["returns"])
    fwd = closes.reindex(me).pct_change().shift(-1)
    log(f"[a191dom] 面板 {closes.shape[0]} 交易日 × {closes.shape[1]} 只")

    observed: dict[str, dict] = {}
    mats: list[tuple[str, np.ndarray, np.ndarray]] = []
    for i, name in enumerate(names, 1):
        st = time.time()
        fac = ALPHAS[name](o, h, l, c, v, a, r).replace([np.inf, -np.inf], np.nan)
        ic0, ic_m, r2_m = orth_scan(fac.loc[me], ctrl, fwd, orth_dates, MASKS)
        ic_mean = np.nanmean(ic_m, axis=0)
        singles = np.array([ic_mean[1 << j] for j in range(len(COLS))])
        observed[name] = {"ic_raw": float(np.nanmean(ic0)) if ic0 else np.nan,
                          "ic_mean": ic_mean,
                          "r2_mean": np.nanmean(r2_m, axis=0),
                          "ic_single_min": float(singles[int(np.nanargmin(np.abs(singles)))])}
        mats.append((name, ic_m, r2_m))
        del fac
        log(f"[a191dom] {i}/{len(names)} {name} 完成（{time.time() - st:.1f}s）")

    reconcile(observed, ref_all)
    ic_mean = {f: v["ic_mean"] for f, v in observed.items()}
    r2_mean = {f: v["r2_mean"] for f, v in observed.items()}
    summary = judge(by_size_table(mats), ic_mean, r2_mean)

    print(f"\n{'因子':<10s}{'无控制':>9s}{'全11控制':>10s}{'k_med':>7s}{'k_all':>7s}"
          f"{'k<=2被吃':>9s}{'R2':>7s}  判读")
    for _, r in summary.iterrows():
        k_med = "—" if pd.isna(r["k_med"]) else f"{r['k_med']:.0f}"
        k_all = "—" if pd.isna(r["k_all"]) else f"{r['k_all']:.0f}"
        print(f"{r['factor']:<10s}{r['ic_nocontrol']:>+9.4f}{r['ic_allctrl']:>+10.4f}"
              f"{k_med:>7s}{k_all:>7s}{r['eaten_frac_k_le2']:>9.0%}{r['r2_allctrl']:>7.3f}"
              f"  {r['verdict']}")
    print(f"  KILL = {dom.KILL:.4f}（§3.5 量级线）；k_med/k_all = 中位/全部子集的 |残差 IC| "
          f"首次降到线下的子集大小；「k<=2 被吃」= 1~2 个控制变量的 {N_K1K2} 个子集里被吃掉的比例")

    print("\n-- 优势分析：解释力最强的控制变量 --")
    for _, r in summary.sort_values("top_avg_increment", ascending=False).iterrows():
        print(f"  {r['factor']:<10s} {r['top_control']:<12s} 平均增量 R2 {r['top_avg_increment']:>6.3f}")

    vc = summary["verdict"].value_counts()
    log(f"[a191dom] 判读分布: {vc.to_dict()}")
    log(f"[a191dom] 耗时 {time.time() - t0:.0f}s")

    if args.limit:
        log(f"[!] 本次只跑了 {len(names)}/{total} 个候选（--limit），产物不可当结论引用")
        return

    OUT_DIR.mkdir(exist_ok=True)
    by_size_table(mats).to_csv(OUT_DIR / "alpha191_dominance_ic_by_size.csv", index=False)
    summary.to_csv(OUT_DIR / "alpha191_dominance_summary.csv", index=False)
    (OUT_DIR / "alpha191_dominance.meta.txt").write_text(
        f"候选 {len(names)}（={REF_CSV.name} 全部「部分独立」）\n"
        f"子集 {len(MASKS)} = 2^{len(COLS)}；正交期 {len(orth_dates)}；分界 {SPLIT}\n"
        f"KILL = {dom.KILL:.4f}（协议 §3.5）；DEAD_ICIR = {dom.DEAD_ICIR}\n"
        f"判读分布：{vc.to_dict()}\n"
        f"对账：ic_nocontrol 1e-12 / ic_allctrl 1e-9 / ic_single_min 1e-9 / r2_allctrl 1e-9\n"
        f"耗时 {time.time() - t0:.0f}s\n", encoding="utf-8")
    log("[a191dom] 产物 -> out/alpha191_dominance_ic_by_size.csv + _summary.csv + .meta.txt")


if __name__ == "__main__":
    main()
