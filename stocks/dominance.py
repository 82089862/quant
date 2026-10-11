"""控制变量子集穷举 + 优势分析：给"过控制无法排除"一个定量判据。

margin_factor.py 手挑了 7 档控制集，看到 `m_shortsell_int` 在 4 个变量时才被吃掉，
无法区分两种解释：真暴露，还是共线变量联手过控制。这个脚本把 11 个控制变量的
**全部 2^11 = 2048 个子集**跑一遍，用两个口径回答：

  1. 残差 IC 在"子集大小"上的分布（衰减带）。7 档手选集只是带里的一条线；
     带如果很宽，说明手选档位代表不了全体。
  2. 优势分析（Budescu 1993）：每个控制变量的平均增量 R² —— 穷举所有不含它的子集，
     取 R²(S∪{j}) − R²(S) 的均值。平均增量高 = 真独立维度；单变量 R² 高但平均增量低
     = 与别的变量共线搭便车（§3.6 陷阱二的成因量化）。

判据（沿用 §3.5 已有的量级线，不自造阈值）：
  KILL = 0.0704 / 5 ≈ 0.0141，其中 0.0704 是本项目唯一验证过的真实因子（20 日反转）的 IC。
  k_med = 第一次「该大小下所有子集的 |残差 IC| 中位数 < KILL」的子集大小
  k_all = 第一次「该大小下所有子集 |残差 IC| 全部 < KILL」的子集大小
  形状标签（优先级从上到下，前两个是死因，后两个是形状）：
    |无控制 IC| < KILL                 -> 本来就无信号，不走换皮/过控制判读
    存在单个控制变量就 |IC| < KILL     -> §3.3 规则①：确凿换皮（并打印是哪个变量）
    k_med <= 2                          -> §3.3 规则②：少数变量就能吃掉
    k_med >= 4 且 k<=2 的多数子集未被吃掉 -> §3.3 规则③：过控制嫌疑
    其余                                 -> 部分独立，交给 §3.5 的量级裁决

  （第一版曾用 |IC| < 0.005 当阈值，与 §3.5 的 0.0704/5 不一致，已废弃。）

自检（跑完自动核）：
  - 子集数合计 = 2^11
  - 全量控制集（k=11）的残差 IC 必须等于 out/margin_factor_control_diag.csv 的「全部11个」
  - 全量控制集的 R² 必须等于 out/margin_factor_orth.csv 的 r2_vs_price
    （两者都必须按各自因子的 NaN 单独剔行 —— 第一版共用一份行集，差 1e-4 被判为不一致）

用法:
    uv run dominance.py --limit-dates 8   # 冒烟 + 计时
    uv run dominance.py                   # 全量（5 因子 × 70 期 × 2048 子集，约 2 分钟）
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import pandas as pd

import capacity
import factors
import margin_factor as mf

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"

CONTROLS = ["ret1", "mom5", "mom20", "mom60", "vol20", "vol60",
            "log_mv", "log_amt", "turnover_lv", "amp", "illiq20"]
FULL_MASK = (1 << len(CONTROLS)) - 1
KILL = 0.0704 / 5        # §3.5：真实因子 IC 的 1/5
DEAD_ICIR = 0.3          # §3.5：ICIR 线


def build_masks(k: int) -> tuple[np.ndarray, np.ndarray]:
    """所有子集 + 每个子集的大小。"""
    masks = np.array([[(m >> i) & 1 for i in range(k)] for m in range(1 << k)], dtype=bool)
    return masks, masks.sum(axis=1)


def _ranks(a: np.ndarray) -> np.ndarray:
    """秩（1..n，并列取平均）减均值，口径同 pandas rank。

    并列必须取平均秩：停牌股的前向收益会是一批完全相同的值，
    顺序秩会让 IC 与 margin_factor.py 差 1e-5 量级。
    """
    order = np.argsort(a, kind="stable")
    s = a[order]
    n = len(a)
    new = np.empty(n, dtype=bool)
    new[0] = True
    new[1:] = s[1:] != s[:-1]
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], n)
    r = np.empty(n)
    r[order] = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    return r - (n + 1) / 2.0


def subset_scan(x: np.ndarray, y: np.ndarray, rrank: np.ndarray,
                masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """一个截面上扫全部子集，返回 (M,) 残差 IC 与 (M,) R²。x/y 已按该因子的 NaN 剔行。"""
    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    if not yc.any():
        nan = np.full(len(masks), np.nan)
        return nan, nan
    # 先标准化：m_bal_chg 这类因子有极端值，不缩放会让正规方程平方条件数，
    # 与 margin_factor.py 的 SVD 解差到 1e-6。R² 与秩 IC 都对此不变。
    sx = xc.std(axis=0)
    xc = xc / np.where(sx > 0, sx, 1.0)
    yc = yc / yc.std()
    sst = float(yc @ yc)
    if sst <= 0:
        nan = np.full(len(masks), np.nan)
        return nan, nan
    gram = xc.T @ xc
    nrm_b = float(rrank @ rrank)
    ic, r2 = np.empty(len(masks)), np.empty(len(masks))
    for m, bits in enumerate(masks):
        idx = np.flatnonzero(bits)
        if len(idx) == 0:
            resid = yc                                   # 无控制
        else:
            c = xc[:, idx].T @ yc
            sub = gram[np.ix_(idx, idx)]
            try:
                beta = np.linalg.solve(sub, c)
            except np.linalg.LinAlgError:                # 完全共线的控制集
                beta = np.linalg.lstsq(sub, c, rcond=None)[0]
            resid = yc - xc[:, idx] @ beta
        r2[m] = 1.0 - float(resid @ resid) / sst
        ra = _ranks(resid)
        den = np.sqrt((ra @ ra) * nrm_b)
        ic[m] = float(ra @ rrank / den) if den > 0 else np.nan
    return ic, r2


def per_date_scan(panel: pd.DataFrame, fwd: pd.DataFrame, ok: pd.DataFrame,
                  dates: pd.DatetimeIndex, masks: np.ndarray
                  ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """逐期扫描，返回 ({因子: (期数, M) IC}, {因子: (期数, M) R²})。"""
    cols = [f"r_{c}" for c in CONTROLS]
    ic_hist: dict[str, list] = {f: [] for f in mf.FACTORS}
    r2_hist: dict[str, list] = {f: [] for f in mf.FACTORS}
    for i, d in enumerate(dates, 1):
        if d not in fwd.index or d not in ok.index:
            continue
        r = fwd.loc[d].dropna()
        sub = panel.loc[panel["date"] == d].dropna(subset=cols)
        keep = ok.loc[d].reindex(sub["symbol"]).fillna(False).astype(bool).to_numpy()
        sub = sub[keep]
        sub = sub[sub["symbol"].isin(r.index)]
        if len(sub) < 30:
            continue
        xs = sub[cols].to_numpy(dtype=float)
        got = False
        for f in mf.FACTORS:                    # 每个因子按自己的 NaN 剔行，口径同 margin_factor
            m = sub[f].notna().to_numpy()
            if m.sum() < 30:
                continue
            # 秩必须在该因子实际用到的行集上算：先算再剔行会让秩不是 1..n，Spearman 会偏
            rrank = _ranks(r.loc[sub["symbol"].to_numpy()[m]].to_numpy(dtype=float))
            ic, r2 = subset_scan(xs[m], sub[f].to_numpy(dtype=float)[m], rrank, masks)
            ic_hist[f].append(ic)
            r2_hist[f].append(r2)
            got = True
        if got and (i % 10 == 0 or i == len(dates)):
            print(f"[dom] 截面 {i}/{len(dates)}")
    return ({f: np.vstack(v) for f, v in ic_hist.items()},
            {f: np.vstack(v) for f, v in r2_hist.items()})


def dominance(r2: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """每个控制变量的平均增量 R²。"""
    out = np.empty(len(CONTROLS))
    for j in range(len(CONTROLS)):
        w = np.array([m for m in range(len(masks)) if not masks[m, j]])
        out[j] = float((r2[w | (1 << j)] - r2[w]).mean())
    return out


def verdict(ic_raw: float, eaten1: list[str], k_med: int | None, k_all: int | None,
            eaten12: float) -> str:
    """只有无控制 IC 明显高于量级线（2 倍）时，"被吃掉"才说明有暴露可谈。"""
    if abs(ic_raw) < KILL:
        return "本来就无信号"
    if eaten1 and abs(ic_raw) >= 2 * KILL:
        return f"确凿换皮：单变量（{'/'.join(eaten1)}）就能吃掉（§3.3 规则①）"
    if k_med is None:
        return "任何子集都吃不掉（死因在别处）"
    if k_med <= 2:
        return "确凿换皮：少数变量就能吃掉（§3.3 规则②）"
    if k_med >= 4 and eaten12 <= 0.5:
        return "过控制嫌疑：要多个（共线）变量联手（§3.3 规则③）"
    return "部分独立，交给 §3.5 量级裁决"


def self_check(by_size: pd.DataFrame, r2_full: pd.Series) -> None:
    diag = pd.read_csv(OUT_DIR / "margin_factor_control_diag.csv", index_col="factor")
    orth = pd.read_csv(OUT_DIR / "margin_factor_orth.csv", index_col="factor")
    n = int(by_size[by_size["factor"] == mf.FACTORS[0]]["n_subsets"].sum())
    assert n == 2 ** len(CONTROLS), f"子集数 {n} != 2^{len(CONTROLS)}"
    for f in mf.FACTORS:
        ic_full = float(by_size[(by_size["factor"] == f)
                                & (by_size["k"] == len(CONTROLS))]["ic_mean"].iloc[0])
        d = abs(ic_full - float(diag.loc[f, "全部11个"]))
        assert d < 1e-9, f"{f} 全量残差 IC 对不上 margin_factor.py: 差 {d:.2e}"
        d2 = abs(float(r2_full[f]) - float(orth.loc[f, "r2_vs_price"]))
        assert d2 < 1e-9, f"{f} 全量 R² 对不上 margin_factor.py: 差 {d2:.2e}"
    print("[dom] 自检通过：子集数、全量残差 IC（<1e-9）、全量 R2（<1e-9）与 margin_factor.py 一致")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(DATA_DIR / "panel_2015_2026.parquet"))
    ap.add_argument("--split", default="2020-12-31")
    ap.add_argument("--limit-dates", type=int, default=0, help="只用前 N 个测试截面（冒烟用）")
    args = ap.parse_args()

    masks, sizes = build_masks(len(CONTROLS))
    t0 = time.time()
    panel_path = pathlib.Path(args.panel)
    closes = factors.load_closes(panel_path)
    me = factors.month_end_dates(closes.index)
    fp = mf.build_factor_panel(me, panel_path)

    last = closes.reindex(me)
    fwd = last.pct_change().shift(-1)
    lim = pd.Series([mf.price_limit_pct(s) for s in closes.columns], index=closes.columns)
    ok = last.pct_change().abs().lt(lim - 0.005)
    test_dates = me[me > pd.Timestamp(args.split)]
    if args.limit_dates:
        test_dates = test_dates[:args.limit_dates]

    px = capacity.build_features(panel_path, 25.0, dates=me)
    panel = fp.merge(px[["date", "symbol"] + [f"r_{c}" for c in CONTROLS]],
                     on=["date", "symbol"], how="inner")
    print(f"[dom] {len(test_dates)} 个测试截面 × {len(masks)} 个子集"
          f" × {len(mf.FACTORS)} 个因子，准备 {time.time() - t0:.0f}s")

    ic_hist, r2_hist = per_date_scan(panel, fwd, ok, test_dates, masks)
    ic_mean = {f: np.nanmean(v, axis=0) for f, v in ic_hist.items()}
    icir = {f: np.nanmean(v, axis=0) / np.nanstd(v, axis=0) for f, v in ic_hist.items()}
    r2_mean = {f: np.nanmean(v, axis=0) for f, v in r2_hist.items()}
    n_period = {f: v.shape[0] for f, v in ic_hist.items()}
    print(f"[dom] 扫描完 {n_period}, 总耗时 {time.time() - t0:.0f}s\n")

    rows = []
    for f in mf.FACTORS:
        for k in range(len(CONTROLS) + 1):
            sel = sizes == k
            ic, r2 = ic_mean[f][sel], r2_mean[f][sel]
            dead = (np.abs(ic) < KILL) | (np.abs(icir[f][sel]) < DEAD_ICIR)
            rows.append({"factor": f, "k": k, "n_subsets": int(sel.sum()),
                         "ic_mean": float(np.nanmean(ic)),
                         "ic_median": float(np.nanmedian(ic)),
                         "ic_min": float(np.nanmin(ic)), "ic_max": float(np.nanmax(ic)),
                         "abs_ic_median": float(np.nanmedian(np.abs(ic))),
                         "abs_ic_max": float(np.nanmax(np.abs(ic))),
                         "n_flip": int((np.sign(ic) != np.sign(ic_mean[f][0])).sum()),
                         "n_eaten": int((np.abs(ic) < KILL).sum()),
                         "dead_frac": float(dead.mean()),
                         "r2_mean": float(np.nanmean(r2))})
    by_size = pd.DataFrame(rows)

    print(f"{'因子':<16s}{'k':>3s}{'子集数':>7s}{'IC中位':>9s}{'|IC|中位':>10s}"
          f"{'IC最小':>9s}{'IC最大':>9s}{'翻转数':>7s}{'不可用占比':>11s}{'R2均值':>9s}")
    for f in mf.FACTORS:
        for _, r in by_size[by_size["factor"] == f].iterrows():
            print(f"{f:<16s}{r['k']:>3.0f}{r['n_subsets']:>7.0f}{r['ic_median']:>+9.4f}"
                  f"{r['abs_ic_median']:>10.4f}{r['ic_min']:>+9.4f}{r['ic_max']:>+9.4f}"
                  f"{r['n_flip']:>7.0f}{r['dead_frac']:>11.2f}{r['r2_mean']:>9.3f}")
    print(f"  KILL = 0.0704/5 = {KILL:.4f}（§3.5 量级线）；不可用 = |IC| < KILL 或 |ICIR| < {DEAD_ICIR}")

    # 单变量残差 IC：哪个控制变量单独就能吃掉它
    single = {f: {CONTROLS[j]: float(ic_mean[f][1 << j]) for j in range(len(CONTROLS))}
              for f in mf.FACTORS}
    print("\n-- 单变量残差 IC（k=1 的 11 个子集）--")
    print(f"{'因子':<16s}{'无控制':>9s}" + "".join(f"{c:>11s}" for c in CONTROLS))
    for f in mf.FACTORS:
        cells = "".join(f"{single[f][c]:>+9.4f}{'!' if abs(single[f][c]) < KILL else ' '}"
                         f"{'':>1s}" for c in CONTROLS)
        print(f"{f:<16s}{ic_mean[f][0]:>+9.4f}{cells}")
    print("  ! = 单个变量就把它打到 KILL 线以下（§3.3 规则①）")

    print("\n-- 判读 --")
    for f in mf.FACTORS:
        s = by_size[(by_size["factor"] == f) & (by_size["k"] > 0)]
        med = s[s["abs_ic_median"] < KILL]["k"]
        allk = s[s["abs_ic_max"] < KILL]["k"]
        eaten1 = [c for c in CONTROLS if abs(single[f][c]) < KILL]
        eaten12 = float(by_size[(by_size["factor"] == f) & (by_size["k"].isin([1, 2]))]
                        ["n_eaten"].sum()) / 66.0
        k_med = int(med.iloc[0]) if len(med) else None
        k_all = int(allk.iloc[0]) if len(allk) else None
        print(f"  {f:<16s} k_med={k_med}  k_all={k_all}  k<=2 被吃掉 {eaten12:.0%}  "
              f"-> {verdict(ic_mean[f][0], eaten1, k_med, k_all, eaten12)}")

    d_rows = []
    print("\n-- 优势分析：控制变量的平均增量 R2（对因子的解释力）--")
    for f in mf.FACTORS:
        inc = dominance(r2_mean[f], masks)
        solo = np.array([r2_mean[f][1 << j] for j in range(len(CONTROLS))])
        ic1 = np.array([single[f][c] for c in CONTROLS])
        print(f"  {f}")
        for j in np.argsort(-inc):
            share = inc[j] / solo[j] if solo[j] > 1e-12 else float("nan")
            print(f"    {CONTROLS[j]:<12s} 单变量R2 {solo[j]:>6.3f}"
                  f"  平均增量R2 {inc[j]:>6.3f}  (增量/单变量 {share:>5.2f})")
            d_rows.append({"factor": f, "control": CONTROLS[j], "ic_single": ic1[j],
                           "r2_solo": solo[j], "r2_avg_increment": inc[j],
                           "increment_over_solo": share})

    OUT_DIR.mkdir(exist_ok=True)
    by_size.to_csv(OUT_DIR / "dominance_ic_by_size.csv", index=False)
    pd.DataFrame(d_rows).to_csv(OUT_DIR / "dominance_var.csv", index=False)
    if not args.limit_dates:
        self_check(by_size, pd.Series({f: r2_mean[f][FULL_MASK] for f in mf.FACTORS}))
    print("\n[dom] 明细 -> out/dominance_ic_by_size.csv, out/dominance_var.csv")
    print("[dom] 局限：只做线性截面控制，未含行业哑变量；11 个控制变量本身高度共线，"
          "优势分析给的是'重要性分摊'，不是因果贡献。")


if __name__ == "__main__":
    main()
