"""Deflated Sharpe Ratio (DSR) + Probability of Backtest Overfitting (PBO)

输入: stocks/out/portfolio_check_yearly.csv (年度 G5 收益)
输出: stocks/out/pbo_dsr.csv
"""
from __future__ import annotations
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

OUT_DIR = pathlib.Path("stocks/out")


def norm_ppf(p: float) -> float:
    """逆正态 CDF (Abramowitz & Stegun 26.2.23)."""
    if p <= 0:
        return float("-inf")
    if p >= 1:
        return float("inf")
    if p < 0.5:
        return -norm_ppf(1 - p)
    t = np.sqrt(-2 * np.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1*t + c2*t**2) / (1 + d1*t + d2*t**2 + d3*t**3)


def dsr_one(returns: pd.Series, n_trials: int, alpha: float = 0.05) -> dict:
    """单个策略的 DSR. returns 是年化收益序列."""
    T = len(returns.dropna())
    if T < 3:
        return {"n_periods": T, "sr_obs": float("nan"), "dsr": float("nan"),
                "sr_bonf": float("nan"), "passed_dsr": False, "passed_bonf": False,
                "note": f"N={T} 不足"}

    sr_obs = float(returns.mean() / returns.std())
    sk = float(returns.skew())
    ku = float(returns.kurt())

    # gamma 修正项
    gamma = 1 + 0.5 * sr_obs * sk + (ku - 3) / 24 * sr_obs**2
    gamma = max(gamma, 1e-12)

    # B = sqrt(2 * gamma * ln(N)) / sqrt(T)
    B = np.sqrt(2 * gamma * np.log(n_trials)) / np.sqrt(T)

    # DSR
    dsr_val = (sr_obs - B) / np.sqrt(gamma)

    # Bonferroni 阈值
    z_bonf = norm_ppf(1 - alpha / (2 * n_trials))
    sr_bonf = z_bonf / np.sqrt(T)

    return {
        "n_periods": T,
        "sr_obs": sr_obs,
        "skew": sk,
        "kurt": ku,
        "gamma": float(gamma),
        "B": float(B),
        "dsr": float(dsr_val),
        "sr_bonf": float(sr_bonf),
        "passed_dsr": bool(dsr_val > 0),
        "passed_bonf": bool(sr_obs > sr_bonf),
    }


def pbo_cscv(returns_panel: pd.DataFrame, n_folds: int = 5) -> dict:
    """PBO via rolling train/test."""
    T, N = returns_panel.shape
    if T < n_folds * 2:
        return {"pbo_estimate": float("nan"), "note": f"T={T} 不足 (需>={n_folds*2})"}

    fold_size = T // n_folds
    oos_list, ios_list = [], []
    for i in range(n_folds):
        tr_end = (i + 1) * fold_size
        te_start = tr_end
        te_end = min(tr_end + fold_size, T)
        if te_end <= te_start:
            break
        train = returns_panel.iloc[:tr_end]
        test = returns_panel.iloc[te_start:te_end]

        train_sr = (train.mean() / train.std()).dropna()
        if len(train_sr) == 0:
            continue
        best_idx = train_sr.idxmax()
        best_sr = float(train_sr.max())

        test_col = test[best_idx]
        test_sr = float(test_col.mean() / test_col.std()) if test_col.std() > 0 else 0.0

        oos_list.append(test_sr)
        ios_list.append(best_sr)

    if not oos_list:
        return {"pbo_estimate": float("nan"), "note": "无有效 OOS"}

    ios_mean = float(np.mean(ios_list))
    oos_mean = float(np.mean(oos_list))
    gap = ios_mean - oos_mean
    pbo = max(0.0, min(1.0, gap / ios_mean)) if ios_mean > 0 else 0.0

    return {
        "ios_mean_sr": ios_mean,
        "oos_mean_sr": oos_mean,
        "gap": gap,
        "pbo_estimate": pbo,
        "passed_pbo": pbo < 0.5,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(OUT_DIR / "portfolio_check_yearly.csv"))
    ap.add_argument("--n-trials", type=int, default=191)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    y = pd.read_csv(args.panel)
    full = y[y["完整年"] == True].copy()
    pivot = full.pivot(index="factor", columns="year", values="G5").sort_index()
    print(f"[pbo_dsr] 候选 {pivot.shape[0]} 个, 完整年 {pivot.shape[1]} 个")

    rows = []
    for c in pivot.index:
        rets = pivot.loc[c].dropna()
        r = dsr_one(rets, n_trials=args.n_trials)
        r["factor"] = c
        rows.append(r)

    dsr_df = pd.DataFrame(rows)
    print("\n--- Deflated Sharpe Ratio ---")
    cols = [c for c in ["factor","n_periods","sr_obs","dsr","sr_bonf","passed_dsr","passed_bonf"] if c in dsr_df.columns]
    print(dsr_df[cols].to_string(index=False))

    print("\n--- PBO (CSCV) ---")
    pbo_r = pbo_cscv(pivot.T, n_folds=args.n_folds)  # 转置: 行=年份, 列=候选
    for k, v in pbo_r.items():
        if isinstance(v, float):
            print(f"  {k}: {v:+.4f}" if 'sr' in k or 'gap' in k or 'pbo' in k else f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    dsr_df.to_csv(OUT_DIR / "pbo_dsr.csv", index=False)
    print(f"\n[pbo_dsr] 明细 -> {OUT_DIR}/pbo_dsr.csv")


if __name__ == "__main__":
    main()
