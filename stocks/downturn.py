"""下跌预警：给定标的 A，在 t-1 找有没有股票 B 的状态能预警 A 在 t 日下跌。

与 leadlag.py 的区别：因变量是二元的「是否下跌」，不是收益率。尾部传染（tail spillover）
在连续相关里会被平均掉，在事件条件概率里才显形——这是这个口径存在的理由。

四类候选信号（B 在 t-1 的状态）：
  B跌3%    B 当日跌超 3%
  B涨3%    B 当日涨超 3%
  B放量2x  B 当日成交量 > 前 20 日均量的 2 倍
  B缩量.5x B 当日成交量 < 前 20 日均量的一半

判据（缺一条就不能算发现）：lift = 条件下跌率/基准下跌率 > 1，且通过 Bonferroni/FDR、
样本外复现、并超过 surrogate 噪声底。

用法: uv run downturn.py --anchor sh600787
"""
from __future__ import annotations

import argparse
import math
import pathlib

import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"


def load_panel(path: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = duckdb.sql(
        f"select date, symbol, close, volume from read_parquet('{path.as_posix()}')"
    ).df()
    close = df.pivot(index="date", columns="symbol", values="close").sort_index()
    vol = df.pivot(index="date", columns="symbol", values="volume").sort_index()
    return close, vol


def build_signals(ret: pd.DataFrame, vol: pd.DataFrame) -> dict[str, np.ndarray]:
    """四类候选信号。放量/缩量以 t-1 之前 20 日均量为基准（shift 保证不含当日）。"""
    vma = vol.rolling(20, min_periods=10).mean().shift(1)
    return {
        "B跌3%": (ret < -0.03).to_numpy(),
        "B涨3%": (ret > 0.03).to_numpy(),
        "B放量2x": (vol > 2 * vma).to_numpy(),
        "B缩量.5x": (vol < 0.5 * vma).to_numpy(),
    }


def scan(target: np.ndarray, sig: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回每列 (触发次数 n, 其中 A 次日下跌的次数 k)。"""
    sel = sig[:-1] & valid[:-1] & valid[1:]
    n = sel.sum(axis=0)
    k = (sel & target[1:][:, None]).sum(axis=0)
    return n, k


def lift_stats(n: np.ndarray, k: np.ndarray, p0: float) -> tuple[np.ndarray, np.ndarray]:
    """条件下跌率、单尾 p 值（只关心 lift > 1）。"""
    n_ = np.maximum(np.asarray(n, dtype=float), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        p1 = np.where(np.asarray(n) >= 30, np.asarray(k, dtype=float) / n_, np.nan)
        z = (p1 - p0) / np.sqrt(p0 * (1 - p0) / n_)
    zf = np.asarray(z, dtype=float).ravel()
    p = np.fromiter(
        (0.5 * math.erfc(v / math.sqrt(2)) if math.isfinite(v) else 1.0 for v in zf),
        dtype=float, count=len(zf),
    )
    return p1, p.reshape(np.shape(z))


def bh_count(p: np.ndarray, alpha: float) -> int:
    ps = np.sort(p)
    return int((ps <= alpha * np.arange(1, len(ps) + 1) / len(ps)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="sh600787")
    ap.add_argument("--panel", default=str(DATA_DIR / "panel_2015_2026.parquet"))
    ap.add_argument("--split", default="2020-12-31")
    ap.add_argument("--topn", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    close, vol = load_panel(pathlib.Path(args.panel))
    symbols = list(close.columns)
    if args.anchor not in symbols:
        raise SystemExit(f"锚点 {args.anchor} 不在面板里")
    ret = close.pct_change()
    valid = ret.notna().to_numpy()
    ret_f = ret.fillna(0.0)
    signals = build_signals(ret_f, vol.fillna(0.0))
    ja = symbols.index(args.anchor)
    down = (ret_f.to_numpy() < 0)
    dates = close.index
    j_ex = [j for j in range(len(symbols)) if j != ja]
    n_tests = len(j_ex) * len(signals)

    tr = np.where(dates <= pd.Timestamp(args.split))[0]
    te = np.where(dates > pd.Timestamp(args.split))[0]
    print(f"[downturn] 锚点 {args.anchor} | 候选 {len(j_ex)} 只 × {len(signals)} 类信号 = {n_tests:,} 次检验")
    print(f"[downturn] Bonferroni 单次阈值 p < {args.alpha / n_tests:.2e}")
    print(f"[downturn] 训练 {dates[tr[0]].date()}~{dates[tr[-1]].date()} | 测试 {dates[te[0]].date()}~{dates[te[-1]].date()}\n")

    rows = []
    for name, sg in signals.items():
        n_tr, k_tr = scan(down[tr][:, ja], sg[tr][:, j_ex], valid[tr][:, j_ex])
        n_te, k_te = scan(down[te][:, ja], sg[te][:, j_ex], valid[te][:, j_ex])
        p0_tr = float(down[tr][:, ja].mean())
        p0_te = float(down[te][:, ja].mean())
        p1_tr, p_tr = lift_stats(n_tr, k_tr, p0_tr)
        p1_te, p_te = lift_stats(n_te, k_te, p0_te)
        thr_bon = args.alpha / n_tests
        print(f"[{name}] 基准下跌率 训练 {p0_tr:.1%} / 测试 {p0_te:.1%}"
              f" | 触发中位 {int(np.median(n_tr[n_tr > 0])) if (n_tr > 0).any() else 0} 次"
              f" | 名义 lift>1 显著 {int((p_tr < 0.05).sum())}"
              f" | Bonferroni {int((p_tr < thr_bon).sum())} | FDR {bh_count(p_tr, args.alpha)}")
        for j in np.argsort(-np.nan_to_num(p1_tr, nan=0.0))[: args.topn]:
            if np.isfinite(p1_tr[j]):
                rows.append({
                    "signal": name, "candidate": symbols[j_ex[j]],
                    "n_train": int(n_tr[j]), "p_down_train": round(float(p1_tr[j]), 4),
                    "lift_train": round(float(p1_tr[j] / p0_tr), 3), "p_train": float(p_tr[j]),
                    "p_down_test": round(float(p1_te[j]), 4) if np.isfinite(p1_te[j]) else None,
                    "lift_test": round(float(p1_te[j] / p0_te), 3) if np.isfinite(p1_te[j]) else None,
                })

    tbl = pd.DataFrame(rows).sort_values("lift_train", ascending=False)
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"downturn_{args.anchor}.csv"
    tbl.to_csv(out, index=False)

    print(f"\n[downturn] 训练期 lift 最高的 {args.topn} 个候选（每种信号）：")
    print(tbl.head(args.topn).to_string(index=False))
    kept = tbl[tbl["lift_test"].notna()]
    print(f"\n[downturn] 这些候选在测试期的 lift 中位 {kept['lift_test'].median():.3f}"
          f" | 仍 >1 的比例 {(kept['lift_test'] > 1).mean():.1%}")
    print(f"[downturn] 明细 -> {out}")
    print("[downturn] 提醒：条件概率高不等于因果，也不等于可交易；未计交易成本与做空可行性。")


if __name__ == "__main__":
    main()
