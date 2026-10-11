"""单因子计算与分层回测（月度调仓）。

信号日 = 每月最后一个交易日收盘，执行日 = 下一个交易日开盘。
把每期股票按因子值分 N 组，组内等权，逐组跑 backtest.run_portfolio。

用法:
    uv run factors.py                                  # 20 日动量，5 层
    uv run factors.py --factor reversal --window 60
    uv run factors.py --factor momentum --window 250    # 一年动量
"""
from __future__ import annotations

import argparse
import pathlib

import duckdb
import numpy as np
import pandas as pd

import backtest
import store

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"


def load_closes(panel_path: pathlib.Path) -> pd.DataFrame:
    """读成 date × symbol 的后复权收盘宽表。"""
    df = duckdb.sql(f"select date, symbol, close from read_parquet('{panel_path.as_posix()}')").df()
    return df.pivot(index="date", columns="symbol", values="close").sort_index()


def momentum(closes: pd.DataFrame, window: int) -> pd.DataFrame:
    """过去 window 个交易日的累计收益率。"""
    return closes.pct_change(window)


def month_end_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每个自然月的最后一个交易日。"""
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.groupby([s.index.year, s.index.month]).max().to_numpy())


def group_members(factor_row: pd.Series, n_groups: int, group: int) -> list[str]:
    """按因子值分组取成员。group=0 是因子最小的一组，n_groups-1 最大。"""
    r = factor_row.dropna()
    if len(r) < n_groups * 2:
        return []
    labels = pd.qcut(r.rank(method="first"), n_groups, labels=False)
    return labels[labels == group].index.tolist()


def build_schedule(factor: pd.DataFrame, n_groups: int, dates: pd.DatetimeIndex) -> dict[int, pd.DataFrame]:
    """每组一张 rebalance 表：index=执行日，columns=symbol，值=等权。"""
    sig = month_end_dates(dates)
    exe = dates[np.minimum(dates.get_indexer(sig) + 1, len(dates) - 1)]
    out = {}
    for g in range(n_groups):
        rows = {}
        for sd, ed in zip(sig, exe):
            members = group_members(factor.loc[sd], n_groups, g)
            rows[ed] = {m: 1.0 / len(members) for m in members} if members else {}
        sched = pd.DataFrame(rows).T
        out[g] = sched[~sched.index.duplicated(keep="last")]
    return out


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman = 各自 rank 后的 Pearson。不引 scipy。"""
    return a.rank().corr(b.rank())


def ic_series(closes: pd.DataFrame, factor: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """月度 Spearman IC：本期因子值 vs 下一期收益。"""
    fwd = closes.reindex(month_end_dates(dates)).pct_change().shift(-1)
    vals = {}
    for sd in fwd.index:
        row = fwd.loc[sd]
        if row.notna().sum() > 5:
            vals[sd] = spearman(factor.loc[sd], row)
    return pd.Series(vals).dropna()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", choices=["momentum", "reversal"], default="momentum")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-10-09")
    args = ap.parse_args()

    panel_path = DATA_DIR / f"panel_{args.start[:4]}_{args.end[:4]}.parquet"
    if not panel_path.exists():
        panel_path = store.build_panel(args.start, args.end)

    closes = load_closes(panel_path)
    dates = closes.index
    print(f"[factor] 宽表 {closes.shape[0]} 交易日 × {closes.shape[1]} 只")

    fac = momentum(closes, args.window)
    if args.factor == "reversal":
        fac = -fac
    print(f"[factor] {args.factor} window={args.window}，每期分 {args.groups} 组，月度调仓")

    panel = duckdb.sql(
        f"select date, symbol, open, close from read_parquet('{panel_path.as_posix()}')"
    ).df()

    # 每组上千只等权，本金必须够大：后复权价常有几千上万，1 亿分到 1000 只
    # 每只 10 万，才能保证每只都买得进。资金太小会让高价股被 floor 成 0 股，组合失真。
    cash0 = 100_000_000.0
    curves: dict[str, pd.Series] = {}
    for g, sched in build_schedule(fac, args.groups, dates).items():
        # lot_size=1：分层研究阶段忽略整手约束（整手留到实盘模拟，那时要用不复权价）
        eq, tr = backtest.run_portfolio(panel, sched, cash0=cash0, lot_size=1)
        curves[f"G{g + 1}"] = eq.set_index("date")["equity"]
        total = eq["equity"].iloc[-1] / cash0 - 1
        print(f"[factor] 第 {g + 1} 组（{'因子最小' if g == 0 else '因子最大' if g == args.groups - 1 else '中间'}）"
              f" 累计 {total:+.2%}  成交 {len(tr):,} 笔  未成交 {int(eq['blocked'].sum()):,} 笔")

    ic = ic_series(closes, fac, dates)
    print(f"[factor] IC 均值 {ic.mean():+.4f}  ICIR {ic.mean() / ic.std():+.2f}  {len(ic)} 期")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"layers_{args.factor}{args.window}_{args.groups}g.csv"
    pd.DataFrame(curves).to_csv(out)
    ic.to_csv(OUT_DIR / f"ic_{args.factor}{args.window}.csv", header=["ic"])
    print(f"[factor] 净值 -> {out}")

    totals = [float(s.iloc[-1]) for s in curves.values()]
    monotonic = all(a <= b for a, b in zip(totals, totals[1:])) or all(a >= b for a, b in zip(totals, totals[1:]))
    print(f"[factor] 分层单调: {'是' if monotonic else '否 —— 回查信号是否前视、或数据缺失'}")


if __name__ == "__main__":
    main()
