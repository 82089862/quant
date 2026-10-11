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

# 每组上千只等权，本金必须够大：后复权价常有几千上万，1 亿分到 1000 只每只 10 万，
# 才能保证每只都买得进。资金太小会让高价股被 floor 成 0 股，组合失真。
CASH0 = 100_000_000.0


def load_closes(panel_path: pathlib.Path) -> pd.DataFrame:
    """读成 date × symbol 的后复权收盘宽表。"""
    df = duckdb.sql(f"select date, symbol, close from read_parquet('{panel_path.as_posix()}')").df()
    return df.pivot(index="date", columns="symbol", values="close").sort_index()


def load_prices(panel_path: pathlib.Path) -> pd.DataFrame:
    """读成回测引擎要的长表（date/symbol/open/close）。"""
    return duckdb.sql(
        f"select date, symbol, open, close from read_parquet('{panel_path.as_posix()}')"
    ).df()


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


def curve_stats(equity: pd.DataFrame, trades: pd.DataFrame, cash0: float) -> dict:
    """日频净值曲线 -> 绩效。月胜率按自然月末重采样；换手按成交额/平均净值年化。

    月胜率的分辨率：140 个月下二项标准误约 4.2%，95%CI 宽约 ±8%，48% 与 50% 不可分辨。
    最大回撤用日频净值，不用月末净值（月末口径会系统性低估回撤）。
    """
    eq = equity.set_index("date")["equity"]
    m = eq.resample("ME").last().dropna()
    ret = m.pct_change().dropna()
    n = len(ret)
    win = float((ret > 0).mean()) if n else float("nan")
    years = n / 12 if n else 0.0
    gross = float((trades["shares"] * trades["price"]).sum())
    return {
        "累计": float(eq.iloc[-1] / cash0 - 1),
        "年化": float((eq.iloc[-1] / cash0) ** (1 / years) - 1) if years else float("nan"),
        "月胜率": win,
        "月胜率CI": 1.96 * float(np.sqrt(win * (1 - win) / n)) if n else float("nan"),
        "月数": n,
        # 夏普按项目惯例年化（同 timing.py 的 perf）：月度收益均值/标准差 × √12
        "年化夏普": float(ret.mean() / ret.std() * np.sqrt(12)) if ret.std() else float("nan"),
        # 回撤用日频净值（比月末口径更深，不用月末口径会低估）
        "最大回撤": float((eq / eq.cummax() - 1).min()),
        "年换手双边": gross / float(eq.mean()) / years if years and eq.mean() else float("nan"),
        "成交笔数": len(trades),
        "被挡": int(equity["blocked"].sum()),
        "费用": float(trades["fee"].sum()),
    }


def run_layers(fac: pd.DataFrame, panel: pd.DataFrame, dates: pd.DatetimeIndex,
               n_groups: int, cash0: float = CASH0,
               lot_size: int = 1) -> tuple[dict[str, pd.Series], list[dict]]:
    """按 fac 分 n_groups 层，逐层跑组合回测。返回 ({组名: 净值}, [逐层绩效])。"""
    curves: dict[str, pd.Series] = {}
    stats: list[dict] = []
    for g, sched in build_schedule(fac, n_groups, dates).items():
        eq, tr = backtest.run_portfolio(panel, sched, cash0=cash0, lot_size=lot_size)
        curves[f"G{g + 1}"] = eq.set_index("date")["equity"]
        stats.append({"组": f"G{g + 1}", **curve_stats(eq, tr, cash0)})
    return curves, stats


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

    panel = load_prices(panel_path)

    # lot_size=1：分层研究阶段忽略整手约束（整手留到实盘模拟，那时要用不复权价）
    curves, stats = run_layers(fac, panel, dates, args.groups, cash0=CASH0, lot_size=1)
    for i, s in enumerate(stats):
        tag = "因子最小" if i == 0 else "因子最大" if i == len(stats) - 1 else "中间"
        print(f"[factor] 第 {i + 1} 组（{tag}） 累计 {s['累计']:+.2%} 年化 {s['年化']:+.2%} "
              f"月胜率 {s['月胜率']:.1%}±{s['月胜率CI']:.1%} 年化夏普 {s['年化夏普']:.2f} "
              f"回撤 {s['最大回撤']:.1%} 年换手(双边) {s['年换手双边']:.1f}x 被挡 {s['被挡']:,}")

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
