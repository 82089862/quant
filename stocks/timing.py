"""市场择时策略池（指数级），思路借自 QuantsPlaybook 的券商金工复现，核心代码自写。

数据只有指数 OHLCV —— 我们 panel 里没有指数，先跑 stocks/index_fetch.py 补。

纳入的策略（都是公开研报的方法，本文件只实现规则、不引第三方框架）：
  rsrs       光大证券 RSRS：滚动 window 日 high/low 回归 beta -> zscore -> ±0.7 阈值
  rsrs_r2    中金 QRS 的修正版：beta 乘 R² 再做同一套 zscore 与阈值
  icu        中泰证券 ICU 均线：Siegel 重复中位数稳健回归，收盘价穿越即持仓
  icu2       同上，快慢双均线（快线上穿慢线持仓）
  llt        广发证券 低延迟趋势线：LLT 递推线，收盘价在线上方持仓
  alligator  招商证券 鳄鱼线：SMA(13/8/5) 各延迟 8/5/3，多头排列新形成即持仓
  tsharpe    滚动夏普简化版（非 Whitelaw 回归口径），滚动 N 日夏普 > 0 持仓

口径与边界：
  - 信号用 T 日收盘算，run_single 在 T+1 开盘执行（backtest.py 的约定）。
  - 只做多、满仓/空仓，不做空 —— 指数做空的成本与可得性没有数据，不假设。
  - 研究阶段 lot_size=1，否则指数点位会把资金 floor 成 0 股（README 回测陷阱 3）。

汇总表固定写 `stocks/out/timing_pool_summary.csv`（入库的事实源，见 import_screen.py）。

用法:
    uv run stocks/timing.py --symbol sh000300
    uv run stocks/timing.py --symbol sh000300 --strategies rsrs,icu,llt
    uv run stocks/timing.py --symbols sh000300,sh000905,sh000016,sh000001,sz399006         --out stocks/out/timing_pool.csv                 # 净值曲线，汇总表另存 _summary.csv
    uv run stocks/import_screen.py load                   # 汇总表镜像进 MySQL
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

import backtest

ROOT = pathlib.Path(__file__).resolve().parent
INDEX_DIR = ROOT / "data" / "index"
OUT_DIR = ROOT / "out"

TRADING_DAYS = 244
ALL_NAMES = ("rsrs", "rsrs_r2", "icu", "icu2", "llt", "alligator", "tsharpe")


def log(msg: str) -> None:
    print(f"[timing] {msg}", flush=True)


def load_index(symbol: str) -> pd.DataFrame:
    """读指数日线。没有就报错并提示先跑 index_fetch.py。"""
    path = INDEX_DIR / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在，先跑 uv run stocks/index_fetch.py")
    df = pd.read_parquet(path).sort_values("date").reset_index(drop=True)
    log(f"{symbol}: {len(df)} 行 {df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}")
    return df


def siegel_ma(y: np.ndarray) -> float:
    """Siegel 重复中位数回归在窗口末点的拟合值（ICU 均线核心，替代 scipy.siegelslopes）。

    slope = 先对每个 i 求 (y_j-y_i)/(j-i) 的中位数，再对这些中位数取中位数；
    intercept = (y - slope*x) 的中位数；最后外推到窗口最后一天。
    """
    n = len(y)
    x = np.arange(n, dtype=float)
    slopes = (y[None, :] - y[:, None]) / (x[None, :] - x[:, None])
    np.fill_diagonal(slopes, np.nan)
    slope = float(np.median(np.nanmedian(slopes, axis=1)))
    intercept = float(np.median(y - slope * x))
    return intercept + slope * (n - 1)


def sig_rsrs(d: pd.DataFrame, window: int = 18, zwin: int = 600,
             upper: float = 0.7, lower: float = -0.7, r2: bool = False) -> pd.Series:
    """RSRS / QRS：滚动 high-low 回归 beta（相关系数×标准差比的等价形式），再做 zscore。

    r2=True 用 beta×corr²（中金修正版），抑制高波动期虚高的 beta。
    阈值外给 ±1，阈值内维持前值；只做多，所以把 -1 压成 0。
    """
    high, low = d["high"], d["low"]
    corr = high.rolling(window).corr(low)
    beta = high.rolling(window).std() * corr / low.rolling(window).std()
    stat = beta * corr ** 2 if r2 else beta
    z = (stat - stat.rolling(zwin).mean()) / stat.rolling(zwin).std()
    raw = pd.Series(np.nan, index=d.index)
    raw[z > upper] = 1.0
    raw[z < lower] = -1.0
    return raw.ffill().clip(lower=0).fillna(0.0)


def sig_icu(d: pd.DataFrame, n: int = 20, slow: int = 0) -> pd.Series:
    """ICU 均线：Siegel 稳健回归均线；slow>0 时用「快线上穿慢线」的双均线口径。"""
    fast = d["close"].rolling(n).apply(siegel_ma, raw=True)
    line = fast.rolling(slow).mean() if slow else fast
    return (d["close"] > line).astype(float)


def sig_llt(d: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    """低延迟趋势线：LLT 递推线，收盘在线上方持仓。alpha 越小越平滑。"""
    p = d["close"].to_numpy(dtype=float)
    n = len(p)
    v = np.full(n, np.nan)
    if n > 2:
        v[0], v[1] = p[0], p[1]
        a = alpha
        for i in range(2, n):
            v[i] = ((a - a * a / 4) * p[i] + (a * a / 2) * p[i - 1]
                    - (a - 3 * a * a / 4) * p[i - 2]
                    + 2 * (1 - a) * v[i - 1] - (1 - a) ** 2 * v[i - 2])
    return pd.Series(p > v, index=d.index, dtype=float)


def sig_alligator(d: pd.DataFrame, periods: tuple[int, int, int] = (13, 8, 5),
                  lag: tuple[int, int, int] = (8, 5, 3)) -> pd.Series:
    """鳄鱼线：三条均线各延迟 lag，多头排列（颚<齿<唇）新形成进场，空头排列新形成离场。"""
    lines = np.column_stack([d["close"].rolling(p).mean().shift(l).to_numpy()
                             for p, l in zip(periods, lag)])
    ok = ~np.isnan(lines).any(axis=1)
    bull = ok & (np.diff(lines, axis=1) > 0).all(axis=1)
    bear = ok & (np.diff(lines, axis=1) < 0).all(axis=1)
    raw = pd.Series(np.nan, index=d.index)
    raw[bull & ~np.r_[False, bull[:-1]]] = 1.0   # 今天刚形成多头排列
    raw[bear & ~np.r_[False, bear[:-1]]] = -1.0
    return raw.ffill().clip(lower=0).fillna(0.0)


def sig_tsharpe(d: pd.DataFrame, window: int = 60) -> pd.Series:
    """滚动夏普：近 window 日收益均值/标准差 > 0 持仓。简化版，非 Whitelaw 回归口径。"""
    ret = np.log(d["close"]).diff()
    sharpe = ret.rolling(window).mean() / ret.rolling(window).std()
    return (sharpe > 0).astype(float)


def perf(equity: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """从 equity 曲线算绩效。费用已在 run_single 内部扣。"""
    eq = equity["equity"].to_numpy(dtype=float)
    ret = pd.Series(eq).pct_change().fillna(0.0)
    years = len(eq) / TRADING_DAYS
    dd = 1 - eq / np.maximum.accumulate(eq)
    std = ret.std()
    return {
        "累计收益": eq[-1] / eq[0] - 1,
        "年化": (eq[-1] / eq[0]) ** (1 / years) - 1 if years > 0 else np.nan,
        "最大回撤": float(dd.max()),
        "夏普": float(ret.mean() / std * np.sqrt(TRADING_DAYS)) if std else np.nan,
        "交易笔数": len(trades),
        "持仓占比": float((equity["shares"] > 0).mean()),
        # 「现金不足」是高价日补仓失败的零头，不是涨跌停挡住，分开报免得误读
        "涨跌停被挡": int(equity["blocked"].isin(["跌停不卖", "涨停不买"]).sum()),
        "补仓失败": int((equity["blocked"] == "现金不足").sum()),
    }


def timing_surrogate(pos: pd.Series, d: pd.DataFrame, split: str, n: int = 300,
                     seed: int = 20261012) -> float:
    """循环平移 surrogate：测试段收益在零分布里的分位（沿用 leadlag.py 的噪声底口径）。

    平移保住策略持仓的自相关与仓位占比，只破坏「哪天持仓」与「哪天上涨」的对应关系。
    分位高 = 择时踩点不是随机撞上的；分位 ≈ 0.5 = 与随机持平，别当有效。
    """
    p = np.nan_to_num(pos.reindex(d.index).to_numpy(dtype=float))
    r = d["close"].pct_change().fillna(0.0).to_numpy(dtype=float)
    test = np.asarray(d.index > pd.Timestamp(split))

    def cum(pv: np.ndarray) -> float:
        strat = np.r_[0.0, pv[:-1] * r[1:]]        # T 日信号、T+1 执行
        return float(np.prod(1 + strat[test]) - 1)

    real = cum(p)
    rng = np.random.default_rng(seed)
    nulls = np.array([cum(np.roll(p, int(k))) for k in rng.integers(1, len(p), n)])
    return float((nulls < real).mean())


def row_of(name: str, equity: pd.DataFrame, trades: pd.DataFrame, split: str) -> dict:
    """一行绩效：全样本 + 训练/测试两段。分段是把同一条曲线按日期切开，不重跑回测。"""
    e = equity.set_index("date")
    t = trades.set_index("date") if len(trades) else pd.DataFrame(index=pd.DatetimeIndex([]))
    cut = pd.Timestamp(split)
    train_e, test_e = e.index <= cut, e.index > cut
    train_t, test_t = t.index <= cut, t.index > cut
    full, tr, te = perf(e, t), perf(e[train_e], t[train_t]), perf(e[test_e], t[test_t])
    return {"策略": name,
            "全累计": full["累计收益"], "全夏普": full["夏普"], "全回撤": full["最大回撤"],
            "训练累计": tr["累计收益"], "训练夏普": tr["夏普"],
            "测试累计": te["累计收益"], "测试夏普": te["夏普"],
            "交易笔数": full["交易笔数"], "涨跌停被挡": full["涨跌停被挡"]}


def build_strategies(names: list[str]) -> dict:
    """策略名 -> 信号函数的显式映射（不用装饰器、不用反射）。"""
    table = {
        "rsrs": lambda d: sig_rsrs(d),
        "rsrs_r2": lambda d: sig_rsrs(d, r2=True),
        "icu": lambda d: sig_icu(d),
        "icu2": lambda d: sig_icu(d, n=20, slow=120),
        "llt": lambda d: sig_llt(d),
        "alligator": lambda d: sig_alligator(d),
        "tsharpe": lambda d: sig_tsharpe(d),
    }
    unknown = [n for n in names if n not in table]
    if unknown:
        raise ValueError(f"未知策略 {unknown}，可选 {list(table)}")
    return {n: table[n] for n in names}


def run_one(symbol: str, names: list[str], cash0: float, split: str,
            surrogate: int) -> tuple[list[dict], dict[str, pd.Series]]:
    """跑一个指数的全部策略，返回（汇总行，净值曲线）。"""
    df = load_index(symbol)
    d = df.set_index("date")
    rows: list[dict] = []
    curves: dict[str, pd.Series] = {}
    for name, fn in build_strategies(names).items():
        equity, trades = backtest.run_single(df, fn(d), symbol, cash0=cash0, lot_size=1)
        row = {"指数": symbol, **row_of(name, equity, trades, split)}
        row["测试分位"] = timing_surrogate(fn(d), d, split, surrogate)
        rows.append(row)
        curves[f"{symbol}_{name}"] = equity.set_index("date")["equity"] / cash0
        log(f"{symbol} {name}: 训练 {row['训练累计']:+.2%} -> 测试 {row['测试累计']:+.2%} "
            f"（全 {row['全累计']:+.2%}，夏普 {row['全夏普']:.2f}，"
            f"{row['交易笔数']} 笔，surrogate 分位 {row['测试分位']:.2f}）")

    bench = d["close"] / d["close"].iloc[0]
    curves[f"{symbol}_买入持有"] = bench
    rows.append({"指数": symbol, **row_of(
        "买入持有",
        pd.DataFrame({"date": bench.index, "equity": bench.to_numpy() * cash0,
                      "shares": 1.0, "blocked": ""}),
        pd.DataFrame(columns=["date"]), split), "测试分位": np.nan})
    return rows, curves


def main() -> None:
    ap = argparse.ArgumentParser(description="指数择时策略池")
    ap.add_argument("--symbol", default="sh000300", help="单个指数")
    ap.add_argument("--symbols", default="", help="多个指数，逗号分隔；给了就覆盖 --symbol")
    ap.add_argument("--strategies", default=",".join(ALL_NAMES))
    ap.add_argument("--cash0", type=float, default=1_000_000.0)
    ap.add_argument("--surrogate", type=int, default=300,
                    help="循环平移零分布次数，测「择时踩点是否胜过随机」")
    ap.add_argument("--split", default="2020-12-31",
                    help="训练/测试分界（该日含在训练段）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    symbols = [s.strip() for s in (args.symbols.split(",") if args.symbols else [args.symbol]) if s.strip()]
    names = [n for n in args.strategies.split(",") if n]

    all_rows: list[dict] = []
    all_curves: dict[str, pd.Series] = {}
    for symbol in symbols:
        rows, curves = run_one(symbol, names, args.cash0, args.split, args.surrogate)
        all_rows += rows
        all_curves |= curves

    out = pd.DataFrame(all_rows).set_index(["指数", "策略"])
    print(out.to_string(float_format=lambda v: f"{v:.4f}"))

    n = len(all_rows) - len(symbols)          # 买持不参与检验
    thresh = 1 - 0.05 / n
    passed = out[out["测试分位"] >= thresh]
    log(f"检验 {n} 次（{len(symbols)} 指数 × {len(names)} 策略），"
        f"Bonferroni 门槛 测试分位 >= {thresh:.4f}，通过 {len(passed)} 个")
    if len(passed):
        print(passed.to_string(float_format=lambda v: f"{v:.4f}"))

    summary_path = pathlib.Path(args.out).with_name(pathlib.Path(args.out).stem + "_summary.csv")         if args.out else OUT_DIR / "timing_pool_summary.csv"
    OUT_DIR.mkdir(exist_ok=True)
    out.reset_index().to_csv(summary_path, index=False)   # 入库的事实源（import_screen.py 读它）
    log(f"汇总表 -> {summary_path}")
    if args.out:
        pd.DataFrame(all_curves).to_csv(args.out)
        log(f"净值曲线 -> {args.out}")
    log("结束")


if __name__ == "__main__":
    main()
