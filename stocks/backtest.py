"""日频单标的回测引擎：T 日收盘出目标仓位，T+1 开盘成交。

四条规则（缺一条回测就是假的）:
  1. T 收盘信号 -> T+1 开盘价成交。不用 T 收盘价成交，那是前视。
  2. 开盘价触及涨停不买、触及跌停不卖。
  3. 当日买入当日不可卖（T+1）：同一开盘时点先卖后买。
  4. 费用双边收：佣金（有最低值）+ 卖出印花税 + 过户费。

已知简化（够用，但必须知道）:
  - 不区分 ST（5% 涨跌幅）与新股上市首日（无涨跌幅）。历史 ST 状态无法从当前源还原。
  - 停牌日在该股日线里直接缺失，循环自动跳过，持仓按上一个收盘价估值，不交易。
  - 涨跌停按收益率判定而非价格取整，因为价格是后复权价、取整到分没有意义。

价格口径：input 的 open/high/low/close 必须是后复权价（store.load_daily 的输出）。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# 涨跌停价取整到分造成的最大偏差：低价股 round(1.01*1.1,2)=1.11 只有 9.90%
LIMIT_TOL = 0.0015


def price_limit_pct(symbol: str) -> float:
    """涨跌停幅度。创业板/科创板 20%，主板 10%。北交所不在数据池内。"""
    code = symbol[2:] if symbol[:2] in ("sh", "sz") else symbol
    return 0.20 if code.startswith(("300", "301", "688", "689")) else 0.10


def run_single(daily: pd.DataFrame, target: pd.Series, symbol: str,
               cash0: float = 1_000_000.0,
               commission: float = 0.00025, commission_min: float = 5.0,
               stamp_tax_sell: float = 0.0005, transfer_fee: float = 0.00001,
               lot_size: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """跑单只股票。target 是 index=date、值=目标权重(0~1) 的 Series。

    返回 (equity, trades)：
      equity: date, cash, shares, close, equity, blocked
      trades: date, side, shares, price, gross, fee
    blocked 记录因涨跌停没做成的动作，回测报告里要看这个数。
    """
    d = daily.sort_values("date").reset_index(drop=True)
    # NaN = 该日不调仓（信号只在特定日子给出时用这个）
    weight = target.reindex(d["date"]).to_numpy(dtype=float)
    dates = d["date"].to_numpy()
    opens = d["open"].to_numpy(dtype=float)
    closes = d["close"].to_numpy(dtype=float)
    pct = price_limit_pct(symbol)

    cash = float(cash0)
    shares = 0
    equity_rows: list[tuple] = []
    trade_rows: list[tuple] = []

    for i in range(len(d)):
        blocked = ""
        # ---- 在 i 日开盘执行 i-1 日收盘产生的目标 ----
        if i > 0 and not math.isnan(weight[i - 1]):
            preclose = closes[i - 1]
            px = opens[i]
            equity_prev = cash + shares * preclose
            want = math.floor(weight[i - 1] * equity_prev / px / lot_size) * lot_size
            ret = px / preclose - 1.0

            if want < shares:  # 先卖
                n = shares - want
                if ret <= -(pct - LIMIT_TOL):
                    n, blocked = 0, "跌停不卖"
                if n > 0:
                    gross = n * px
                    fee = max(gross * commission, commission_min) + gross * stamp_tax_sell + gross * transfer_fee
                    cash += gross - fee
                    shares -= n
                    trade_rows.append((dates[i], "sell", n, px, gross, fee))

            elif want > shares:  # 后买
                n = want - shares
                if ret >= pct - LIMIT_TOL:
                    n, blocked = 0, "涨停不买"
                while n > 0:  # 现金不够就按手数缩
                    gross = n * px
                    fee = max(gross * commission, commission_min) + gross * transfer_fee
                    if gross + fee <= cash:
                        break
                    n -= lot_size
                if n > 0:
                    gross = n * px
                    fee = max(gross * commission, commission_min) + gross * transfer_fee
                    cash -= gross + fee
                    shares += n
                    trade_rows.append((dates[i], "buy", n, px, gross, fee))
                elif not blocked:
                    blocked = "现金不足"

        equity_rows.append((dates[i], cash, shares, closes[i], cash + shares * closes[i], blocked))

    equity = pd.DataFrame(equity_rows, columns=["date", "cash", "shares", "close", "equity", "blocked"])
    trades = pd.DataFrame(trade_rows, columns=["date", "side", "shares", "price", "gross", "fee"])
    return equity, trades


def hold_all(daily: pd.DataFrame) -> pd.Series:
    """买入持有信号：只在第一个交易日收盘给满仓，之后不调仓。

    注意 weight=1.0 每天都给 = 每日再平衡，不是买入持有。
    """
    w = pd.Series(float("nan"), index=daily["date"])
    if len(w) > 0:
        w.iloc[0] = 1.0
    return w


def portfolio_value(cash: float, shares: np.ndarray, prices: np.ndarray) -> float:
    """组合总市值。停牌/未上市价为 NaN，按 0 计。

    不能直接写 (shares * prices).sum()：0 股 × NaN = NaN，会把整个组合净值污染成 NaN，
    之后每一次目标股数都算不出来，回测全程空仓——这个 bug 不会报错，只会静默给 0 收益。
    """
    return cash + float((shares * np.nan_to_num(prices, nan=0.0)).sum())


def run_portfolio(panel: pd.DataFrame, rebalance: pd.DataFrame, cash0: float = 1_000_000.0,
                  commission: float = 0.00025, commission_min: float = 5.0,
                  stamp_tax_sell: float = 0.0005, transfer_fee: float = 0.00001,
                  lot_size: int = 100,
                  delist_dates: dict[str, pd.Timestamp] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """多标的组合。panel 是长表(date/symbol/open/close)。

    rebalance: index = **执行日**（开盘成交那天），columns = symbol，值 = 目标权重。
    NaN/缺失列 = 权重 0（清仓）；整行全 NaN = 该日不调仓。调用方负责把信号日 shift 到执行日。

    delist_dates: {symbol: 退市日}。**必须传，否则幸存者偏差换皮**：退市股行情到退市日就断，
    下面的 `closes.ffill()` 会让它永远停在最后一个价格、永远卖不掉——那等于认为退市后还值那个价。
    传了则退市日之后 open/close 置 0：持仓价值归零（损失全额体现），并以 0 价强平（不拦跌停），
    因为退市是规则外事件，不受涨跌停约束。代价：每次强平收 5 元最低佣金（相对本金可忽略）。

    成交规则与 run_single 一致：涨跌停/停牌不成交，同日先卖后买。
    equity 的 blocked 列是当日没成交的笔数——这个数大了，回测和实盘就会分家。
    """
    d = panel[["date", "symbol", "open", "close"]]
    opens = d.pivot(index="date", columns="symbol", values="open").sort_index()
    closes = d.pivot(index="date", columns="symbol", values="close").sort_index()
    closes_ff = closes.ffill()
    if delist_dates:
        for sym, dt in delist_dates.items():
            if sym in closes_ff.columns:
                after = closes_ff.index > dt
                closes_ff.loc[after, sym] = 0.0
                opens.loc[after, sym] = 0.0
    dates = closes.index
    symbols = list(closes.columns)
    sched = rebalance.reindex(columns=symbols).reindex(dates)
    limits = np.array([price_limit_pct(s) for s in symbols])

    cash = float(cash0)
    shares = np.zeros(len(symbols), dtype=float)
    equity_rows: list[tuple] = []
    trade_rows: list[tuple] = []

    for i in range(len(dates)):
        blocked = 0
        if i > 0:
            w = sched.iloc[i]
            if not w.isna().all():
                px = opens.iloc[i].to_numpy(dtype=float)
                pre = closes_ff.iloc[i - 1].to_numpy(dtype=float)
                equity_prev = portfolio_value(cash, shares, pre)
                want_val = w.fillna(0.0).to_numpy(dtype=float) * equity_prev
                with np.errstate(invalid="ignore", divide="ignore"):
                    want = np.floor(want_val / px / lot_size) * lot_size
                want[~np.isfinite(want)] = 0.0
                delta = want - shares

                for j in np.where(delta < 0)[0]:  # 先卖
                    if not math.isfinite(px[j]):
                        blocked += 1
                        continue
                    # px=0 是退市归零，允许成交（卖价 0）且不受涨跌停约束
                    if px[j] > 0 and (pre[j] <= 0 or px[j] / pre[j] - 1 <= -(limits[j] - LIMIT_TOL)):
                        blocked += 1
                        continue
                    n = -delta[j]
                    gross = n * px[j]
                    fee = max(gross * commission, commission_min) + gross * stamp_tax_sell + gross * transfer_fee
                    cash += gross - fee
                    shares[j] -= n
                    trade_rows.append((dates[i], symbols[j], "sell", n, px[j], fee))

                for j in np.where(delta > 0)[0]:  # 后买
                    if not math.isfinite(px[j]) or px[j] <= 0:
                        blocked += 1
                        continue
                    if pre[j] <= 0 or px[j] / pre[j] - 1 >= limits[j] - LIMIT_TOL:
                        blocked += 1
                        continue
                    n = delta[j]
                    while n > 0:
                        gross = n * px[j]
                        fee = max(gross * commission, commission_min) + gross * transfer_fee
                        if gross + fee <= cash:
                            break
                        n -= lot_size
                    if n > 0:
                        gross = n * px[j]
                        fee = max(gross * commission, commission_min) + gross * transfer_fee
                        cash -= gross + fee
                        shares[j] += n
                        trade_rows.append((dates[i], symbols[j], "buy", n, px[j], fee))
                    else:
                        blocked += 1

        equity_rows.append((dates[i], cash, portfolio_value(cash, shares, closes_ff.iloc[i].to_numpy(dtype=float)), blocked))

    equity = pd.DataFrame(equity_rows, columns=["date", "cash", "equity", "blocked"])
    trades = pd.DataFrame(trade_rows, columns=["date", "symbol", "side", "shares", "price", "fee"])
    return equity, trades


if __name__ == "__main__":
    import store

    d = store.load_daily("sh600519", "2015-01-01", "2026-10-09")
    eq, tr = run_single(d, hold_all(d), "sh600519", lot_size=1,
                        commission=0.0, commission_min=0.0, stamp_tax_sell=0.0, transfer_fee=0.0)
    print(f"茅台买入持有 {d['date'].iloc[0]} ~ {d['date'].iloc[-1]}")
    print(f"  期末净值 {eq['equity'].iloc[-1]:,.2f}（本金 1,000,000）")
    print(f"  复权涨幅 {d['close'].iloc[-1] / d['open'].iloc[1] - 1:.2%}")
    print(f"  净值涨幅 {eq['equity'].iloc[-1] / 1_000_000 - 1:.2%}")
