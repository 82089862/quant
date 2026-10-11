"""回测引擎自检：手算对账 + 涨跌停 + 仓位调仓 + 费用。断言不过 = 引擎错了，不许往下走因子。

用法: uv run test_backtest.py
"""
from __future__ import annotations

import math

import pandas as pd

import store
from backtest import hold_all, run_portfolio, run_single

NO_COST = dict(commission=0.0, commission_min=0.0, stamp_tax_sell=0.0, transfer_fee=0.0)


def make(opens: list[float], closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(opens))
    return pd.DataFrame({"date": dates, "open": opens, "high": opens, "low": opens, "close": closes})


def test_buy_and_hold_math() -> None:
    """真实数据买入持有：期末净值必须与手算严格一致，且净值曲线正比于复权价曲线。"""
    d = store.load_daily("sh600519", "2015-01-01", "2026-10-09")
    cash0 = 1_000_000.0
    eq, tr = run_single(d, hold_all(d), "sh600519", cash0=cash0, lot_size=1, **NO_COST)

    buy_px = float(d["open"].iloc[1])
    shares = math.floor(cash0 / buy_px)
    cash_left = cash0 - shares * buy_px
    expect = cash_left + shares * float(d["close"].iloc[-1])
    got = float(eq["equity"].iloc[-1])
    assert abs(got - expect) < 1e-6, f"期末净值 {got:.6f} != 手算 {expect:.6f}"

    # 建仓后（i>=1）净值 - 剩余现金 应严格正比于后复权价（买入持有没有别的现金流）
    lhs = (eq["equity"] - cash_left).to_numpy()[1:]
    rhs = (eq["close"].to_numpy() * shares)[1:]
    assert abs(lhs - rhs).max() < 1e-6, "净值曲线与复权价曲线不成比例"
    assert len(tr) == 1 and tr["side"].iloc[0] == "buy", f"成交笔数异常: {len(tr)}"
    assert (eq["blocked"] == "").all(), "不该有涨跌停阻塞"
    print(f"[test] ① 买入持有对账: 期末 {got:,.2f} == 手算 {expect:,.2f}（{len(eq)} 个交易日，1 笔成交）")


def test_limit_up_blocks_buy() -> None:
    """开盘涨停不许买；次日不再涨停就能买。"""
    d = make(opens=[10.0, 11.0, 11.0], closes=[10.0, 11.0, 11.0])
    target = pd.Series([1.0, 1.0, 1.0], index=d["date"])
    eq, tr = run_single(d, target, "sh600000", cash0=1000.0, lot_size=1, **NO_COST)
    assert eq["blocked"].iloc[1] == "涨停不买", f"第2日应为涨停不买，实际 {eq['blocked'].iloc[1]!r}"
    assert eq["shares"].iloc[1] == 0, "涨停日不该有持仓"
    assert eq["shares"].iloc[2] > 0, "次日应已建仓"
    print("[test] ② 涨停不买: 第2日 blocked，第3日建仓成功")


def test_limit_down_blocks_sell() -> None:
    """开盘跌停不许卖；次日不跌停就卖得掉。"""
    d = make(opens=[10.0, 10.0, 9.0, 9.0], closes=[10.0, 10.0, 9.0, 9.0])
    target = pd.Series([1.0, 0.0, 0.0, 0.0], index=d["date"])
    eq, tr = run_single(d, target, "sh600000", cash0=1000.0, lot_size=1, **NO_COST)
    held = eq["shares"].iloc[1]
    assert held > 0, "第2日应建仓"
    assert eq["blocked"].iloc[2] == "跌停不卖", f"第3日应为跌停不卖，实际 {eq['blocked'].iloc[2]!r}"
    assert eq["shares"].iloc[2] == held, "跌停日持仓不该变"
    assert eq["shares"].iloc[3] == 0, "第4日应已清仓"
    print("[test] ③ 跌停不卖: 第3日 blocked，第4日清仓成功")


def test_target_weight_and_t1_order() -> None:
    """仓位按目标权重调，且同一天不会既买又卖（先卖后买保证 T+1）。"""
    d = make(opens=[10.0] * 5, closes=[10.0] * 5)
    target = pd.Series([1.0, 1.0, 0.5, 0.5, 0.0], index=d["date"])
    eq, tr = run_single(d, target, "sh600000", cash0=1000.0, lot_size=1, **NO_COST)
    # 价格恒定 10、本金 1000 -> 满仓 100 股，半仓 50 股，空仓 0 股
    # i=1 执行 w[0]=1 建仓；i=3 执行 w[2]=0.5 减半；i=4 执行 w[3]=0.5 维持
    assert eq["shares"].tolist() == [0, 100, 100, 50, 50], f"持仓序列异常: {eq['shares'].tolist()}"
    assert abs(eq["equity"].iloc[-1] - 1000.0) < 1e-9, "价格不动时净额应为本金"
    same_day = tr.groupby("date")["side"].nunique()
    assert (same_day <= 1).all(), "同一天不能同时出现买和卖"
    print("[test] ④ 仓位调仓: 0->100->100->50 股持仓序列正确，同日无买卖混单")


def test_cost_math() -> None:
    """费用手算对账：佣金下限 5 元、卖出印花税、过户费。"""
    d = make(opens=[10.0] * 3, closes=[10.0] * 3)
    target = pd.Series([1.0, 0.0, 0.0], index=d["date"])
    eq, tr = run_single(d, target, "sh600000", cash0=10_000.0, lot_size=1,
                        commission=0.00025, commission_min=5.0,
                        stamp_tax_sell=0.0005, transfer_fee=0.00001)

    buy = tr[tr["side"] == "buy"].iloc[0]
    n = int(buy["shares"])
    buy_fee = max(n * 10.0 * 0.00025, 5.0) + n * 10.0 * 0.00001
    sell_fee = max(n * 10.0 * 0.00025, 5.0) + n * 10.0 * 0.0005 + n * 10.0 * 0.00001
    expect_cash = 10_000.0 - n * 10.0 - buy_fee + n * 10.0 - sell_fee

    assert n == 999, f"扣费后应买 999 股，实际 {n}"
    assert abs(buy["fee"] - buy_fee) < 1e-9, f"买入费 {buy['fee']} != 手算 {buy_fee}"
    assert abs(tr[tr["side"] == "sell"].iloc[0]["fee"] - sell_fee) < 1e-9
    assert abs(eq["cash"].iloc[-1] - expect_cash) < 1e-9, f"期末现金 {eq['cash'].iloc[-1]} != 手算 {expect_cash}"
    print(f"[test] ⑤ 费用对账: 999 股，买费 {buy_fee:.4f} 卖费 {sell_fee:.4f}，期末现金 {expect_cash:.4f}")


def test_delisting_zeroes_position() -> None:
    """退市股在退市日之后归零，且能强平（不受跌停拦），blocked 不虚增。

    这条断言存在的理由：不传 delist_dates 时 closes.ffill() 会让退市股永远停在
    最后一个价格、永远卖不掉——那是另一种幸存者偏差，而且不报错。
    """
    dates = pd.bdate_range("2024-01-01", periods=5)
    alive = pd.DataFrame({"date": dates, "symbol": "sh600000", "open": 10.0, "close": 10.0})
    # B 只有前 3 天有行情，第 3 天（dates[2]）后退市
    dead = pd.DataFrame({"date": dates[:3], "symbol": "sh600001",
                         "open": [10.0, 9.0, 8.0], "close": [10.0, 9.0, 8.0]})
    panel = pd.concat([alive, dead], ignore_index=True)
    sched = pd.DataFrame({"sh600000": 0.5, "sh600001": 0.5}, index=dates[[1, 2, 4]])

    eq_zero, tr_zero = run_portfolio(panel, sched, cash0=1000.0, lot_size=1,
                                     delist_dates={"sh600001": dates[2]}, **NO_COST)
    eq_ff, tr_ff = run_portfolio(panel, sched, cash0=1000.0, lot_size=1, **NO_COST)

    assert eq_zero["equity"].iloc[-1] < eq_ff["equity"].iloc[-1], \
        f"退市归零应低于 ffill 估值: {eq_zero['equity'].iloc[-1]} vs {eq_ff['equity'].iloc[-1]}"
    assert abs(eq_zero["equity"].iloc[-1] - 505.0) < 1e-9, \
        f"手算应为 505（现金5 + 卖A 250 + A 25股×10），实际 {eq_zero['equity'].iloc[-1]}"
    sold = tr_zero[(tr_zero["symbol"] == "sh600001") & (tr_zero["side"] == "sell")]
    assert len(sold) == 1 and sold["price"].iloc[0] == 0.0, "退市股应以 0 价被强平一次"
    # 不处理时 open 是 NaN（面板里已无该股），买入被 isfinite 拦下并计入 blocked；
    # 归零版把 open 置 0，强平不被拦，所以当日 blocked 必须是 0。
    assert eq_zero["blocked"].iloc[-1] == 0, "0 价强平当日不该计入 blocked"
    assert eq_ff["blocked"].sum() > eq_zero["blocked"].sum(), \
        "不处理那版应因 NaN 开盘价多吃一次 blocked（退市股没法买回来）"
    print(f"[test] ⑥ 退市归零: 净值 {eq_zero['equity'].iloc[-1]:.2f} vs 不处理 {eq_ff['equity'].iloc[-1]:.2f}，"
          f"0 价强平 1 笔，blocked 未虚增")


def main() -> None:
    test_buy_and_hold_math()
    test_limit_up_blocks_buy()
    test_limit_down_blocks_sell()
    test_target_weight_and_t1_order()
    test_cost_math()
    test_delisting_zeroes_position()
    print("[test] 全部通过")


if __name__ == "__main__":
    main()
