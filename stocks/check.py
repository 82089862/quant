"""数据层自检。断言不过 = 上游源变了，或代码有 bug。回测跑出怪数时先跑这个。

用法:
    uv run check.py
    uv run check.py --full     # 再扫本地全部分片（慢，分钟级）
"""
from __future__ import annotations

import argparse
import pathlib

import akshare as ak
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DAILY_DIR = DATA_DIR / "daily"

# 用作基准的股票：从上到下无借壳、无 ST、无停牌潮的干净标的
BASELINE = "sh600519"
CHECK_RANGE = ("2015-01-01", "2026-10-09")
UP_DOWN_LIMIT = 0.1001  # 主板涨跌停 10% + 浮点容差


def pull(symbol: str, adjust: str, start: str = CHECK_RANGE[0], end: str = CHECK_RANGE[1]) -> pd.DataFrame:
    return ak.stock_zh_a_daily(symbol=symbol, start_date=start, end_date=end, adjust=adjust)


def check_reproducible() -> float:
    """同一段行情，不同起始日拉，重叠部分必须逐点相同。否则后复权不是绝对基准，无法增量。"""
    a = pull(BASELINE, "hfq", "2015-01-01")
    b = pull(BASELINE, "hfq", "2024-01-01")
    m = a.merge(b, on="date", suffixes=("_a", "_b"))
    assert len(m) > 500, f"重叠区间太短: {len(m)}"
    diff = float((m["close_a"] - m["close_b"]).abs().max())
    assert diff == 0.0, f"后复权不可复现，最大差 {diff}"
    print(f"[check] ① 后复权可复现: 重叠 {len(m)} 行, 最大差 {diff}")
    return diff


def check_factor_consistency() -> None:
    """hfq 价 / 不复权价 必须等于 hfq-factor 表在该日的累积因子。"""
    raw = pull(BASELINE, "").set_index("date")["close"]
    hfq = pull(BASELINE, "hfq").set_index("date")["close"]
    fac = pull(BASELINE, "hfq-factor").set_index("date")["hfq_factor"].sort_index()
    raw.index = pd.to_datetime(raw.index)
    hfq.index = pd.to_datetime(hfq.index)
    fac.index = pd.to_datetime(fac.index)
    m = pd.concat([raw.rename("raw"), hfq.rename("hfq")], axis=1).dropna()
    m["ratio"] = m["hfq"] / m["raw"]
    last_day = m.index[-1]
    expected = float(fac[fac.index <= last_day].iloc[-1])
    got = float(m["ratio"].iloc[-1])
    # hfq 价只保留 2 位小数，量化误差 rel ~1e-6，故用相对容差
    assert abs(got - expected) / expected < 1e-5, f"因子不一致: ratio={got} factor表={expected} @ {last_day}"
    print(f"[check] ② 复权自洽: hfq/raw={got:.8f} == 因子表 {expected:.8f} @ {last_day}")


def check_price_band() -> None:
    """后复权日收益必须落在涨跌停区间内。残留除权跳变会冲出去。"""
    h = pull(BASELINE, "hfq").set_index("date")["close"]
    h.index = pd.to_datetime(h.index)
    r = h.pct_change().dropna()
    bad = r[(r > UP_DOWN_LIMIT) | (r < -UP_DOWN_LIMIT)]
    assert len(bad) == 0, f"发现 {len(bad)} 个越界日收益: {bad.head(5).to_dict()}"
    print(f"[check] ③ 收益区间: {len(r)} 个交易日全部在 ±{UP_DOWN_LIMIT:.2%} 内（min {r.min():.4f} / max {r.max():.4f}）")


def check_shards(full: bool) -> None:
    """本地分片结构完整：日期升序无重复、关键列无 NaN、收盘价为正。"""
    paths = sorted(DAILY_DIR.glob("*.parquet"))
    assert paths, f"{DAILY_DIR} 下没有分片，先跑 fetch.py"
    if not full:
        paths = paths[:20]
    for p in paths:
        df = pd.read_parquet(p)
        assert len(df) > 0, f"{p.name} 空表"
        assert df["date"].is_monotonic_increasing, f"{p.name} 日期非升序"
        assert not df["date"].duplicated().any(), f"{p.name} 日期重复"
        assert df["close"].gt(0).all(), f"{p.name} 存在非正收盘价"
        assert df[["open", "high", "low", "close"]].notna().all().all(), f"{p.name} 价格列有 NaN"
    total = len(list(DAILY_DIR.glob("*.parquet")))
    print(f"[check] ④ 分片: 抽检 {len(paths)} / 现存 {total} 个，结构正常")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    check_reproducible()
    check_factor_consistency()
    check_price_band()
    check_shards(args.full)
    print("[check] 全部通过")


if __name__ == "__main__":
    main()
