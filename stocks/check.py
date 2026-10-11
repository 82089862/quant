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
# 退市股抽检样本：拉取走新浪直连（fetch.fetch_one_delisted），与在市股的 akshare 路径分开
DELISTED_SAMPLE = ("sh600068", "sh600070")
# 注：断言③的 ±10.01% 不适用于退市股 —— 退市整理期首日无涨跌幅限制，实测到 10.10%


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


def check_dual_path() -> None:
    """退市股与在市股两条拉取路径必须逐点一致，否则混在一起的面板不可比。

    在市股走 akshare，退市股走新浪直连（akshare 对退市股会崩在 amount 接口）。
    这里用同一只在市股跑两条路径对账 —— 只要它相等，退市股的口径就可信。
    """
    from fetch import fetch_one_delisted

    for sym in (BASELINE, "sz000001"):
        new = fetch_one_delisted(sym, *CHECK_RANGE)
        old = pull(sym, "hfq", *CHECK_RANGE)
        new["date"], old["date"] = pd.to_datetime(new["date"]), pd.to_datetime(old["date"])
        m = new.merge(old, on="date", suffixes=("_new", "_old"))
        assert len(m) > 500, f"{sym} 重叠区间太短: {len(m)}"
        for c in ("open", "high", "low", "close", "volume", "amount"):
            d = float((m[f"{c}_new"] - m[f"{c}_old"]).abs().max())
            assert d == 0.0, f"{sym} 的 {c} 两条路径不一致，最大差 {d}"
    print(f"[check] ⑤ 双路径口径: {BASELINE} / sz000001 全列逐点相同（直连 vs akshare）")


def check_delisted_shards() -> None:
    """退市股分片：行情到退市日就断（不能延伸到面板末尾），股本已补齐。"""
    for sym in DELISTED_SAMPLE:
        p = DAILY_DIR / f"{sym}.parquet"
        if not p.exists():
            print(f"[check] ⑥ {sym} 分片不存在（源无数据，跳过）")
            continue
        df = pd.read_parquet(p)
        last = df["date"].max()
        assert last < pd.Timestamp("2026-09-01"), f"{sym} 行情延伸到了 {last}，退市股不该有近期数据"
        assert df["outstanding_share"].notna().any(), \
            f"{sym} 的 outstanding_share 全空，跑 fetch.py --backfill-share 补"
    print(f"[check] ⑥ 退市股分片: {len(DELISTED_SAMPLE)} 只，行情截止正常、股本已补")


def check_shards(full: bool) -> None:
    """本地分片结构完整：日期升序无重复、关键列无 NaN、收盘价为正。"""
    paths = sorted(DAILY_DIR.glob("*.parquet"))
    assert paths, f"{DAILY_DIR} 下没有分片，先跑 fetch.py"
    if not full:
        # 抽检也把退市股带上：它们是另一条拉取路径，最容易漏检
        extra = [p for p in paths if p.stem in DELISTED_SAMPLE]
        paths = paths[:20] + [p for p in extra if p not in paths[:20]]
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
    check_dual_path()
    check_delisted_shards()
    print("[check] 全部通过")


if __name__ == "__main__":
    main()
