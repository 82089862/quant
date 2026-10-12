"""读数据的唯一入口。回测 / 因子代码只许通过这里拿数据。

约定:
  - 价格一律是新浪后复权价，可直接算收益率，无除权跳变。
  - 成交额 amount 与成交量 volume 是不复权口径（真实成交），算流动性约束用它们。
  - symbol 形如 sh600519 / sz000001。
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import freeze_guard  # noqa: E402

DATA_DIR = ROOT / "data"
DAILY_DIR = DATA_DIR / "daily"
UNIVERSE_PATH = DATA_DIR / "universe.parquet"


def load_universe() -> pd.DataFrame:
    df = pd.read_parquet(UNIVERSE_PATH)
    for col in ("ipoDate", "outDate"):
        df[col] = pd.to_datetime(df[col])
    return df


def delist_dates(symbols) -> dict[str, pd.Timestamp]:
    """退市日表 {symbol: outDate}，喂给 backtest.run_portfolio 做退市归零。

    不传这张表就是另一种幸存者偏差：退市股行情到退市日就断，面板里的 ffill
    会让它永远停在最后一个价格、永远卖不掉，等于认为退市后还值那个价。
    """
    u = load_universe()
    u = u[u["outDate"].notna() & u["symbol"].isin(set(symbols))]
    return dict(zip(u["symbol"], u["outDate"]))


def load_daily(symbol: str, start: str | None = None, end: str | None = None,
               daily_dir: pathlib.Path = DAILY_DIR) -> pd.DataFrame:
    """一只股票的后复权日线，按日期升序。缺文件返回空表。"""
    path = daily_dir / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path).sort_values("date").reset_index(drop=True)
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def load_panel(symbols: list[str], start: str, end: str,
               daily_dir: pathlib.Path = DAILY_DIR) -> pd.DataFrame:
    """横截面面板: date, symbol, open, high, low, close, volume, amount, turnover。

    5400 个分片逐个读，慢（分钟级）。要反复用时先调 build_panel() 合成年度分片。
    """
    frames = []
    for sym in symbols:
        d = load_daily(sym, start, end, daily_dir)
        if len(d) == 0:
            continue
        frames.append(d.drop(columns=["outstanding_share"]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def trading_dates(symbols: list[str], start: str, end: str,
                  daily_dir: pathlib.Path = DAILY_DIR) -> pd.DatetimeIndex:
    """以覆盖最广的那只股票为准取交易日历（避开停牌缺口）。"""
    best: pd.Series | None = None
    for sym in symbols:
        d = load_daily(sym, start, end, daily_dir)
        if best is None or len(d) > len(best):
            best = d["date"]
    if best is None:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(best)


def build_panel(start: str, end: str, out_path: pathlib.Path | None = None) -> pathlib.Path:
    """把所有分片合成一个 parquet，供横截面查询。走 DuckDB，不经过 pandas。

    end 越过冻结边界（协议 §9.1）时：产物改名带完整日期，并**拒绍写冻结快照**——
    越界数据只能进研究流程，否则冻结判据会在不知情的情况下读到未来数据。
    """
    import duckdb

    if end > freeze_guard.FREEZE_END:
        frozen = freeze_guard.frozen_panel_path()
        if out_path is not None and pathlib.Path(out_path).resolve() == frozen.resolve():
            raise ValueError(
                f"拒绍：越界数据（end={end} > 冻结边界 {freeze_guard.FREEZE_END}）不得写冻结快照 "
                f"{frozen.name}；改用 {freeze_guard.research_panel_path(start, end).name}")
        out = (pathlib.Path(out_path) if out_path is not None
               else freeze_guard.research_panel_path(start, end))
        print(f"[store] ⚠ 越界数据（{end} > 冻结边界 {freeze_guard.FREEZE_END}）→ 写研究面板 "
              f"{out.name}，不替换冻结快照（协议 §9.1）")
    else:
        out = (pathlib.Path(out_path) if out_path is not None
               else DATA_DIR / f"panel_{start[:4]}_{end[:4]}.parquet")
    src = (DAILY_DIR / "*.parquet").as_posix()
    print(f"[store] 合成面板 {start} ~ {end}")
    duckdb.sql(f"""
        copy (
            select * from read_parquet('{src}')
            where date >= date '{start}' and date <= date '{end}'
            order by date, symbol
        ) to '{out.as_posix()}' (format parquet, compression zstd)
    """)
    n = duckdb.sql(f"select count(*) from read_parquet('{out.as_posix()}')").fetchone()[0]
    print(f"[store] 写入 {out} ({n:,} 行)")
    return out
