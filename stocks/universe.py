"""全量证券档案（代码 / 名称 / 上市退市日 / 类型 / 状态）。数据源 baostock。

一次性产出 data/universe.parquet，之后所有股票池筛选都读它，不再联网。

用法:
    uv run universe.py
"""
from __future__ import annotations

import pathlib

import baostock as bs
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UNIVERSE_PATH = DATA_DIR / "universe.parquet"

UNIVERSE_COLS = ["code", "code_name", "ipoDate", "outDate", "type", "status"]


def fetch_universe() -> pd.DataFrame:
    """拉 baostock 全量证券档案（原始编码形如 sh.600519）。"""
    print("[universe] baostock: login")
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login 失败: {lg.error_code} {lg.error_msg}")
    try:
        rs = bs.query_stock_basic()
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if rs.error_code != "0":
            raise RuntimeError(f"query_stock_basic 失败: {rs.error_code} {rs.error_msg}")
    finally:
        bs.logout()
    df = pd.DataFrame(rows, columns=rs.fields)[UNIVERSE_COLS]
    print(f"[universe] baostock: 收到 {len(df)} 条")
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """sh.600519 -> sh600519；空日期转 NaT，日期列转 datetime。"""
    out = df.copy()
    out["symbol"] = out["code"].str.replace(".", "", regex=False)
    for col in ("ipoDate", "outDate"):
        out[col] = pd.to_datetime(out[col].replace("", None))
    return out


def pick_stocks(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """挑 type=1（股票）且在 [start, end] 内实际挂牌过的。北交所不在 baostock 覆盖内。"""
    d = df[df["type"] == "1"].copy()
    d = d[d["ipoDate"] <= pd.Timestamp(end)]
    d = d[d["outDate"].isna() | (d["outDate"] >= pd.Timestamp(start))]
    return d.sort_values("symbol").reset_index(drop=True)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = normalize(fetch_universe())
    df.to_parquet(UNIVERSE_PATH, index=False)
    print(f"[universe] 写入 {UNIVERSE_PATH} ({len(df)} 行)")
    for start, end in [("2015-01-01", "2026-10-09")]:
        s = pick_stocks(df, start, end)
        alive = s[s["status"] == "1"]
        dead = s[s["status"] == "0"]
        print(f"[universe] {start}~{end}: 挂牌过 {len(s)} 只（在市 {len(alive)} / 已退 {len(dead)}）")


if __name__ == "__main__":
    main()
