"""指数日线拉取（新浪）-> data/index/{symbol}.parquet。

股票走 fetch.py（新浪后复权）；指数没有复权概念、字段也没有流通股本/换手，
所以落单独目录，不混进 data/daily/。字段与股票对齐到同一套英文名，
只是没有 outstanding_share / turnover / amount（新浪指数接口只给到成交量）。

用法:
    uv run stocks/index_fetch.py                         # 拉默认 5 个宽基指数
    uv run stocks/index_fetch.py --symbols sh000300      # 指定
    uv run stocks/index_fetch.py --force                 # 重拉
"""
from __future__ import annotations

import argparse
import pathlib
import time

import akshare as ak
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
INDEX_DIR = ROOT / "data" / "index"
DEFAULT_SYMBOLS = ["sh000300", "sh000905", "sh000016", "sh000001", "sz399006"]
KEEP_COLS = ["date", "open", "high", "low", "close", "volume"]


def fetch_one(symbol: str) -> pd.DataFrame:
    """拉一个指数的全部日线；源无数据返回空表。"""
    df = ak.stock_zh_index_daily(symbol=symbol)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=KEEP_COLS)
    out = df[KEEP_COLS].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = symbol
    return out.sort_values("date").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="拉指数日线到 stocks/data/index/")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--start", default="2015-01-01", help="只保留该日期之后的行")
    ap.add_argument("--force", action="store_true", help="已存在也重拉")
    args = ap.parse_args()

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    failed: list[str] = []

    for i, sym in enumerate(symbols, 1):
        path = INDEX_DIR / f"{sym}.parquet"
        if path.exists() and not args.force:
            print(f"[index] {i}/{len(symbols)} {sym} 已有，跳过")
            continue
        t0 = time.time()
        try:
            df = fetch_one(sym)
        except Exception as ex:
            failed.append(sym)
            print(f"[index] {i}/{len(symbols)} {sym} 失败: {type(ex).__name__}: {ex}")
            continue
        if len(df) == 0:
            failed.append(sym)
            print(f"[index] {i}/{len(symbols)} {sym} 源无数据")
            continue
        df = df[df["date"] >= pd.Timestamp(args.start)].reset_index(drop=True)
        df.to_parquet(tmp := path.with_suffix(".parquet.tmp"), index=False)
        tmp.replace(path)  # 原子落盘
        print(f"[index] {i}/{len(symbols)} {sym} {len(df)} 行 "
              f"{df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d} {time.time() - t0:.1f}s")

    print(f"[index] 完成，失败 {len(failed)}: {failed}")


if __name__ == "__main__":
    main()
