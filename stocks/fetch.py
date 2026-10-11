"""行情拉取：新浪（akshare stock_zh_a_daily，后复权）-> data/daily/{symbol}.parquet。

一只股票一个文件，文件存在即视为已拉取，天然断点续传。拉完的失败清单写 data/failed.txt。

用法:
    uv run fetch.py --start 2015-01-01            # 拉全部缺的
    uv run fetch.py --start 2015-01-01 --limit 20 # 先试拉 20 只
    uv run fetch.py --start 2015-01-01 --retry    # 只重试 failed.txt 里的
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import akshare as ak
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DAILY_DIR = DATA_DIR / "daily"
UNIVERSE_PATH = DATA_DIR / "universe.parquet"
FAILED_PATH = DATA_DIR / "failed.txt"

KEEP_COLS = [
    "date", "open", "high", "low", "close",
    "volume", "amount", "outstanding_share", "turnover",
]


def fetch_one(symbol: str, start: str, end: str) -> pd.DataFrame:
    """拉一只股票的后复权日线；源无数据返回空表。"""
    df = ak.stock_zh_a_daily(symbol=symbol, start_date=start, end_date=end, adjust="hfq")
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=KEEP_COLS)
    out = df[KEEP_COLS].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = symbol
    return out


def fetch_all(symbols: list[str], start: str, end: str, daily_dir: pathlib.Path,
              tries: int = 3) -> list[str]:
    """逐只拉取，返回最终失败的 symbol 列表。"""
    daily_dir.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    empty: list[str] = []
    for i, sym in enumerate(symbols, 1):
        path = daily_dir / f"{sym}.parquet"
        err = ""
        for attempt in range(1, tries + 1):
            try:
                df = fetch_one(sym, start, end)
                if len(df) == 0:
                    empty.append(sym)
                    print(f"[fetch] {i}/{len(symbols)} {sym} 源无数据")
                    err = ""
                    break
                df.to_parquet(tmp := path.with_suffix(".parquet.tmp"), index=False)
                tmp.replace(path)  # 原子落盘：中途被杀不留下半写文件
                err = ""
                break
            except Exception as ex:  # 源不稳定，重试即可
                err = f"{type(ex).__name__}: {ex}"
                if attempt < tries:
                    time.sleep(1.5 * attempt)
        else:
            failed.append(sym)
        if err:
            print(f"[fetch] {i}/{len(symbols)} {sym} 失败: {err}")
        elif i % 50 == 0 or i == len(symbols):
            print(f"[fetch] 进度 {i}/{len(symbols)} 失败 {len(failed)} 空 {len(empty)}", flush=True)
    print(f"[fetch] 完成: 成功 {len(symbols) - len(failed) - len(empty)}，空 {len(empty)}，失败 {len(failed)}")
    return failed


def load_symbols(start: str, end: str, limit: int | None) -> list[str]:
    """从证券档案取在市股票（退市股本轮不拉）。"""
    df = pd.read_parquet(UNIVERSE_PATH)
    df["ipoDate"] = pd.to_datetime(df["ipoDate"])
    df = df[(df["type"] == "1") & (df["status"] == "1")]
    df = df[df["ipoDate"] <= pd.Timestamp(end)]
    syms = sorted(df["symbol"].tolist())
    return syms[:limit] if limit else syms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--retry", action="store_true", help="只重试 failed.txt")
    args = ap.parse_args()

    if args.retry:
        symbols = [s.strip() for s in FAILED_PATH.read_text().split() if s.strip()]
        print(f"[fetch] 重试 {len(symbols)} 只")
    else:
        symbols = load_symbols(args.start, args.end, args.limit)
        HAVE = {p.stem for p in DAILY_DIR.glob("*.parquet")}
        symbols = [s for s in symbols if s not in HAVE]
        print(f"[fetch] 待拉取 {len(symbols)} 只，区间 {args.start} ~ {args.end}")

    t0 = time.time()
    failed = fetch_all(symbols, args.start, args.end, DAILY_DIR)
    FAILED_PATH.write_text("\n".join(failed))
    print(f"[fetch] 总耗时 {time.time() - t0:.0f}s，失败清单 {FAILED_PATH}")


if __name__ == "__main__":
    sys.exit(main())
