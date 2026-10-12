"""行情拉取：新浪（后复权）-> data/daily/{symbol}.parquet。

一只股票一个文件，文件存在即视为已拉取，天然断点续传。拉完的失败清单写 data/failed.txt。

在市股走 akshare 的 stock_zh_a_daily；**退市股走新浪直连兜底**（见 fetch_one_delisted）。
两条路径口径已由 check.py 的断言⑤逐点对齐。

用法:
    uv run fetch.py --start 2015-01-01            # 拉全部缺的（默认拉到冻结边界）
    uv run fetch.py --start 2015-01-01 --limit 20 # 先试拉 20 只
    uv run fetch.py --start 2015-01-01 --retry    # 只重试 failed.txt 里的
    uv run fetch.py --start 2015-01-01 --delisted # 连同窗口内退市股（255 只）一起拉
    uv run fetch.py --start 2015-01-01 --backfill-share  # 给退市股补 outstanding_share
    uv run fetch.py --allow-future                # 拉到今天（越界数据，只能进研究流程）

冻结边界（协议 §9.1）：默认 --end = 冻结日，且拒绝越过；要拉更新的数据必须显式 --allow-future，
拉下来的新数据只进研究流程，**不得替换冻结快照**（合成面板侧由 store.build_panel 拦截）。
"""
from __future__ import annotations

import argparse
import functools
import json
import pathlib
import sys
import time

import akshare as ak
import pandas as pd
import py_mini_racer
import requests
from akshare.stock.stock_zh_a_sina import (
    hk_js_decode,
    zh_sina_a_stock_hfq_url,
    zh_sina_a_stock_hist_url,
)

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import freeze_guard  # noqa: E402

DATA_DIR = ROOT / "data"
DAILY_DIR = DATA_DIR / "daily"
UNIVERSE_PATH = DATA_DIR / "universe.parquet"
FAILED_PATH = DATA_DIR / "failed.txt"

KEEP_COLS = [
    "date", "open", "high", "low", "close",
    "volume", "amount", "outstanding_share", "turnover",
]


SINA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://finance.sina.com.cn/",
}


@functools.lru_cache(maxsize=1)
def _js_decoder() -> py_mini_racer.MiniRacer:
    """新浪历史行情是一段混淆 JS，解码器建一次就够（每只股票重建要 0.5s 量级）。"""
    d = py_mini_racer.MiniRacer()
    d.eval(hk_js_decode)
    return d


def fetch_one_delisted(symbol: str, start: str, end: str) -> pd.DataFrame:
    """退市股：直连新浪 hist（不复权）× hfq 累积因子，返回与 fetch_one 同构的表。

    akshare 的 stock_zh_a_daily 对退市股必崩：它把 amount 接口的结果交给 demjson.decode，
    而新浪对退市股返回 `(null)`（实测 sh600068/sh600070），解空串报 JSONDecodeError。
    但 hist / hfq 两条接口对退市股都是好的，所以这里跳过 amount 接口自己组装。

    outstanding_share / turnover 新浪对退市股不提供，留 NaN ——
    这两列只被 capacity.py 与正交化控制集用，缺失行在 dropna 处被剔除；
    需要补时跑 tools/backfill_delisted_share.py（baostock 的 volume/turn 反推）。
    复权算法与 akshare 的 hfq 分支逐字对齐，check.py 断言⑤负责证明这一点。
    """
    r = requests.get(zh_sina_a_stock_hist_url.format(symbol), headers=SINA_HEADERS, timeout=30)
    r.raise_for_status()
    rows = _js_decoder().call("d", r.text.split("=")[1].split(";")[0].replace('"', ""))
    if not rows:
        return pd.DataFrame(columns=KEEP_COLS + ["symbol"])
    px = pd.DataFrame(rows)
    px.index = pd.to_datetime(px["date"], errors="coerce").dt.date
    px = px[["open", "high", "low", "close", "volume", "amount"]].astype(float)

    fac_text = requests.get(zh_sina_a_stock_hfq_url.format(symbol),
                            headers=SINA_HEADERS, timeout=30).text
    fac_json = fac_text.split("=", 1)[1]
    fac = pd.DataFrame(json.loads(fac_json[:fac_json.rindex("}") + 1])["data"])
    fac.columns = ["date", "hfq_factor"]
    fac.index = pd.to_datetime(fac["date"]).dt.date

    df = px.join(fac[["hfq_factor"]], how="outer").ffill().astype(float).dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for c in ("open", "high", "low", "close"):
        df[c] = (df[c] * df["hfq_factor"]).round(2)
    df = df[(df.index >= pd.Timestamp(start).date()) & (df.index <= pd.Timestamp(end).date())]
    df["outstanding_share"] = float("nan")
    df["turnover"] = float("nan")
    out = df.drop(columns="hfq_factor").reset_index()
    out.columns = KEEP_COLS
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = symbol
    return out[KEEP_COLS + ["symbol"]]


def fetch_one(symbol: str, start: str, end: str) -> pd.DataFrame:
    """拉一只股票的后复权日线；源无数据返回空表。

    在市股走 akshare（现有分片即此口径，不动）；退市股退回新浪直连。
    """
    try:
        df = ak.stock_zh_a_daily(symbol=symbol, start_date=start, end_date=end, adjust="hfq")
    except Exception:
        return fetch_one_delisted(symbol, start, end)
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


def delisted_symbols(start: str) -> list[str]:
    """窗口内退市的股票（去幸存者偏差要补的那批，255 只）。"""
    df = pd.read_parquet(UNIVERSE_PATH)
    df["outDate"] = pd.to_datetime(df["outDate"])
    return sorted(df[(df["type"] == "1") & (df["status"] == "0")
                     & (df["outDate"] >= pd.Timestamp(start))]["symbol"].tolist())


def backfill_share(daily_dir: pathlib.Path, symbols: list[str]) -> None:
    """退市股补 outstanding_share / turnover：新浪对退市股不提供，用 baostock 反推。

    turn 是百分数，outstanding_share = volume / (turn/100)。在市股实测反推值与新浪
    一致到 0.1%~2% —— 这两列只进 log_mv / turnover_lv / illiq20 三个控制变量，
    量级对就够（不对这些值做跨期比较）。只填全 NaN 的分片，不碰已有值。
    """
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")
    done = skipped = 0
    try:
        for i, sym in enumerate(symbols, 1):
            path = daily_dir / f"{sym}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            if not df["outstanding_share"].isna().all():
                skipped += 1
                continue
            rs = bs.query_history_k_data_plus(
                f"{sym[:2]}.{sym[2:]}", "date,volume,turn",
                start_date=str(df["date"].min().date()), end_date=str(df["date"].max().date()),
                frequency="d", adjustflag="3")
            rows = []
            while (rs.error_code == "0") and rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                print(f"[backfill] {i}/{len(symbols)} {sym} baostock 无数据")
                continue
            b = pd.DataFrame(rows, columns=rs.fields)
            b["date"] = pd.to_datetime(b["date"])
            b["volume"] = pd.to_numeric(b["volume"], errors="coerce")
            b["turn"] = pd.to_numeric(b["turn"], errors="coerce")
            b = b[b["turn"] > 0]
            b["share_bs"] = b["volume"] / (b["turn"] / 100)
            m = df.merge(b[["date", "share_bs"]], on="date", how="left")
            df["outstanding_share"] = m["share_bs"].to_numpy()
            df["turnover"] = df["volume"] / df["outstanding_share"]
            df.to_parquet(tmp := path.with_suffix(".parquet.tmp"), index=False)
            tmp.replace(path)  # 原子落盘，与 fetch_all 同规矩
            done += 1
            if i % 50 == 0:
                print(f"[backfill] {i}/{len(symbols)} 已填 {done} 跳过 {skipped}", flush=True)
    finally:
        bs.logout()
    print(f"[backfill] 完成: 填 {done} 只，跳过（已有值）{skipped} 只")


def load_symbols(start: str, end: str, limit: int | None,
                 include_delisted: bool = False) -> list[str]:
    """从证券档案取在市股票；include_delisted 时连同窗口内退市股一起取。

    退市股只取 outDate >= start 的（255 只）——更早退市的，行情全落在研究区间外，
    拉进来只会让 panel 多 82 个空列。
    """
    df = pd.read_parquet(UNIVERSE_PATH)
    df["ipoDate"] = pd.to_datetime(df["ipoDate"])
    df["outDate"] = pd.to_datetime(df["outDate"])
    listed = (df["type"] == "1") & (df["status"] == "1")
    if include_delisted:
        listed |= (df["type"] == "1") & (df["status"] == "0") & (df["outDate"] >= pd.Timestamp(start))
    df = df[listed]
    df = df[df["ipoDate"] <= pd.Timestamp(end)]
    syms = sorted(df["symbol"].tolist())
    return syms[:limit] if limit else syms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None,
                    help=f"拉取上界；默认 = 冻结边界 {freeze_guard.FREEZE_END}（协议 §9.1）")
    ap.add_argument("--allow-future", action="store_true",
                    help="显式允许拉取冻结边界之后的数据（只进研究流程，不替换冻结快照）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--retry", action="store_true", help="只重试 failed.txt")
    ap.add_argument("--delisted", action="store_true",
                    help="连同窗口内退市股一起拉（去幸存者偏差；退市股走新浪直连兜底）")
    ap.add_argument("--backfill-share", action="store_true",
                    help="给退市股补 outstanding_share/turnover（baostock 反推），拉完退市股后跑")
    args = ap.parse_args()

    if args.end is None:
        end = (pd.Timestamp.today().strftime("%Y-%m-%d") if args.allow_future
               else freeze_guard.FREEZE_END)
    else:
        end = args.end
    if end > freeze_guard.FREEZE_END:
        if not args.allow_future:
            print(f"[fetch] 拒绝：--end {end} 越过冻结边界 {freeze_guard.FREEZE_END}（协议 §9.1）；"
                  f"要拉未来数据加 --allow-future（只进研究流程，不得替换冻结快照）", file=sys.stderr)
            return 2
        print(f"[fetch] [!] 越界拉取（--allow-future）：{end} > 冻结边界 {freeze_guard.FREEZE_END}；"
              f"新数据只能进研究流程，冻结快照的完整性由 freeze_guard.py verify 守")

    if args.backfill_share:
        syms = delisted_symbols(args.start)
        HAVE = {p.stem for p in DAILY_DIR.glob("*.parquet")}
        backfill_share(DAILY_DIR, [s for s in syms if s in HAVE])
        return

    if args.retry:
        symbols = [s.strip() for s in FAILED_PATH.read_text().split() if s.strip()]
        print(f"[fetch] 重试 {len(symbols)} 只")
    else:
        symbols = load_symbols(args.start, end, args.limit, args.delisted)
        HAVE = {p.stem for p in DAILY_DIR.glob("*.parquet")}
        symbols = [s for s in symbols if s not in HAVE]
        print(f"[fetch] 待拉取 {len(symbols)} 只，区间 {args.start} ~ {end}")

    t0 = time.time()
    failed = fetch_all(symbols, args.start, end, DAILY_DIR)
    FAILED_PATH.write_text("\n".join(failed))
    print(f"[fetch] 总耗时 {time.time() - t0:.0f}s，失败清单 {FAILED_PATH}")


if __name__ == "__main__":
    sys.exit(main())
