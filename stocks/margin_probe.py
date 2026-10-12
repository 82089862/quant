"""融资融券接入前的价值实测：只抓抽样交易日，看有效维度 PR 从 7.5 涨到多少。

判据（唯一）：
  在同一批股票上算 PR（23 个价格特征）vs PR（23 个 + 5 个融资融券特征）。
  涨得多 = 真新维度；涨不动 = 又一批同族数据。

数据被抓到 data/margin_cache/ 缓存，重跑不再打网络。

用法:
    uv run margin_probe.py                      # 抽样日期的每 3 个取 1 个
    uv run margin_probe.py --step 1             # 全部抽样日期（慢，约 20 分钟）
"""
from __future__ import annotations

import argparse
import pathlib
import time

import duckdb
import numpy as np
import pandas as pd

import capacity

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "margin_cache"

# 沪深两市都有的字段才算（SSE 多了融资偿还额，SZSE 多了融券余额，都不通用）
MARGIN_FEATURES = ["m_bal_ratio", "m_buy_int", "m_short_ratio",
                   "m_shortsell_int", "m_bal_chg"]


def fetch_one(day: pd.Timestamp, cache_dir: pathlib.Path,
              sleep: float = 0.2) -> pd.DataFrame:
    """抓一天的沪深两市融资融券明细，缓存成 parquet。返回 date/symbol + 原始字段。"""
    import akshare as ak

    cache = cache_dir / f"{day:%Y%m%d}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    ymd = f"{day:%Y%m%d}"
    frames = []
    for exch, fn, code_col in (
        ("sh", ak.stock_margin_detail_sse, "标的证券代码"),
        ("sz", ak.stock_margin_detail_szse, "证券代码"),
    ):
        for attempt in range(3):
            try:
                d = fn(date=ymd)
                break
            except Exception as e:  # 东财/交易所接口偶发断连
                if attempt == 2:
                    print(f"  [{day:%Y-%m-%d}] {exch} 放弃: {type(e).__name__}")
                    d = None
                else:
                    time.sleep(1.0 + attempt)
        if d is None or len(d) == 0:
            continue
        d = d.rename(columns={code_col: "code"})
        d["symbol"] = exch + d["code"].astype(str).str.zfill(6)
        d["date"] = day
        keep = ["date", "symbol", "融资余额", "融资买入额", "融券余量", "融券卖出量"]
        frames.append(d[[c for c in keep if c in d.columns]])
        time.sleep(sleep)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "symbol", "融资余额", "融资买入额", "融券余量", "融券卖出量"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def fetch_margin(days: list[pd.Timestamp], cache_dir: pathlib.Path) -> pd.DataFrame:
    n_cached = sum((cache_dir / f"{d:%Y%m%d}.parquet").exists() for d in days)
    print(f"[margin] {len(days)} 个交易日，已缓存 {n_cached}，需抓 {len(days) - n_cached}")
    frames = []
    for i, d in enumerate(days, 1):
        frames.append(fetch_one(d, cache_dir))
        if i % 20 == 0 or i == len(days):
            print(f"[margin] {i}/{len(days)}")
    return pd.concat(frames, ignore_index=True)


def margin_features(margin: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """把融资融券原始额换算成与价格特征同量纲的截面特征。"""
    m = margin.merge(panel[["date", "symbol", "close", "volume", "amount",
                            "outstanding_share"]], on=["date", "symbol"], how="inner")
    mv = m["close"] * m["outstanding_share"]
    m["m_bal_ratio"] = m["融资余额"] / mv
    m["m_buy_int"] = m["融资买入额"] / m["amount"]
    m["m_short_ratio"] = m["融券余量"] / m["outstanding_share"]
    m["m_shortsell_int"] = m["融券卖出量"] / m["volume"]
    m = m.sort_values(["symbol", "date"])
    # shortcut: 个别月缺数据时 pct_change 会跨月，非严格月环比。与 margin_factor.py 同口径（那份先改的）
    m["m_bal_chg"] = m.groupby("symbol")["融资余额"].pct_change(fill_method=None)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=float, default=25.0)
    ap.add_argument("--step", type=int, default=3, help="抽样日期里每 N 个取 1 个")
    args = ap.parse_args()

    panel_path = DATA_DIR / "panel_2015_2026.parquet"
    all_days = capacity.sample_dates(panel_path, args.sample)
    days = all_days[::args.step]
    print(f"[margin] 抽样 {len(all_days)} 个交易日，每 {args.step} 取 1 -> {len(days)} 个")

    margin = fetch_margin(days, CACHE_DIR)

    # 价格面板：与融资融券完全相同的日期集，才能同一批股票上对比
    df = capacity.build_features(panel_path, args.sample, dates=days)
    lit = ", ".join(f"date '{pd.Timestamp(d).date()}'" for d in days)
    raw = duckdb.sql(f"""
        select date, symbol, close, volume, amount, outstanding_share
        from read_parquet('{panel_path.as_posix()}')
        where date in ({lit})
    """).df()

    mf = margin_features(margin, raw)
    keep = ["date", "symbol"] + MARGIN_FEATURES
    merged = df.merge(mf[keep], on=["date", "symbol"], how="inner").dropna(
        subset=MARGIN_FEATURES)
    cover = len(merged) / len(df)
    print(f"[margin] 覆盖: 面板 {len(df):,} 行 -> 有融资融券 {len(merged):,} 行 ({cover:.1%})")

    base = [f"r_{c}" for c in capacity.FEATURES]
    # 融资融券特征在匹配子集内重新做截面 rank，与价格特征同尺度
    g = merged.groupby("date", sort=False)
    for c in MARGIN_FEATURES:
        merged[f"r_{c}"] = g[c].rank(pct=True)
    marg_cols = [f"r_{c}" for c in MARGIN_FEATURES]

    x_base = merged[base].to_numpy(dtype=np.float64)
    x_marg = merged[marg_cols].to_numpy(dtype=np.float64)
    x_all = np.column_stack([x_base, x_marg])

    pr_base = capacity.pr_of(x_base)
    pr_all = capacity.pr_of(x_all)

    corr_b = np.corrcoef(x_base, rowvar=False)
    evals, evecs = np.linalg.eigh(corr_b)
    order = np.argsort(evals)[::-1]
    scores = (x_base - x_base.mean(0)) @ evecs[:, order]
    basis = scores[:, :capacity.effective_dims(evals)["n90"]]
    probe = {c: capacity.basis_r2(basis, x_marg[:, k] - x_marg[:, k].mean())
             for k, c in enumerate(MARGIN_FEATURES)}

    print("\n" + "=" * 68)
    print("融资融券价值实测")
    print("=" * 68)
    print(f"对比样本          {len(merged):,} 行 / {merged['date'].nunique()} 个截面"
          f" / {merged['symbol'].nunique()} 只股票")
    print(f"价格特征 {len(base)} 维     PR = {pr_base:.2f}")
    print(f"加 5 个融资特征     PR = {pr_all:.2f}    净增 {pr_all - pr_base:+.2f}")
    print("\n-- 每个融资特征落在价格基（前 11 主成分）上的 R^2 --")
    for c, v in probe.items():
        print(f"  {c:<18} R^2 = {v:.3f}")
    print("\n判据: PR 净增 +0.5 以上算真新维度；+0.1 以内是同族数据。")
    print("      R^2 低 + PR 涨 才是正交维度；R^2 高说明价格已经隐含了它。")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    merged[["date", "symbol"] + MARGIN_FEATURES].to_parquet(
        DATA_DIR / "margin_features.parquet", index=False)
    print(f"[margin] 特征落盘 -> {DATA_DIR / 'margin_features.parquet'}")


if __name__ == "__main__":
    main()
