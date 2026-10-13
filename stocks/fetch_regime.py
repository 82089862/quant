"""外部宏观状态 -> 月频 regime 表（喂 stocks/regime_ic.py，对应假设 H12）。

只做两件事：拉 6 个外部序列、按**发布滞后**对齐到因子 IC 的月末交易日。

目标轴来自 panel 的实际交易日（与 `factors.ic_series` 的 `month_end_dates` 同源），
**不用日历月末**——用日历月末（如 2015-01-31）与 IC 序列（2015-01-30）join 对不上。

用法:
    uv run stocks/fetch_regime.py            # 写 stocks/data/regime_monthly.csv
    uv run stocks/fetch_regime.py --check    # 只打印诊断，不写文件
"""
from __future__ import annotations

import argparse
import pathlib

import akshare as ak
import duckdb
import numpy as np
import pandas as pd

import factors

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_PATH = DATA_DIR / "regime_monthly.csv"

# 陈旧上限：超过这个天数没有新的真实观测就不再用（避免 asof 把 3 个月前的值静默外推）。
# 8 = 美债/商品市场数据的最长自然间隔（周末 + 假日 ≤ 4 天）也覆盖 CFTC 周频（≤ 7 天）。
MAX_STALE_DAYS = 8


# ---------- 数据源（每个源只拉一次，SPEC 里复用） ----------

def _src_cftc() -> pd.DataFrame:
    """CFTC 非商业持仓（投机盘）。1986-01-15 起，1993 起周频。"""
    df = ak.macro_usa_cftc_nc_holding()
    df["日期"] = pd.to_datetime(df["日期"])
    return df.set_index("日期").sort_index()


def _src_lme() -> pd.DataFrame:
    """LME 库存 / 注销仓单。2014-07-02 起，序列止于 2026-07-17（会触发陈旧丢弃）。"""
    df = ak.macro_euro_lme_stock()
    df["日期"] = pd.to_datetime(df["日期"])
    return df.set_index("日期").sort_index()


def _src_rates() -> pd.DataFrame:
    """中美国债收益率。1990-12-19 起。"""
    df = ak.bond_zh_us_rate()
    df["日期"] = pd.to_datetime(df["日期"])
    return df.set_index("日期").sort_index()


def _src_fut(symbol: str) -> pd.DataFrame:
    """新浪外盘期货日线。CL 1996 起；GC / HG 均 2016-10-12 起（新浪 10 年上限）。"""
    df = ak.futures_foreign_hist(symbol=symbol)[["date", "close"]].dropna()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


# ---------- 6 个自变量（口径写死在这里，是唯一 provenance） ----------
# level: 取「月末 - lag 日」的最后一个真实观测；delta: 对 level 再做月末环比。
# 一律用强度 / 变化率而非水平量（依据：margin_probe 实测「水平量一半是 size/换手影子」）。

SPEC: list[dict] = [
    {"name": "cftc_usd_net_ratio", "lag": 3, "delta": False,
     "what": "美元投机净持仓 / (多+空)  —— 投机拥挤度（价格算不出）",
     "level": lambda s: s["cftc"]["美元-净仓位"] /
                        (s["cftc"]["美元-多头仓位"] + s["cftc"]["美元-空头仓位"])},
    {"name": "lme_cu_cancel_ratio", "lag": 1, "delta": False,
     "what": "LME 铜注销仓单 / 铜库存  —— 实需紧张度（价格算不出）",
     "level": lambda s: s["lme"]["铜-注销仓单"] / s["lme"]["铜-库存"]},
    {"name": "us_10y2y", "lag": 0, "delta": False,
     "what": "美债 10Y-2Y 期限利差（百分点）  —— 衰退预期",
     "level": lambda s: s["rates"]["美国国债收益率10年-2年"]},
    {"name": "us_10y_chg", "lag": 0, "delta": True,
     "what": "美债 10Y 的月末环比变化（百分点）  —— 全球贴现率冲击",
     "level": lambda s: s["rates"]["美国国债收益率10年"]},
    {"name": "wti_ret", "lag": 0, "delta": True,
     "what": "WTI 月收益  —— 供给侧能源成本冲击",
     "level": lambda s: s["fut"]["CL"]},
    {"name": "cu_au_ratio_chg", "lag": 0, "delta": True,
     "what": "铜金比（HG/GC）的月末环比变化  —— 风险偏好（工业 vs 避险）",
     "level": lambda s: s["fut"]["HG"] / s["fut"]["GC"]},
]


def month_end_axis(panel_path: pathlib.Path) -> pd.DatetimeIndex:
    """panel 的实际月末交易日 —— 必须与 factors.ic_series 的 index 逐日相同。"""
    dates = duckdb.sql(
        f"select distinct date from read_parquet('{panel_path.as_posix()}') order by date"
    ).df()["date"]
    return factors.month_end_dates(pd.DatetimeIndex(dates))


def sample_at(s: pd.Series, axis: pd.DatetimeIndex, lag: int) -> tuple[pd.Series, pd.Series]:
    """月末 - lag 日的最后一个真实观测 + 该观测的陈旧天数。

    asof 只取「该时点及之前」的最后一条，所以 lag 天然防前视；
    但若源序列已断，asof 会静默外推 —— 所以必须同时返回陈旧天数供调用方丢弃。
    """
    t = axis - pd.Timedelta(days=lag)
    # asof(DatetimeIndex) 返回的是**按 where 标签对齐**的 Series（where = axis - lag），
    # 直接塞回 index=axis 会按标签重对齐、lag>0 时全错位成 NaN —— 必须转 ndarray 按位置对齐。
    val = np.asarray(s.asof(t), dtype=float)
    # DatetimeIndex.asof 只收标量，逐元素取（142 期，无需向量化）
    idx = pd.DatetimeIndex(s.index)
    last = [idx.asof(x) for x in t] if len(s) else [pd.NaT] * len(t)
    stale = pd.Series([(a - b).days if pd.notna(b) else 9999 for a, b in zip(t, last)], index=axis)
    return pd.Series(val, index=axis), stale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-10-09")
    ap.add_argument("--check", action="store_true", help="只打印诊断，不写文件")
    args = ap.parse_args()

    panel_path = DATA_DIR / f"panel_{args.start[:4]}_{args.end[:4]}.parquet"
    if not panel_path.exists():
        raise SystemExit(f"缺 panel：{panel_path}（先跑 stocks/store.py 合成）")
    axis = month_end_axis(panel_path)
    print(f"[regime] 目标轴 {len(axis)} 个月末交易日：{axis[0].date()} ~ {axis[-1].date()}")

    print("[regime] 拉外部序列（每源一次）…")
    src = {"cftc": _src_cftc(), "lme": _src_lme(), "rates": _src_rates(),
           "fut": {"CL": _src_fut("CL"), "HG": _src_fut("HG"), "GC": _src_fut("GC")}}
    src["fut"] = {k: v["close"] for k, v in src["fut"].items()}

    cols: dict[str, pd.Series] = {}
    print(f"\n{'变量':22s} {'lag':>3s} {'非空':>5s} {'丢弃':>5s} {'覆盖区间':26s} 最后真实观测")
    for spec in SPEC:
        lvl = spec["level"](src).astype(float)
        v, stale = sample_at(lvl, axis, spec["lag"])
        dropped = int((stale > MAX_STALE_DAYS).sum())
        v = v.where(stale <= MAX_STALE_DAYS)
        if spec["delta"]:
            v = v.pct_change() if spec["name"].endswith("_ret") else v.diff()
        cols[spec["name"]] = v
        cov = f"{lvl.index.min().date()} ~ {lvl.index.max().date()}"
        print(f"{spec['name']:22s} {spec['lag']:3d} {int(v.notna().sum()):5d} {dropped:5d} {cov:26s} {lvl.index.max().date()}")
        print(f"{'':22s} {spec['what']}")

    out = pd.DataFrame(cols, index=axis)
    out.index.name = "date"

    # 验收：除 cu_au_ratio_chg（HG/GC 只有 10 年）外，非空月数须 >= 130
    short = [c for c in out.columns if out[c].notna().sum() < 130 and c != "cu_au_ratio_chg"]
    if short:
        raise SystemExit(f"[regime] 非空月数 < 130：{short}（源断档或 lag 接错，先查）")
    n_cu = int(out["cu_au_ratio_chg"].notna().sum())
    print(f"\n[regime] 断言通过：5 个变量非空 >= 130；cu_au_ratio_chg 非空 {n_cu} < 130 "
          f"（HG/GC 均 2016-10-12 起，已知）→ 按计划 §7.2 决策**分列报告**，不截齐其余 5 个")
    print(f"[regime] 最后真实观测日：{out.index[-1].date()}；"
          f"LME 存量序列止于 {src['lme'].index.max().date()} → 其尾部 2~3 个月已按陈旧丢弃")

    if args.check:
        print("[regime] --check：未写文件")
        return
    DATA_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_PATH)
    print(f"[regime] {out.shape[1]} 列 × {out.shape[0]} 行 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
