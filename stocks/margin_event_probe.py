"""两融「强平事件」原料盘点（口径 A + 对照 C）。

⚠ 本脚本是**原料盘点，不是验证**：
  - 只输出「事件频次 / 事件后收益分布」这类描述量
  - 不设判据、不做多重检验、不判通过失败、不写 H 卡
  - 停手期（至 2027-04-11）禁止新增防线（`假设登记表.md` §0 规则 3）
  - 收益列同时给两个口径，**信息口径不是可交易口径**

为什么做：D 组（换口径重验两融）的月度线性路已死（协议 §9.3 实盘口径翻负），
剩下唯一站得住的分支是**事件口径**。读数 1+2（PR 净增 / R²）是无条件线性度量，
本来就看不见条件性效应，所以只能从事件频次与分布来看。

机制（`假设登记表.md` §0 规则 1 要求）：
    两融账户维持担保比例跌破平仓线后，**券商强制平仓** —— 持仓以市价卖出、
    卖方不挑价格。这是**日级**的被迫交易，月度线性口径取月均值会把它抹平。
    留痕方向：事件日之后应该有正的反转。

口径：
    A  个股单日融资余额环比跌破阈值（梯子 -3% / -5% / -10%，只作披露不选优）
    C  全市场融资余额合计单日跌幅最大的若干天（对照，做大盘强平潮）

两个收益口径（必须分开读）：
    信息口径  close(T) -> close(T+N)     —— 两融数据是**盘后**公布，T 收盘时还不知道
    可交易    open(T+1) -> close(T+N)    —— 次日开盘买入，这才是能落地的算法

用法:
    uv run stocks/margin_event_probe.py
"""
from __future__ import annotations

import argparse
import pathlib

import duckdb
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"

CACHE_GLOB = (DATA_DIR / "margin_cache" / "*.parquet").as_posix()
PANEL = (DATA_DIR / "panel_2015_2026.parquet").as_posix()

THRESHOLDS = [0.03, 0.05, 0.10]
STAT_COLS = ["info1", "info5", "info20", "exec1", "exec5", "exec20"]

BASE_SQL = f"""
with td as (
    select date, lag(date) over (order by date) as prev_td
    from (select distinct date from read_parquet('{PANEL}'))
),
raw as (
    select date, symbol, "融资余额" as bal
    from read_parquet('{CACHE_GLOB}')
    where "融资余额" is not null
),
chg as (
    select date, symbol, bal,
           lag(date) over w as pdate,
           lag(bal)  over w as pbal
    from raw
    window w as (partition by symbol order by date)
),
ev as (
    -- 只留真正的「上一交易日」环比：停牌断开的那一段不算单日变化
    select c.date, c.symbol, c.bal, c.bal / nullif(c.pbal, 0) - 1 as chg1
    from chg c join td on td.date = c.date
    where c.pdate = td.prev_td and c.pbal > 0
),
px as (
    select date, symbol, open, close
    from read_parquet('{PANEL}')
    where close > 0 and open > 0
),
fwd as (
    select date, symbol, open, close,
           lead(close, 1)  over w as c1,
           lead(close, 5)  over w as c5,
           lead(close, 20) over w as c20,
           lead(open,  1)  over w as o1
    from px
    window w as (partition by symbol order by date)
)
select e.date, e.symbol, e.chg1,
       f.c1  / f.close - 1 as info1,
       f.c5  / f.close - 1 as info5,
       f.c20 / f.close - 1 as info20,
       f.c1  / f.o1    - 1 as exec1,
       f.c5  / f.o1    - 1 as exec5,
       f.c20 / f.o1    - 1 as exec20
from ev e join fwd f on f.date = e.date and f.symbol = e.symbol
"""

MARKET_SQL = f"""
select date, sum("融资余额") as tot
from read_parquet('{CACHE_GLOB}')
where "融资余额" is not null
group by date
order by date
"""


def self_check(con: duckdb.DuckDBPyConnection, base: pd.DataFrame) -> None:
    """环比必须真的接在上一交易日上（与 SQL 侧独立重算对账，防窗口接错）。"""
    assert base["chg1"].notna().all(), "chg1 有 NaN（pbal>0 与 prev_td 两道闸没生效）"
    assert base["chg1"].median() > -0.05, f"环比中位数异常 {base['chg1'].median():.4f}"

    sym = base["symbol"].value_counts().index[0]
    raw = con.sql(f'''select date, "融资余额" as bal from read_parquet('{CACHE_GLOB}')
                      where symbol = '{sym}' and "融资余额" > 0 order by date''').df()
    raw["date"] = pd.to_datetime(raw["date"])
    raw["prev"] = raw["bal"].shift(1)
    raw["pdate"] = raw["date"].shift(1)
    td = pd.to_datetime(con.sql(
        f"select distinct date from read_parquet('{PANEL}') order by date").df()["date"])
    prev_map = dict(zip(td, td.shift(1)))
    keep = raw["pdate"] == raw["date"].map(prev_map)
    exp = raw[keep].set_index("date")["bal"] / raw[keep].set_index("date")["prev"] - 1
    got = base[base["symbol"] == sym].set_index("date")["chg1"]
    diff = float((exp - got).abs().max())
    assert diff < 1e-12, f"独立重算对账失败 max|diff|={diff}（{sym}）"

    n_ev = int((base["chg1"] <= -0.05).sum())
    assert n_ev > 0, "阈值 -5% 下事件数为 0，口径或数据接错了"
    print(f"[event] 自检通过：与 pandas 侧重算 {sym} 逐行一致（max|diff|={diff:.1e}），"
          f"-5% 事件 {n_ev:,} 行")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    con = duckdb.connect()
    print("[event] 读两融明细 + 面板 ...")
    base = con.sql(BASE_SQL).df()
    base["date"] = pd.to_datetime(base["date"])
    print(f"[event] 样本 {len(base):,} 行 / {base['date'].nunique()} 个交易日 "
          f"/ {base['symbol'].nunique()} 只")
    self_check(con, base)

    # -- 事件频次与分布 --（阈值梯子只作披露，不要挑最优）
    print("\n" + "=" * 72)
    print("A 口径：个股单日融资余额环比跌破阈值 —— 事件后收益分布")
    print("=" * 72)
    print(f"{'阈值':>6} {'组':<8} {'n':>10} {'占比':>7} "
          f"{'info1均值':>10} {'info5均值':>10} {'info20均值':>11} "
          f"{'exec1均值':>10} {'exec5均值':>10} {'exec20均值':>11}")
    rows = []
    for th in THRESHOLDS:
        ev = base["chg1"] <= -th
        for label, mask in (("事件", ev), ("非事件", ~ev)):
            sub = base[mask]
            rec = {"阈值": -th, "组": label, "n": int(mask.sum()),
                   "占比": float(mask.mean())}
            for c in STAT_COLS:
                rec[f"{c}_均值"] = float(sub[c].mean())
                rec[f"{c}_中位"] = float(sub[c].median())
            rows.append(rec)
            print(f"{-th:>6.0%} {label:<8} {rec['n']:>10,} {rec['占比']:>6.2%} "
                  + " ".join(f"{rec[f'{c}_均值']:>10.4f}".replace("float", "") for c in STAT_COLS))
    summary = pd.DataFrame(rows)

    # -- 事件按年分布 --
    ev5 = base["chg1"] <= -0.05
    by_year = base[ev5].groupby(base.loc[ev5, "date"].dt.year).size()
    all_year = base.groupby(base["date"].dt.year).size()
    print("\n-- -5% 事件按年（看是否只集中在某一年）--")
    for y in all_year.index:
        n = int(by_year.get(y, 0))
        print(f"  {y}  {n:>7,} / {all_year[y]:>9,}  {n / all_year[y]:>6.2%}")

    # -- 对照 C：全市场融资余额合计跌幅最大的若干天 --
    mkt = con.sql(MARKET_SQL).df()
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt["chg"] = mkt["tot"].pct_change()
    worst = mkt.nsmallest(12, "chg")[["date", "tot", "chg"]]
    day_ret = base.groupby("date")[["info1", "exec1", "info5"]].mean()
    worst = worst.join(day_ret, on="date")
    print("\n-- C 口径（对照）：全市场融资余额合计单日跌幅最大的 12 天 --")
    print(worst.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  全样本日均：info1 {base['info1'].mean():.4f} / "
          f"exec1 {base['exec1'].mean():.4f} / info5 {base['info5'].mean():.4f}")

    OUT_DIR.mkdir(exist_ok=True)
    out = pathlib.Path(args.out) if args.out else OUT_DIR / "margin_event_probe.csv"
    summary.to_csv(out, index=False)
    worst.to_csv(OUT_DIR / "margin_event_market_days.csv", index=False)
    print(f"\n[event] 汇总 -> {out}")
    print("  证据等级：研究数据（非独立验证）；性质：原料披露，不是判据，"
          "不得用于淘汰/纳入任何候选")
    print("  未做：涨跌停过滤、扣费、多重检验、样本内外切分 —— 要当判据必须先补这些")


if __name__ == "__main__":
    main()
