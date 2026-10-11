"""国泰君安 Alpha191 因子库（短周期价量因子），自写实现。

来源：公式集来自公开的国泰君安《数量化专题之八十四：基于短周期价量特征的多因子选股体系》，
参照实现见开源项目 popbo/alphas 的 alphas191.py。**本文件不引任何第三方因子库**：
算子（含 ICU 之外的 Tsrank / Decaylinear / Wma / Regbeta / Lowday / Highday）全部自写，
且不用 rolling.apply 的 python 回调（14.9M 次调用会跑成小时级），全部向量化。

评估口径沿用本项目已有的东西：factors.py 的月度 Spearman IC + 训练/测试切分 +
Bonferroni/FDR（协议 §3），不新增防线。

## 本轮排除的 43 个（见 SKIP，理由逐条写在代码里）

- `vwap` 类 36 个：我们的 `amount/volume` 是**真实成交价**，`close` 是**后复权价**，
  实测 sh600519 2026-10-09 二者差 8.85 倍（1268.34 vs 11218.61）。同日纯比值可约掉复权因子，
  但 `VWAP-CLOSE` 这类**差值**、`SUM(VWAP,10)` 这类**跨日**运算全部失真，所以整类排除。
- `benchmark_*` 类 3 个（75/181/182）：我们没有指数面板。
- 恒等于 0 的 4 个（30/143/149/190）：公式集里的占位，无信息。
- α054：原式的 `STD(ABS(CLOSE-OPEN))` 无窗口 = 全样本标准差，**用了未来数据**（前视），不实现。

## 用法

    uv run stocks/alpha191.py --only alpha006,alpha014        # 冒烟测试
    uv run stocks/alpha191.py                                # 全量，写 out/alpha191_ic.csv
    uv run stocks/alpha191.py --limit-symbols 800            # 抽 800 只先跑通
"""
from __future__ import annotations

import argparse
import math
import pathlib
import time
from statistics import NormalDist

import duckdb
import numpy as np
import pandas as pd

import capacity
import dominance as dom
import factors
import margin_factor as mf
import store

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"

SKIP = {
    **{f"alpha{n:03d}": "依赖 vwap：amount/volume 是真实价、close 是后复权价，差值/跨日运算口径不可比"
       for n in (7, 8, 12, 13, 16, 17, 26, 36, 39, 41, 44, 45, 61, 64, 73, 74, 77, 87, 90,
                 92, 101, 108, 114, 119, 120, 121, 124, 125, 130, 131, 138, 154, 156, 163, 170, 179)},
    **{f"alpha{n:03d}": "依赖 benchmark_close/open：我们没有指数面板，择时那套是市场级信号"
       for n in (75, 181, 182)},
    **{f"alpha{n:03d}": "公式恒为 0（原集合里的占位），无信息"
       for n in (30, 143, 149, 190)},
    "alpha054": "原式 STD(ABS(CLOSE-OPEN)) 无窗口 = 全样本标准差，含前视（用到未来数据），不实现",
}


def log(msg: str) -> None:
    print(f"[a191] {msg}", flush=True)


def self_check() -> None:
    """1..191 必须被「实现」或「跳过」恰好覆盖一次，否则公式集被动过。"""
    done, skipped = set(ALPHAS), set(SKIP)
    all_ids = {f"alpha{n:03d}" for n in range(1, 192)}
    assert not (done & skipped), f"既实现又跳过: {sorted(done & skipped)}"
    assert done | skipped == all_ids, f"未覆盖: {sorted(all_ids - done - skipped)}"
    log(f"自检通过：实现 {len(done)} + 跳过 {len(skipped)} = 191")


def load_wide(panel_path: pathlib.Path, limit_symbols: int | None) -> dict[str, pd.DataFrame]:
    """读 panel 成 date × symbol 宽表。returns 用 close 的一阶比值。"""
    log(f"读 {panel_path.name}")
    cols = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
    df = duckdb.sql(f"select {', '.join(cols)} from read_parquet('{panel_path.as_posix()}')").df()
    if limit_symbols:
        keep = sorted(df["symbol"].unique())[:limit_symbols]
        df = df[df["symbol"].isin(keep)]
    wide: dict[str, pd.DataFrame] = {}
    for col in cols[2:]:
        wide[col] = df.pivot(index="date", columns="symbol", values=col).sort_index()
    wide["returns"] = wide["close"] / wide["close"].shift(1) - 1
    log(f"宽表 {wide['close'].shape[0]} 交易日 × {wide['close'].shape[1]} 只")
    return wide


# ---------------------------------------------------------------- 算子（全向量化）

def blank(x: pd.DataFrame) -> pd.DataFrame:
    """同形状的全 NaN 宽表，给条件赋值用。"""
    return pd.DataFrame(np.nan, index=x.index, columns=x.columns)


def pick(cond: pd.DataFrame, yes: pd.DataFrame, no) -> pd.DataFrame:
    """cond 为真取 yes，否则取 no（no 可以是标量或同形状表）。"""
    return yes.where(cond, no)


def pos(cond: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """cond 为真取 x，否则取 0。"""
    return x.where(cond, 0.0)


def valid(x: pd.DataFrame, w: int) -> pd.DataFrame:
    """窗口内有 w 个有效值的掩码，用来把窗口不足的行压回 NaN。"""
    return x.notna().astype(float).rolling(w).sum() == w


def Log(x): return np.log(x.abs().where(x > 0))
def Abs(x): return x.abs()
def Sign(x): return np.sign(x)
def Rank(x): return x.rank(axis=1, method="min", pct=True)
def Delta(x, n): return x.diff(n)
def Delay(x, n): return x.shift(n)
def Sum(x, w): return x.rolling(w).sum()
def Prod(x, w):
    """窗口连乘。Rolling 没有 prod，改走 log 空间 + 负号计数，全向量化（本库只用到 w=1）。"""
    neg = (x < 0).astype(float).rolling(w).sum()
    mag = np.exp(np.log(x.abs().replace(0.0, np.nan)).rolling(w).sum())
    return mag.where(neg % 2 == 0, -mag).where(valid(x, w))
def Mean(x, w): return x.rolling(w).mean()
def Std(x, w): return x.rolling(w).std()
def Tsmax(x, w): return x.rolling(w).max()
def Tsmin(x, w): return x.rolling(w).min()
def Max(x, y): return np.maximum(x, y)
def Min(x, y): return np.minimum(x, y)
def Rowmax(x): return x.max(axis=1)
def Rowmin(x): return x.min(axis=1)
def Sma(x, n, m): return x.ewm(alpha=m / n, adjust=False).mean()
def Sequence(n): return np.arange(1, n + 1)
def Count(cond, w): return cond.astype(float).rolling(w).sum()
def Sumif(x, w, cond): return pos(cond, x).rolling(w).sum()


def Corr(x, y, w):
    """滚动相关。窗口内某一侧为常数时相关无定义，按参考实现填 0；前 w-1 行压回 NaN。"""
    r = x.rolling(w).corr(y).fillna(0.0)
    r.iloc[: w - 1] = np.nan
    return r


def Cov(x, y, w): return x.rolling(w).cov(y)


def Tsrank(x, w):
    """窗口内最后一个值的顺位（1 起算，同值取平均秩）。

    用 w 次 shift 的向量化写法代替 rolling.apply(rankdata)，否则 14.6M 次 python 回调。
    """
    last = x
    less = blank(x).fillna(0.0)
    equal = blank(x).fillna(0.0)
    for k in range(w):
        s = x.shift(k)
        less += (s < last).astype(float)
        equal += (s == last).astype(float)
    r = less + (equal + 1.0) / 2.0
    return r.where(valid(x, w))


def Decaylinear(x, w):
    """线性衰减加权平均，最新值权重 w、最旧权重 1（与参考实现同向）。"""
    num = blank(x).fillna(0.0)
    for k in range(w):
        num += (w - k) * x.shift(k)
    r = num / (w * (w + 1) / 2)
    return r.where(valid(x, w))


def Wma(x, w):
    """权重 0.9^k（最新值权重 1）的加权平均。"""
    weights = np.power(0.9, np.arange(w, dtype=float))
    num = blank(x).fillna(0.0)
    for k in range(w):
        num += weights[k] * x.shift(k)
    r = num / weights.sum()
    return r.where(valid(x, w))


def Regbeta(x, w):
    """窗口内对时间 0..w-1 做一元回归的斜率，闭式解，不用 polyfit。"""
    t = np.arange(w, dtype=float)
    st, stt = t.sum(), (t * t).sum()
    num = blank(x).fillna(0.0)
    for k in range(w):
        num += t[w - 1 - k] * x.shift(k)          # x.shift(k) 对应时间 t = w-1-k
    sx = x.rolling(w).sum()
    slope = (num - st * sx / w) / (stt - st * st / w)
    return slope.where(valid(x, w))


def Lowday(x, w):
    """窗口内最小值距最后一天的天数（1 起算）。"""
    return x.rolling(w).apply(lambda a: len(a) - int(np.nanargmin(a)), raw=True)


def Highday(x, w):
    """窗口内最大值距最后一天的天数（1 起算）。"""
    return x.rolling(w).apply(lambda a: len(a) - int(np.nanargmax(a)), raw=True)


# ---------------------------------------------------------------- 147 个公式

def a001(o, h, l, c, v, a, r): return -1 * Corr(Rank(Delta(Log(v), 1)), Rank((c - o) / o), 6)
def a002(o, h, l, c, v, a, r): return -1 * Delta(((c - l) - (h - c)) / (h - l), 1)
def a003(o, h, l, c, v, a, r):
    d = c - Delay(c, 1)
    return Sum(pick(d == 0, c * 0, pick(d > 0, c - Min(l, Delay(c, 1)), c - Max(h, Delay(c, 1)))), 6)
def a004(o, h, l, c, v, a, r):
    x, y = Sum(c, 8) / 8 + Std(c, 8), Sum(c, 2) / 2
    part = pick(x < y, c * 0 - 1, c * 0 + 1)
    part = part.where(~(x == y), -1.0)
    return part.where(~((x == y) & (v / Mean(v, 20) >= 1)), 1.0)
def a005(o, h, l, c, v, a, r): return -1 * Tsmax(Corr(Tsrank(v, 5), Tsrank(h, 5), 5), 3)
def a006(o, h, l, c, v, a, r): return -1 * Rank(Sign(Delta(o * 0.85 + h * 0.15, 4)))
def a009(o, h, l, c, v, a, r):
    return Sma(((h + l) / 2 - (Delay(h, 1) + Delay(l, 1)) / 2) * (h - l) / v, 7, 2)
def a010(o, h, l, c, v, a, r): return Rank(Tsmax(pick(r < 0, Std(r, 20), c) ** 2, 5))
def a011(o, h, l, c, v, a, r): return Sum(((c - l) - (h - c)) / (h - l) * v, 6)
def a014(o, h, l, c, v, a, r): return c - Delay(c, 5)
def a015(o, h, l, c, v, a, r): return o / Delay(c, 1) - 1
def a018(o, h, l, c, v, a, r): return c / Delay(c, 5)
def a019(o, h, l, c, v, a, r):
    d = c - Delay(c, 5)
    return pick(d < 0, d / Delay(c, 5), pick(d == 0, c * 0, d / c))
def a020(o, h, l, c, v, a, r): return (c - Delay(c, 6)) / Delay(c, 6) * 100
def a021(o, h, l, c, v, a, r): return Regbeta(Mean(c, 6), 6)
def a022(o, h, l, c, v, a, r):
    s = (c - Mean(c, 6)) / Mean(c, 6)
    return Sma(s - Delay(s, 3), 12, 1)
def a023(o, h, l, c, v, a, r):
    cond, sd = c > Delay(c, 1), Std(c, 20)
    up, dn = pos(cond, sd), pos(~cond, sd)
    return 100 * Sma(up, 20, 1) / (Sma(up, 20, 1) + Sma(dn, 20, 1))
def a024(o, h, l, c, v, a, r): return Sma(c - Delay(c, 5), 5, 1)
def a025(o, h, l, c, v, a, r):
    return ((-1 * Rank(Delta(c, 7) * (1 - Rank(Decaylinear(v / Mean(v, 20), 9)))))
            * (1 + Rank(Sum(r, 250))))
def a027(o, h, l, c, v, a, r):
    return Wma((c - Delay(c, 3)) / Delay(c, 3) * 100 + (c - Delay(c, 6)) / Delay(c, 6) * 100, 12)
def a028(o, h, l, c, v, a, r):
    # 参考实现第二个 Sma 的分母写成 Tsmax(LOW,9)，与第一个 Sma 的 Tsmin(LOW,9) 不一致（疑笔误），
    # 这里照抄参考实现，未"修正"，避免把改错当成改进
    return (3 * Sma((c - Tsmin(l, 9)) / (Tsmax(h, 9) - Tsmin(l, 9)) * 100, 3, 1)
            - 2 * Sma(Sma((c - Tsmin(l, 9)) / (Tsmax(h, 9) - Tsmax(l, 9)) * 100, 3, 1), 3, 1))
def a029(o, h, l, c, v, a, r): return (c - Delay(c, 6)) / Delay(c, 6) * v
def a031(o, h, l, c, v, a, r): return (c - Mean(c, 12)) / Mean(c, 12) * 100
def a032(o, h, l, c, v, a, r): return -1 * Sum(Rank(Corr(Rank(h), Rank(v), 3)), 3)
def a033(o, h, l, c, v, a, r):
    return ((-1 * Tsmin(l, 5) + Delay(Tsmin(l, 5), 5))
            * Rank((Sum(r, 240) - Sum(r, 20)) / 220) * Tsrank(v, 5))
def a034(o, h, l, c, v, a, r): return Mean(c, 12) / c
def a035(o, h, l, c, v, a, r):
    return -1 * Min(Rank(Decaylinear(Delta(o, 1), 15)),
                    Rank(Decaylinear(Corr(v, o * 0.65 + o * 0.35, 17), 7)))
def a037(o, h, l, c, v, a, r):
    x = Sum(o, 5) * Sum(r, 5)
    return -1 * Rank(x - Delay(x, 10))
def a038(o, h, l, c, v, a, r): return pos(Sum(h, 20) / 20 < h, -1 * Delta(h, 2))
def a040(o, h, l, c, v, a, r):
    cond = c > Delay(c, 1)
    return Sum(pos(cond, v), 26) / Sum(pos(~cond, v), 26) * 100
def a042(o, h, l, c, v, a, r): return -1 * Rank(Std(h, 10)) * Corr(h, v, 10)
def a043(o, h, l, c, v, a, r):
    d = c - Delay(c, 1)
    return Sum(pick(d > 0, v, pick(d < 0, -1 * v, v * 0)), 6)
def a046(o, h, l, c, v, a, r): return (Mean(c, 3) + Mean(c, 6) + Mean(c, 12) + Mean(c, 24)) / (4 * c)
def a047(o, h, l, c, v, a, r): return Sma((Tsmax(h, 6) - c) / (Tsmax(h, 6) - Tsmin(l, 6)) * 100, 9, 1)
def a048(o, h, l, c, v, a, r):
    s = Sign(c - Delay(c, 1)) + Sign(Delay(c, 1) - Delay(c, 2)) + Sign(Delay(c, 2) - Delay(c, 3))
    return (-1 * Rank(s) * Sum(v, 5)) / Sum(v, 20)
def a049(o, h, l, c, v, a, r):
    x = Max(Abs(h - Delay(h, 1)), Abs(l - Delay(l, 1)))
    cond = (h + l) > (Delay(h, 1) + Delay(l, 1))
    p1, p2 = pos(~cond, x), pos(cond, x)
    return Sum(p1, 12) / (Sum(p1, 12) + Sum(p2, 12))
def a050(o, h, l, c, v, a, r):
    x = Max(Abs(h - Delay(h, 1)), Abs(l - Delay(l, 1)))
    cond = (h + l) <= (Delay(h, 1) + Delay(l, 1))
    p1, p2 = pos(~cond, x), pos(cond, x)
    return (Sum(p1, 12) - Sum(p2, 12)) / (Sum(p1, 12) + Sum(p2, 12))
def a051(o, h, l, c, v, a, r):
    x = Max(Abs(h - Delay(h, 1)), Abs(l - Delay(l, 1)))
    cond = (h + l) <= (Delay(h, 1) + Delay(l, 1))
    p1, p2 = pos(~cond, x), pos(cond, x)
    return Sum(p1, 12) / (Sum(p1, 12) + Sum(p2, 12))
def a052(o, h, l, c, v, a, r):
    m = Delay((h + l + c) / 3, 1)
    return Sum(Max(h - m, 0.0), 26) / Sum(Max(m - l, 0.0), 26) * 100
def a053(o, h, l, c, v, a, r): return Count(c > Delay(c, 1), 12) / 12 * 100
def a055(o, h, l, c, v, a, r):
    A, B, C = Abs(h - Delay(c, 1)), Abs(l - Delay(c, 1)), Abs(h - Delay(l, 1))
    D = Abs(Delay(c, 1) - Delay(o, 1))
    c1, c2, c3 = (A > B) & (A > C), (B > C) & (B > A), (C >= A) & (C >= B)
    part = pick(c1, A + B / 2 + D / 4, pick(c2, B + A / 2 + D / 4, C + D / 4))
    part0 = 16 * (c + (c - o) / 2 - Delay(o, 1))
    return Sum(part0 / part * Max(A, B), 20)
def a056(o, h, l, c, v, a, r):
    A = Rank(o - Tsmin(o, 12))
    B = Rank(Rank(Corr(Sum((h + l) / 2, 19), Sum(Mean(v, 40), 19), 13)) ** 5)
    return pos(A < B, c * 0 + 1)
def a057(o, h, l, c, v, a, r): return Sma((c - Tsmin(l, 9)) / (Tsmax(h, 9) - Tsmin(l, 9)) * 100, 3, 1)
def a058(o, h, l, c, v, a, r): return Count(c > Delay(c, 1), 20) / 20 * 100
def a059(o, h, l, c, v, a, r):
    # 参考实现第三个分支用 Max(LOW,DELAY(CLOSE,1))，与 α003 的 Max(HIGH,...) 不一致（疑笔误），照抄
    d = c - Delay(c, 1)
    return Sum(pick(d == 0, c * 0, pick(d > 0, c - Min(l, Delay(c, 1)), c - Max(l, Delay(c, 1)))), 20)
def a060(o, h, l, c, v, a, r): return Sum(((c - l) - (h - c)) / (h - l) * v, 20)
def a062(o, h, l, c, v, a, r): return -1 * Corr(h, Rank(v), 5)
def a063(o, h, l, c, v, a, r):
    d = c - Delay(c, 1)
    return Sma(Max(d, 0.0), 6, 1) / Sma(Abs(d), 6, 1) * 100
def a065(o, h, l, c, v, a, r): return Mean(c, 6) / c
def a066(o, h, l, c, v, a, r): return (c - Mean(c, 6)) / Mean(c, 6) * 100
def a067(o, h, l, c, v, a, r):
    d = c - Delay(c, 1)
    return Sma(Max(d, 0.0), 24, 1) / Sma(Abs(d), 24, 1) * 100
def a068(o, h, l, c, v, a, r):
    return Sma(((h + l) / 2 - (Delay(h, 1) + Delay(l, 1)) / 2) * (h - l) / v, 15, 2)
def a069(o, h, l, c, v, a, r):
    do = Delay(o, 1)
    DTM = pos(o > do, Max(h - o, o - do))
    DBM = pos(o < do, Max(o - l, o - do))
    sd, sb = Sum(DTM, 20), Sum(DBM, 20)
    return pick(sd > sb, (sd - sb) / sd, (sd - sb) / sb).where(~(sd == sb), 0.0)
def a070(o, h, l, c, v, a, r): return Std(a, 6)
def a071(o, h, l, c, v, a, r): return (c - Mean(c, 24)) / Mean(c, 24) * 100
def a072(o, h, l, c, v, a, r): return Sma((Tsmax(h, 6) - c) / (Tsmax(h, 6) - Tsmin(l, 6)) * 100, 15, 1)
def a076(o, h, l, c, v, a, r):
    x = Abs(c / Delay(c, 1) - 1) / v
    return Std(x, 20) / Mean(x, 20)
def a078(o, h, l, c, v, a, r):
    m = (h + l + c) / 3
    return (m - Mean(m, 12)) / (0.015 * Mean(Abs(c - Mean(m, 12)), 12))
def a079(o, h, l, c, v, a, r):
    d = c - Delay(c, 1)
    return Sma(Max(d, 0.0), 12, 1) / Sma(Abs(d), 12, 1) * 100
def a080(o, h, l, c, v, a, r): return (v - Delay(v, 5)) / Delay(v, 5) * 100
def a081(o, h, l, c, v, a, r): return Sma(v, 21, 2)
def a082(o, h, l, c, v, a, r): return Sma((Tsmax(h, 6) - c) / (Tsmax(h, 6) - Tsmin(l, 6)) * 100, 20, 1)
def a083(o, h, l, c, v, a, r): return -1 * Rank(Cov(Rank(h), Rank(v), 5))
def a084(o, h, l, c, v, a, r):
    d = c - Delay(c, 1)
    return Sum(pick(d > 0, v, pick(d < 0, v * 0, -1 * v)), 20)
def a085(o, h, l, c, v, a, r): return Tsrank(v / Mean(v, 20), 20) * Tsrank(-1 * Delta(c, 7), 8)
def a086(o, h, l, c, v, a, r):
    A = (Delay(c, 20) - Delay(c, 10)) / 10 - (Delay(c, 10) - c) / 10
    return pick(A > 0.25, c * 0 - 1,
                pick(A < 0, c * 0 + 1, -1 * (c - Delay(c, 1))))
def a088(o, h, l, c, v, a, r): return (c - Delay(c, 20)) / Delay(c, 20) * 100
def a089(o, h, l, c, v, a, r):
    f, s = Sma(c, 13, 2), Sma(c, 27, 2)
    return 2 * (f - s - Sma(f - s, 10, 2))
def a091(o, h, l, c, v, a, r):
    return -1 * Rank(c - Tsmax(c, 5)) * Rank(Corr(Mean(v, 40), l, 5))
def a093(o, h, l, c, v, a, r):
    do = Delay(o, 1)
    return Sum(pos(o < do, Max(o - l, o - do)), 20)
def a094(o, h, l, c, v, a, r):
    d = c - Delay(c, 1)
    return Sum(pick(d > 0, v, pick(d < 0, -1 * v, v * 0)), 30)
def a095(o, h, l, c, v, a, r): return Std(a, 20)
def a096(o, h, l, c, v, a, r):
    return Sma(Sma((c - Tsmin(l, 9)) / (Tsmax(h, 9) - Tsmin(l, 9)) * 100, 3, 1), 3, 1)
def a097(o, h, l, c, v, a, r): return Std(v, 10)
def a098(o, h, l, c, v, a, r):
    cond = Delta(Sum(c, 100) / 100, 100) / Delay(c, 100) <= 0.05
    return pick(cond, -1 * (c - Tsmin(c, 100)), -1 * Delta(c, 3))
def a099(o, h, l, c, v, a, r): return -1 * Rank(Cov(Rank(c), Rank(v), 5))
def a100(o, h, l, c, v, a, r): return Std(v, 20)
def a102(o, h, l, c, v, a, r):
    d = v - Delay(v, 1)
    return Sma(Max(d, 0.0), 6, 1) / Sma(Abs(d), 6, 1) * 100
def a103(o, h, l, c, v, a, r): return (20 - Lowday(l, 20)) / 20 * 100
def a104(o, h, l, c, v, a, r): return -1 * (Delta(Corr(h, v, 5), 5) * Rank(Std(c, 20)))
def a105(o, h, l, c, v, a, r): return -1 * Corr(Rank(o), Rank(v), 10)
def a106(o, h, l, c, v, a, r): return c - Delay(c, 20)
def a107(o, h, l, c, v, a, r):
    return (-1 * Rank(o - Delay(h, 1))) * Rank(o - Delay(c, 1)) * Rank(o - Delay(l, 1))
def a109(o, h, l, c, v, a, r):
    x = Sma(h - l, 10, 2)
    return x / Sma(x, 10, 2)
def a110(o, h, l, c, v, a, r):
    dc = Delay(c, 1)
    return Sum(Max(h - dc, 0.0), 20) / Sum(Max(dc - l, 0.0), 20) * 100
def a111(o, h, l, c, v, a, r):
    x = v * ((c - l) - (h - c)) / (h - l)
    return Sma(x, 11, 2) - Sma(x, 4, 2)
def a112(o, h, l, c, v, a, r):
    d = c - Delay(c, 1)
    p1, p2 = pos(d > 0, d), pos(d < 0, Abs(d))
    return (Sum(p1, 12) - Sum(p2, 12)) / (Sum(p1, 12) + Sum(p2, 12)) * 100
def a113(o, h, l, c, v, a, r):
    return -1 * ((Rank(Sum(Delay(c, 5), 20) / 20) * Corr(c, v, 2))
                 * Rank(Corr(Sum(c, 5), Sum(c, 20), 2)))
def a115(o, h, l, c, v, a, r):
    return (Rank(Corr(h * 0.9 + c * 0.1, Mean(v, 30), 10))
            ** Rank(Corr(Tsrank((h + l) / 2, 4), Tsrank(v, 10), 7)))
def a116(o, h, l, c, v, a, r): return Regbeta(c, 20)
def a117(o, h, l, c, v, a, r):
    return Tsrank(v, 32) * (1 - Tsrank((c + h) - l, 16)) * (1 - Tsrank(r, 32))
def a118(o, h, l, c, v, a, r): return Sum(h - o, 20) / Sum(o - l, 20) * 100
def a122(o, h, l, c, v, a, r):
    s = Sma(Sma(Sma(Log(c), 13, 2), 13, 2), 13, 2)
    return (s - Delay(s, 1)) / Delay(s, 1)
def a123(o, h, l, c, v, a, r):
    A = Rank(Corr(Sum((h + l) / 2, 20), Sum(Mean(v, 60), 20), 9))
    B = Rank(Corr(l, v, 6))
    return pos(A < B, c * 0 - 1)
def a126(o, h, l, c, v, a, r): return (c + h + l) / 3
def a127(o, h, l, c, v, a, r):
    return Mean((100 * (c - Tsmax(c, 12)) / Tsmax(c, 12)) ** 2, 12) ** 0.5
def a128(o, h, l, c, v, a, r):
    A = (h + l + c) / 3
    cond = A > Delay(A, 1)
    return 100 - 100 / (1 + Sum(pos(cond, A * v), 14) / Sum(pos(~cond, A * v), 14))
def a129(o, h, l, c, v, a, r):
    d = c - Delay(c, 1)
    return Sum(pos(d < 0, Abs(d)), 12)
def a132(o, h, l, c, v, a, r): return Mean(a, 20)
def a133(o, h, l, c, v, a, r):
    return (20 - Highday(h, 20)) / 20 * 100 - (20 - Lowday(l, 20)) / 20 * 100
def a134(o, h, l, c, v, a, r): return (c - Delay(c, 12)) / Delay(c, 12) * v
def a135(o, h, l, c, v, a, r): return Sma(Delay(c / Delay(c, 20), 1), 20, 1)
def a136(o, h, l, c, v, a, r): return -1 * Rank(Delta(r, 3)) * Corr(o, v, 10)
def a137(o, h, l, c, v, a, r):
    A, B = Abs(h - Delay(c, 1)), Abs(l - Delay(c, 1))
    C, D = Abs(h - Delay(l, 1)), Abs(Delay(c, 1) - Delay(o, 1))
    c1, c2 = (A > B) & (A > C), (B > C) & (B > A)
    part = pick(c1, A + B / 2 + D / 4, pick(c2, B + A / 2 + D / 4, C + D / 4))
    part0 = 16 * (c + (c - o) / 2 - Delay(o, 1))
    return part0 / part.replace(0.0, np.nan) * Max(A, B)
def a139(o, h, l, c, v, a, r): return -1 * Corr(o, v, 10)
def a140(o, h, l, c, v, a, r):
    return Min(Rank(Decaylinear(Rank(o) + Rank(l) - Rank(h) - Rank(c), 8)),
               Tsrank(Decaylinear(Corr(Tsrank(c, 8), Tsrank(Mean(v, 60), 20), 8), 7), 3))
def a141(o, h, l, c, v, a, r): return -1 * Rank(Corr(Rank(h), Rank(Mean(v, 15)), 9))
def a142(o, h, l, c, v, a, r):
    return ((-1 * Rank(Tsrank(c, 10))) * Rank(Delta(Delta(c, 1), 1))
            * Rank(Tsrank(v / Mean(v, 20), 5)))
def a144(o, h, l, c, v, a, r):
    cond = c < Delay(c, 1)
    x = Abs(c / Delay(c, 1) - 1) / a
    return Sumif(x, 20, cond) / Count(cond, 20)
def a145(o, h, l, c, v, a, r): return (Mean(v, 9) - Mean(v, 26)) / Mean(v, 12) * 100
def a146(o, h, l, c, v, a, r):
    ret = (c - Delay(c, 1)) / Delay(c, 1)
    q = ret - Sma(ret, 61, 2)
    return Mean(q, 20) * q / Sma(Sma(ret, 61, 2) ** 2, 61, 2)
def a147(o, h, l, c, v, a, r): return Regbeta(Mean(c, 12), 12)
def a148(o, h, l, c, v, a, r):
    cond = Rank(Corr(o, Sum(Mean(v, 60), 9), 6)) < Rank(o - Tsmin(o, 14))
    return pos(cond, c * 0 - 1)
def a150(o, h, l, c, v, a, r): return (c + h + l) / 3 * v
def a151(o, h, l, c, v, a, r): return Sma(c - Delay(c, 20), 20, 1)
def a152(o, h, l, c, v, a, r):
    x = Delay(Sma(Delay(c / Delay(c, 9), 1), 9, 1), 1)
    return Sma(Mean(x, 12) - Mean(x, 26), 9, 1)
def a153(o, h, l, c, v, a, r):
    return (Mean(c, 3) + Mean(c, 6) + Mean(c, 12) + Mean(c, 24)) / 4
def a155(o, h, l, c, v, a, r):
    f, s = Sma(v, 13, 2), Sma(v, 27, 2)
    return f - s - Sma(f - s, 10, 2)
def a157(o, h, l, c, v, a, r):
    x = Rank(Rank(Log(Sum(Tsmin(Rank(Rank(-1 * Rank(Delta(c - 1, 5)))), 2), 1))))
    return Tsmin(Prod(x, 1), 5) + Tsrank(Delay(-1 * r, 6), 5)
def a158(o, h, l, c, v, a, r):
    s = Sma(c, 15, 2)
    return ((h - s) - (l - s)) / c
def a159(o, h, l, c, v, a, r):
    dc = Delay(c, 1)
    lo, rng = Min(l, dc), Max(h, dc) - Min(l, dc)
    return ((c - Sum(lo, 6)) / Sum(rng, 6) * 12 * 24
            + (c - Sum(lo, 12)) / Sum(rng, 12) * 6 * 24
            + (c - Sum(lo, 24)) / Sum(rng, 24) * 6 * 12) * 100 / (6 * 12 + 6 * 24 + 12 * 24)
def a160(o, h, l, c, v, a, r):
    return Sma(pos(c <= Delay(c, 1), Std(c, 20)), 20, 1)
def a161(o, h, l, c, v, a, r):
    dc = Delay(c, 1)
    return Mean(Max(Max(h - l, Abs(dc - h)), Abs(dc - l)), 12)
def a162(o, h, l, c, v, a, r):
    d = c - Delay(c, 1)
    A = Sma(Max(d, 0.0), 12, 1) / Sma(Abs(d), 12, 1) * 100
    return (A - Tsmin(A, 12)) / (Sma(A, 12, 1) - Tsmin(A, 12))
def a164(o, h, l, c, v, a, r):
    cond = c > Delay(c, 1)
    x = pick(cond, 1 / (c - Delay(c, 1)), c * 0 + 1)
    rng = (h - l).replace(0.0, np.nan)
    return Sma((x - Tsmin(x, 12)) / rng * 100, 13, 2)
def a165(o, h, l, c, v, a, r):
    x = Sum(c - Mean(c, 48), 48)
    return -1 * (1 / Std(c, 48).div(Rowmin(x), axis=0)).sub(Rowmax(x), axis=0)
def a166(o, h, l, c, v, a, r):
    ret = c / Delay(c, 1) - 1
    p1 = -20 * (20 - 1) ** 1.5 * Sum(ret - Mean(ret, 20), 20)
    p2 = (20 - 1) * (20 - 2) * (Sum(Mean(c / Delay(c, 1), 20) ** 2, 20)) ** 1.5
    return p1 / p2
def a167(o, h, l, c, v, a, r):
    d = c - Delay(c, 1)
    return Sum(pos(d > 0, d), 12)
def a168(o, h, l, c, v, a, r): return -1 * v / Mean(v, 20)
def a169(o, h, l, c, v, a, r):
    x = Delay(Sma(c - Delay(c, 1), 9, 1), 1)
    return Sma(Mean(x, 12) - Mean(x, 26), 10, 1)
def a171(o, h, l, c, v, a, r):
    return (-1 * ((l - c) * (o ** 5))) / ((c - h) * (c ** 5))
def a172(o, h, l, c, v, a, r):
    TR = Max(Max(h - l, Abs(h - Delay(c, 1))), Abs(l - Delay(c, 1)))
    HD, LD = h - Delay(h, 1), Delay(l, 1) - l
    p1 = pos((LD > 0) & (LD > HD), LD)
    p2 = pos((HD > 0) & (HD > LD), HD)
    x = Sum(p1, 14) * 100 / Sum(TR, 14)
    y = Sum(p2, 14) * 100 / Sum(TR, 14)
    return Mean(Abs(x - y) / (x + y) * 100, 6)
def a173(o, h, l, c, v, a, r):
    return (3 * Sma(c, 13, 2) - 2 * Sma(Sma(c, 13, 2), 13, 2)
            + Sma(Sma(Sma(Log(c), 13, 2), 13, 2), 13, 2))
def a174(o, h, l, c, v, a, r):
    return Sma(pos(c > Delay(c, 1), Std(c, 20)), 20, 1)
def a175(o, h, l, c, v, a, r):
    dc = Delay(c, 1)
    return Mean(Max(Max(h - l, Abs(dc - h)), Abs(dc - l)), 6)
def a176(o, h, l, c, v, a, r):
    return Corr(Rank((c - Tsmin(l, 12)) / (Tsmax(h, 12) - Tsmin(l, 12))), Rank(v), 6)
def a177(o, h, l, c, v, a, r): return (20 - Highday(h, 20)) / 20 * 100
def a178(o, h, l, c, v, a, r): return (c - Delay(c, 1)) / Delay(c, 1) * v
def a180(o, h, l, c, v, a, r):
    cond = Mean(v, 20) < v
    d = Delta(c, 7)
    return pick(cond, -1 * Tsrank(Abs(d), 60) * Sign(d), -1 * v)
def a183(o, h, l, c, v, a, r):
    x = Sum(c - Mean(c, 24), 24)
    return -1 * (1 / Std(c, 24).div(Rowmin(x), axis=0)).sub(Rowmax(x), axis=0)
def a184(o, h, l, c, v, a, r):
    return Rank(Corr(Delay(o - c, 1), c, 200)) + Rank(o - c)
def a185(o, h, l, c, v, a, r): return Rank(-1 * ((1 - (o / c)) ** 2))
def a186(o, h, l, c, v, a, r):
    TR = Max(Max(h - l, Abs(h - Delay(c, 1))), Abs(l - Delay(c, 1)))
    HD, LD = h - Delay(h, 1), Delay(l, 1) - l
    p1 = pos((LD > 0) & (LD > HD), LD)
    p2 = pos((HD > 0) & (HD > LD), HD)
    x = Sum(p1, 14) * 100 / Sum(TR, 14)
    y = Sum(p2, 14) * 100 / Sum(TR, 14)
    e = Mean(Abs(x - y) / (x + y) * 100, 6)
    return (e + Delay(e, 6)) / 2
def a187(o, h, l, c, v, a, r):
    do = Delay(o, 1)
    return Sum(pos(o <= do, Max(h - o, o - do)), 20)
def a188(o, h, l, c, v, a, r):
    x = h - l
    return (x - Sma(x, 11, 2)) / Sma(x, 11, 2) * 100
def a189(o, h, l, c, v, a, r): return Mean(Abs(c - Mean(c, 6)), 6)
def a191(o, h, l, c, v, a, r): return Corr(Mean(v, 20), l, 5) + (h + l) / 2 - c


# 显式映射，不用 globals()/getattr 反射，grep 得到、diff 看得见
ALPHAS = {
    "alpha001": a001, "alpha002": a002, "alpha003": a003, "alpha004": a004,
    "alpha005": a005, "alpha006": a006, "alpha009": a009, "alpha010": a010,
    "alpha011": a011, "alpha014": a014, "alpha015": a015, "alpha018": a018,
    "alpha019": a019, "alpha020": a020, "alpha021": a021, "alpha022": a022,
    "alpha023": a023, "alpha024": a024, "alpha025": a025, "alpha027": a027,
    "alpha028": a028, "alpha029": a029, "alpha031": a031, "alpha032": a032,
    "alpha033": a033, "alpha034": a034, "alpha035": a035, "alpha037": a037,
    "alpha038": a038, "alpha040": a040, "alpha042": a042, "alpha043": a043,
    "alpha046": a046, "alpha047": a047, "alpha048": a048, "alpha049": a049,
    "alpha050": a050, "alpha051": a051, "alpha052": a052, "alpha053": a053,
    "alpha055": a055, "alpha056": a056, "alpha057": a057, "alpha058": a058,
    "alpha059": a059, "alpha060": a060, "alpha062": a062, "alpha063": a063,
    "alpha065": a065, "alpha066": a066, "alpha067": a067, "alpha068": a068,
    "alpha069": a069, "alpha070": a070, "alpha071": a071, "alpha072": a072,
    "alpha076": a076, "alpha078": a078, "alpha079": a079, "alpha080": a080,
    "alpha081": a081, "alpha082": a082, "alpha083": a083, "alpha084": a084,
    "alpha085": a085, "alpha086": a086, "alpha088": a088, "alpha089": a089,
    "alpha091": a091, "alpha093": a093, "alpha094": a094, "alpha095": a095,
    "alpha096": a096, "alpha097": a097, "alpha098": a098, "alpha099": a099,
    "alpha100": a100, "alpha102": a102, "alpha103": a103, "alpha104": a104,
    "alpha105": a105, "alpha106": a106, "alpha107": a107, "alpha109": a109,
    "alpha110": a110, "alpha111": a111, "alpha112": a112, "alpha113": a113,
    "alpha115": a115, "alpha116": a116, "alpha117": a117, "alpha118": a118,
    "alpha122": a122, "alpha123": a123, "alpha126": a126, "alpha127": a127,
    "alpha128": a128, "alpha129": a129, "alpha132": a132, "alpha133": a133,
    "alpha134": a134, "alpha135": a135, "alpha136": a136, "alpha137": a137,
    "alpha139": a139, "alpha140": a140, "alpha141": a141, "alpha142": a142,
    "alpha144": a144, "alpha145": a145, "alpha146": a146, "alpha147": a147,
    "alpha148": a148, "alpha150": a150, "alpha151": a151, "alpha152": a152,
    "alpha153": a153, "alpha155": a155, "alpha157": a157, "alpha158": a158,
    "alpha159": a159, "alpha160": a160, "alpha161": a161, "alpha162": a162,
    "alpha164": a164, "alpha165": a165, "alpha166": a166, "alpha167": a167,
    "alpha168": a168, "alpha169": a169, "alpha171": a171, "alpha172": a172,
    "alpha173": a173, "alpha174": a174, "alpha175": a175, "alpha176": a176,
    "alpha177": a177, "alpha178": a178, "alpha180": a180, "alpha183": a183,
    "alpha184": a184, "alpha185": a185, "alpha186": a186, "alpha187": a187,
    "alpha188": a188, "alpha189": a189, "alpha191": a191,
}



# ---------------------------------------------------------------- 评估

def build_orth_inputs(panel_path: pathlib.Path, me: pd.DatetimeIndex):
    """正交化要用 11 个控制变量的月末截面，直接复用 dominance.py 的那套（口径必须一致）。"""
    log("准备正交化控制集（11 个价格因子）")
    px = capacity.build_features(panel_path, 25.0, dates=me)
    ctrl = {c: px.pivot(index="date", columns="symbol", values=f"r_{c}").reindex(me)
            for c in dom.CONTROLS}
    masks = np.zeros((len(dom.CONTROLS) + 2, len(dom.CONTROLS)), dtype=bool)
    for j in range(len(dom.CONTROLS)):
        masks[j, j] = True                      # 单控制变量
    masks[-1] = True                            # 全 11 个
    # masks[0] 全 0 = 无控制，用来和 ic_series 对账
    return ctrl, masks


def orth_scan(fac: pd.DataFrame, ctrl: dict, fwd: pd.DataFrame, dates: pd.DatetimeIndex,
              masks: np.ndarray) -> tuple[list[float], np.ndarray, np.ndarray]:
    """逐期截面：因子值对控制集回归取残差，再算残差的秩 IC。

    返回 (对账用 IC 列表, IC 矩阵, R² 矩阵)。对账列表在**因子自己的行集**上算，
    与 factors.ic_series 同口径，用来证明这套接线没接错；IC 矩阵用「控制变量齐全」的行集，
    两个行集不同（缺控制特征的股票会被剔掉），所以只能比量级不能比相等。

    与 dominance.py 的差别：这里**不剔除涨跌停** —— 这一步是信息口径，不是可交易口径。
    """
    cols = list(dom.CONTROLS)
    ic0: list[float] = []
    ic_hist, r2_hist = [], []
    for dt in dates:
        if dt not in fac.index or dt not in fwd.index:
            continue
        y_all, f_all = fac.loc[dt], fwd.loc[dt]
        own = pd.DataFrame({"y": y_all, "fwd": f_all}).dropna()
        if len(own) >= 30:
            ic0.append(factors.spearman(own["y"], own["fwd"]))
        block = pd.DataFrame({c: ctrl[c].loc[dt] for c in cols})
        block["y"] = y_all.reindex(block.index)
        block["fwd"] = f_all.reindex(block.index)
        block = block.dropna()
        if len(block) < 30:
            continue
        y = block["y"].to_numpy(dtype=float)
        rrank = dom._ranks(block["fwd"].to_numpy(dtype=float))   # 秩必须在实际行集上算
        if not np.any(rrank):
            continue
        ic, r2 = dom.subset_scan(block[cols].to_numpy(dtype=float), y, rrank, masks)
        ic_hist.append(ic)
        r2_hist.append(r2)
    return ic0, np.array(ic_hist), np.array(r2_hist)


def orth_verdict(ic_raw: float, ic_by_mask: np.ndarray, r2_all: float) -> tuple[str, str, float]:
    """按协议 §3.3/§3.5 已有阈值判读，不自造阈值。返回 (标签, 说明, 单控制中最强的那个 IC)。"""
    kill = dom.KILL
    if not np.isfinite(ic_raw) or abs(ic_raw) < kill:
        return "F5 本来就无信号", f"无控制 IC {ic_raw:+.4f} 已低于量级线 {kill:.4f}", ic_raw
    singles = ic_by_mask[1:-1]
    j = int(np.nanargmin(np.abs(singles))) if np.isfinite(singles).any() else -1
    if j >= 0 and abs(singles[j]) < kill:
        return ("F3 确凿换皮", f"单个控制变量 {dom.CONTROLS[j]} 就把它打到 "
                             f"{singles[j]:+.4f}（< {kill:.4f}）", float(singles[j]))
    ic_all = float(ic_by_mask[-1]) if len(ic_by_mask) else np.nan
    if np.isfinite(ic_all) and abs(ic_all) > kill:
        return "部分独立", f"全 11 控制后残差 IC 仍 {ic_all:+.4f}（R² {r2_all:.3f}）", float(singles[j])
    return ("被多变量联手吃掉", f"单控制最弱只到 {singles[j]:+.4f}，全 11 控制后 "
                              f"{ic_all:+.4f}（R² {r2_all:.3f}）", float(singles[j]))


def ic_stats(ic: pd.Series, split: str) -> dict:
    """一段 IC 序列的均值/ICIR/t。t 用均值/标准差×sqrt(期数)（与 margin_factor.py 同口径）。"""
    n = len(ic)
    mean, std = ic.mean(), ic.std()
    t = mean / std * math.sqrt(n) if std and n else np.nan
    return {"n": n, "ic_mean": mean, "ic_std": std, "icir": mean / std if std else np.nan, "t": t}


def p_value(t: float) -> float:
    """双侧正态近似 p 值，不引 scipy。"""
    if not np.isfinite(t):
        return np.nan
    return 2 * (1 - NormalDist().cdf(abs(t)))


def gate(rows: list[dict]) -> None:
    """就地补 Bonferroni / BH-FDR 两列。检验次数 = 实际算出来的因子数。"""
    ps = np.array([r["p"] for r in rows], dtype=float)
    m = np.isfinite(ps).sum()
    if not m:
        return
    thresh = 0.05 / m
    for r in rows:
        r["bonferroni"] = int(np.isfinite(r["p"]) and r["p"] < thresh)
    order = np.argsort(np.where(np.isfinite(ps), ps, np.inf))
    for rank, idx in enumerate(order[:m], 1):
        rows[idx]["fdr"] = int(rows[idx]["p"] <= rank / m * 0.05)
    log(f"检验 {m} 个因子，Bonferroni 阈值 p < {thresh:.2e}；"
        f"过 Bonferroni {sum(r['bonferroni'] for r in rows)} 个，过 FDR {sum(r.get('fdr', 0) for r in rows)} 个")


def main() -> None:
    ap = argparse.ArgumentParser(description="GTJA Alpha191 因子库（自写实现）月度 IC 筛选")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-10-09")
    ap.add_argument("--split", default="2020-12-31", help="训练/测试分界（该日含在训练段）")
    ap.add_argument("--only", default="", help="只跑这些因子，逗号分隔，冒烟测试用")
    ap.add_argument("--limit-symbols", type=int, default=None, help="只取前 N 只，冒烟测试用")
    ap.add_argument("--no-orth", action="store_true", help="跳过正交化（冒烟用）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    self_check()

    panel_path = DATA_DIR / f"panel_{args.start[:4]}_{args.end[:4]}.parquet"
    if not panel_path.exists():
        panel_path = store.build_panel(args.start, args.end)

    wide = load_wide(panel_path, args.limit_symbols)
    closes = wide["close"]
    dates = closes.index
    o, h, l, c, v, a, r = (wide["open"], wide["high"], wide["low"], wide["close"],
                          wide["volume"], wide["amount"], wide["returns"])
    me = factors.month_end_dates(dates)
    fwd = closes.reindex(me).pct_change().shift(-1)
    log(f"月频 {len(me)} 期，训练/测试分界 {args.split}")

    ctrl, masks = build_orth_inputs(panel_path, me) if not args.no_orth else (None, None)
    orth_dates = me[(me > pd.Timestamp(args.split))]   # 只在测试段判读，避免训练段泄漏

    names = args.only.split(",") if args.only else list(ALPHAS)
    rows: list[dict] = []
    t0 = time.time()
    for i, name in enumerate(names, 1):
        st = time.time()
        try:
            fac = ALPHAS[name](o, h, l, c, v, a, r)
            fac = fac.replace([np.inf, -np.inf], np.nan)
            ic = factors.ic_series(closes, fac.loc[me], dates)
        except Exception as exc:
            log(f"{i}/{len(names)} {name} 失败: {type(exc).__name__}: {exc}")
            rows.append({"factor": name, "status": "报错", "reason": f"{type(exc).__name__}: {exc}"})
            continue
        full = ic_stats(ic, args.split)
        tr = ic_stats(ic[ic.index <= pd.Timestamp(args.split)], args.split)
        te = ic_stats(ic[ic.index > pd.Timestamp(args.split)], args.split)
        sign_same = int(np.isfinite(tr["ic_mean"]) and np.isfinite(te["ic_mean"])
                        and np.sign(tr["ic_mean"]) == np.sign(te["ic_mean"]))
        extra: dict = {}
        if ctrl is not None:
            ic0, ic_m, r2_m = orth_scan(fac.loc[me], ctrl, fwd, orth_dates, masks)
            if len(ic_m):
                # 对账：这一列必须和 factors.ic_series 在测试段的 IC 均值逐位相等，
                # 不等就说明控制集接错了行、或秩算错了行集，直接停
                gap = abs(float(np.mean(ic0)) - float(te["ic_mean"])) if ic0 else np.inf
                if gap > 1e-12:
                    raise ValueError(f"{name}: 正交化对账不过，行集/秩接错（差 {gap:.2e}）")
                by_mask = np.nanmean(ic_m, axis=0)
                r2_all = float(np.nanmean(r2_m[:, -1]))
                tag, why, ic1 = orth_verdict(float(np.mean(ic0)), by_mask, r2_all)
                extra = {"ic_nocontrol": float(np.mean(ic0)), "ic_single_min": ic1,
                         "ic_allctrl": float(by_mask[-1]), "r2_allctrl": r2_all,
                         "orth_n": int(ic_m.shape[0]), "verdict": tag, "verdict_why": why}
        rows.append({"factor": name, "status": "已算", "reason": "",
                     **full, "p": p_value(full["t"]), "bonferroni": 0, "fdr": 0,
                     "ic_train": tr["ic_mean"], "ic_test": te["ic_mean"],
                     "t_train": tr["t"], "t_test": te["t"], "sign_same": sign_same, **extra})
        log(f"{i}/{len(names)} {name} IC {full['ic_mean']:+.4f} ICIR {full['icir']:+.2f} "
            f"t {full['t']:+.2f} | 训练 {tr['ic_mean']:+.4f} 测试 {te['ic_mean']:+.4f} "
            f"（{time.time() - st:.1f}s）")

    calc = [row for row in rows if row["status"] == "已算"]
    gate(calc)
    for n, reason in SKIP.items():
        rows.append({"factor": n, "status": "跳过", "reason": reason, "bonferroni": 0, "fdr": 0})

    out = pd.DataFrame(rows).sort_values(["status", "factor"])
    OUT_DIR.mkdir(exist_ok=True)
    path = pathlib.Path(args.out) if args.out else OUT_DIR / "alpha191_ic.csv"
    out.to_csv(path, index=False)
    log(f"共 {len(out)} 行（已算 {len(calc)} / 跳过 {len(SKIP)} / 报错 "
        f"{len(out) - len(calc) - len(SKIP)}），耗时 {time.time() - t0:.0f}s -> {path}")

    if calc:
        top = sorted(calc, key=lambda x: -abs(x["ic_mean"]))[:10]
        cols = ["factor", "ic_mean", "icir", "t", "bonferroni", "fdr", "ic_train", "ic_test"]
        if "verdict" in calc[0]:
            cols += ["ic_nocontrol", "ic_single_min", "ic_allctrl", "r2_allctrl", "verdict"]
        print(pd.DataFrame(top)[cols].to_string(index=False))
        if "verdict" in calc[0]:
            vc = pd.Series([r["verdict"] for r in calc]).value_counts()
            log(f"判读分布: {vc.to_dict()}")
            keep = [r for r in calc if r["verdict"] == "部分独立"
                    and r["bonferroni"] and r["sign_same"]]
            log(f"过 Bonferroni + 训练测试同号 + 全控制后仍独立: {len(keep)} 个")
            for r in sorted(keep, key=lambda x: -abs(x["ic_allctrl"]))[:15]:
                print(f"    {r['factor']:<10s} 无控制 {r['ic_nocontrol']:+.4f} -> 全控制 "
                      f"{r['ic_allctrl']:+.4f}  R² {r['r2_allctrl']:.3f}  t {r['t']:+.2f}")


if __name__ == "__main__":
    main()
