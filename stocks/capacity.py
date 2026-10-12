"""信息容量盘点：8 个数值列里到底有几个独立维度。

用途（建仓前判重，省掉一整轮回测）：
  1. 任何新想法先投影到既有基上算 R^2（见 basis_r2），>0.95 的直接扔掉。
  2. 有效维度 D 是负控的理论来源：1100 万行只采样了 D 维空间里的一小块。
  3. D 饱和后继续做特征工程不再增加信息，只有换数据或换目标才有用。

用法:
    uv run capacity.py                 # 抽样 25% 交易日，约 640 个截面
    uv run capacity.py --sample 50
"""
from __future__ import annotations

import argparse
import pathlib

import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"

# 全部从 open/high/low/close/volume/amount/outstanding_share/turnover 合法算出，无外部数据。
FEATURES = [
    "ret1", "gap", "intra", "amp", "upper_sh", "lower_sh", "close_pos", "vwap_dev",
    "log_amt", "log_mv", "turnover_lv", "turnover_rel",
    "mom5", "mom20", "mom60", "ma_dev20", "ma_dev60",
    "vol20", "vol60", "park20", "downvol20", "max_ret20", "illiq20",
]

S1_EXPR = """
    close / nullif(lag(close) over w, 0) - 1                      as ret1,
    open  / nullif(lag(close) over w, 0) - 1                      as gap,
    close / nullif(open, 0) - 1                                   as intra,
    (high - low) / nullif(lag(close) over w, 0)                   as amp,
    (high - greatest(open, close)) / nullif(lag(close) over w, 0) as upper_sh,
    (least(open, close) - low) / nullif(lag(close) over w, 0)     as lower_sh,
    (close - low) / nullif(high - low, 0)                         as close_pos,
    close / nullif(amount / nullif(volume, 0), 0) - 1             as vwap_dev,
    ln(nullif(amount, 0))                                         as log_amt,
    ln(nullif(close * outstanding_share, 0))                       as log_mv,
    ln(high / nullif(low, 0))                                     as hl_log
"""

S2_EXPR = """
    turnover                                as turnover_lv,
    close / nullif(lag(close, 5)  over w, 0) - 1              as mom5,
    close / nullif(lag(close, 20) over w, 0) - 1              as mom20,
    close / nullif(lag(close, 60) over w, 0) - 1              as mom60,
    close / nullif(avg(close)   over w20, 0) - 1              as ma_dev20,
    close / nullif(avg(close)   over w60, 0) - 1              as ma_dev60,
    turnover / nullif(avg(turnover) over w20, 0) - 1          as turnover_rel,
    stddev_samp(ret1) over w20                                as vol20,
    stddev_samp(ret1) over w60                                as vol60,
    sqrt(avg(hl_log * hl_log) over w20 / (4 * ln(2)))         as park20,
    sqrt(avg(case when ret1 < 0 then ret1 * ret1 end) over w20) as downvol20,
    max(ret1) over w20                                        as max_ret20,
    avg(abs(ret1) / nullif(amount, 0)) over w20               as illiq20
"""

# 探针特征：刻意选 FEATURES 里没有的（10 日动量、10 日波动、5 日换手波动），
# 用来实测 basis_r2 —— 如果它们也能被既有基解释，就是「判重」有效的直接证据。
PROBES = ["probe_mom10", "probe_vol10", "probe_turn_std5"]
S2_PROBE_EXPR = """
    close / nullif(lag(close, 10) over w, 0) - 1   as probe_mom10,
    stddev_samp(ret1) over w10                     as probe_vol10,
    stddev_samp(turnover) over w5                  as probe_turn_std5
"""


def sample_dates(panel_path: pathlib.Path, sample_pct: float) -> list[pd.Timestamp]:
    """复现 build_features 的抽样交易日，供外部按其日期集对齐。"""
    rows = duckdb.sql(f"""
        select date from (select distinct date from read_parquet('{panel_path.as_posix()}'))
        using sample {sample_pct:g} percent (system, 42)
        order by date
    """).fetchall()
    return [r[0] for r in rows]


def build_features(panel_path: pathlib.Path, sample_pct: float,
                   dates: list | None = None) -> pd.DataFrame:
    """DuckDB 里算完全部特征，抽样的交易日做截面 rank（pct，0~1）。

    dates 非空时用它替代内部抽样，外部按同一日期集对齐时用。
    """
    has_dates = dates is not None and len(dates) > 0
    print(f"[capacity] 读 {panel_path.name}"
          f"{'，指定 ' + str(len(dates)) + ' 个交易日' if has_dates else f'，抽样交易日 {sample_pct:g}%'}")
    if has_dates:
        lit = ", ".join(f"date '{pd.Timestamp(d).date()}'" for d in dates)
        days_sql = f"select unnest([{lit}]) as date"
    else:
        days_sql = (f"select date from (select distinct date from base)\n"
                    f"        using sample {sample_pct:g} percent (system, 42)")
    rank_cols = "\n    , ".join(
        f"(rank() over (partition by date order by {c}) - 0.5)"
        f" / count(*) over (partition by date) as r_{c}"
        for c in FEATURES + PROBES
    )
    not_null = " and ".join(f"{c} is not null" for c in FEATURES + PROBES)
    sql = f"""
    with base as (
        select date, symbol, open, high, low, close, volume, amount,
               outstanding_share, turnover
        from read_parquet('{panel_path.as_posix()}')
        where close > 0 and volume > 0 and amount > 0
    ),
    days as (
        {days_sql}
    ),
    s1 as (
        select date, symbol, close, amount, turnover, volume,
               {S1_EXPR}
        from base
        window w as (partition by symbol order by date)
    ),
    s2 as (
        select date, symbol, close, ret1, gap, intra, amp, upper_sh, lower_sh, close_pos,
               vwap_dev, log_amt, log_mv, {S2_EXPR}, {S2_PROBE_EXPR}
        from s1
        window w   as (partition by symbol order by date),
               w5  as (partition by symbol order by date
                       rows between 4 preceding and current row),
               w10 as (partition by symbol order by date
                       rows between 9 preceding and current row),
               w20 as (partition by symbol order by date
                       rows between 19 preceding and current row),
               w60 as (partition by symbol order by date
                       rows between 59 preceding and current row)
    ),
    s3 as (
        select * from s2
        where date in (select date from days) and {not_null}
    )
    select date, symbol, {rank_cols}
    from s3
    """
    df = duckdb.sql(sql).df()
    print(f"[capacity] 抽样后 {len(df):,} 行 / {df['date'].nunique()} 个交易日")
    return df


def spearman_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """截面 rank 之后再算 Pearson == Spearman。返回 (矩阵, 特征矩阵)。"""
    x = df[[f"r_{c}" for c in FEATURES]].to_numpy(dtype=np.float64)
    c = np.corrcoef(x, rowvar=False)
    return c, x


def null_spectrum(n_feat: int, n_draw: int = 30, n_row: int = 200_000) -> np.ndarray:
    """纯 iid uniform 的特征值谱。截面 rank 后每天的边际近似均匀，故全局 iid 是合适的底噪。"""
    rng = np.random.default_rng(0)
    acc = np.zeros(n_feat)
    for _ in range(n_draw):
        x = rng.random((n_row, n_feat))
        acc += np.linalg.eigvalsh(np.corrcoef(x, rowvar=False))
    return acc / n_draw


def redundancy_pairs(corr: np.ndarray, thresh: float = 0.9) -> pd.DataFrame:
    """成对 |Spearman| 超阈值的特征对。"""
    rows = []
    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            if abs(corr[i, j]) >= thresh:
                rows.append({"a": FEATURES[i], "b": FEATURES[j],
                             "spearman": corr[i, j], "r2": corr[i, j] ** 2})
    return pd.DataFrame(rows).sort_values("r2", ascending=False) if rows else pd.DataFrame()


def multivariate_r2(x: np.ndarray) -> pd.DataFrame:
    """每个特征在其余全部特征上的 R^2。R^2>0.95 = 不含新信息（成对低相关也能抓到）。"""
    rows = []
    for k, name in enumerate(FEATURES):
        others = np.delete(x, k, axis=1)
        beta, *_ = np.linalg.lstsq(others, x[:, k], rcond=None)
        resid = x[:, k] - others @ beta
        rows.append({"feature": name,
                     "r2_vs_others": 1.0 - resid.var() / x[:, k].var()})
    return pd.DataFrame(rows).sort_values("r2_vs_others", ascending=False)


def pr_of(x: np.ndarray) -> float:
    """一组特征的参与率有效维度。"""
    ev = np.linalg.eigvalsh(np.corrcoef(x, rowvar=False))
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def effective_dims(eigvals: np.ndarray) -> dict:
    """参与率 PR=(Σλ)^2/Σλ^2，以及解释 90% / 95% 方差所需主成分数。"""
    ev = np.sort(eigvals)[::-1]
    share = np.cumsum(ev) / ev.sum()

    def n_for(q: float) -> int:
        return int(np.searchsorted(share, q) + 1)

    return {"PR": float(ev.sum() ** 2 / (ev ** 2).sum()),
            "n90": n_for(0.90), "n95": n_for(0.95),
            "top1_share": float(ev[0] / ev.sum())}


def basis_r2(basis: np.ndarray, x: np.ndarray) -> float:
    """候选特征 x 能被既有基 basis（n×D）解释的比例。>0.95 说明不含新信息，不必回测。"""
    beta, *_ = np.linalg.lstsq(basis, x, rcond=None)
    resid = x - basis @ beta
    return float(1.0 - resid.var() / x.var())


def self_check(x: np.ndarray) -> None:
    """basis_r2 的两个端点：自己投自己 = 1；纯噪声应该 ≈ 0。"""
    rng = np.random.default_rng(0)
    a = basis_r2(x, x[:, 0])
    b = basis_r2(x, rng.random(len(x)))
    assert abs(a - 1.0) < 1e-6, f"自投影应为 1，实得 {a}"
    assert b < 0.05, f"噪声不该被解释，实得 {b}"
    print(f"[capacity] 自检通过：自投影 R^2={a:.6f}，噪声 R^2={b:.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=float, default=25.0, help="抽样交易日百分比")
    ap.add_argument("--panel", default=None)
    args = ap.parse_args()

    panel_path = pathlib.Path(args.panel) if args.panel else DATA_DIR / "panel_2015_2026.parquet"
    if not panel_path.exists():
        raise SystemExit(f"缺面板文件 {panel_path}")

    rows_total = duckdb.sql(
        f"select count(*) from read_parquet('{panel_path.as_posix()}')").fetchone()[0]

    df = build_features(panel_path, args.sample)
    corr, x = spearman_matrix(df)
    ev = np.linalg.eigvalsh(corr)
    eff = effective_dims(ev)
    null = null_spectrum(len(FEATURES))
    pairs = redundancy_pairs(corr)
    r2 = multivariate_r2(x)

    self_check(x)

    # 探针：FEATURES 之外的 3 个特征，能在既有主成分基上被解释多少
    evals, evecs = np.linalg.eigh(corr)
    order = np.argsort(evals)[::-1]
    scores = (x - x.mean(0)) @ evecs[:, order]
    probe_r2 = {p: basis_r2(scores[:, :eff["n90"]], df[f"r_{p}"].to_numpy() - df[f"r_{p}"].mean())
                for p in PROBES}

    print("\n" + "=" * 68)
    print("信息容量盘点")
    print("=" * 68)
    print(f"原始数值列          8   (open/high/low/close/volume/amount/share/turnover)")
    print(f"合法派生特征        {len(FEATURES)}")
    print(f"样本                全量 {rows_total:,} 行 / 本次抽样 {len(df):,} 行")
    print(f"有效维度 PR         {eff['PR']:.1f}    (sum L)^2 / sum L^2")
    print(f"解释 90% 方差需要   {eff['n90']} 个主成分")
    print(f"解释 95% 方差需要   {eff['n95']} 个主成分")
    print(f"第一主成分          {np.sort(ev)[::-1][0]:.2f}   (噪声基准 {null.max():.2f}"
          f"，{np.sort(ev)[::-1][0] / null.max():.1f}x)")

    # 单日截面特征 vs 加上多日窗口特征：多日窗口到底带来了多少新维度
    day_idx = [FEATURES.index(f) for f in FEATURES[:11]]  # ret1..turnover_lv
    pr_day = pr_of(x[:, day_idx])
    print(f"有效维度 单日 11 维    {pr_day:.2f}")
    print(f"有效维度 全部 23 维    {eff['PR']:.2f}    多日窗口净增 {eff['PR'] - pr_day:+.2f}")

    print("\n-- 特征值谱：实测 vs iid uniform 噪声 --")
    ev_desc = np.sort(ev)[::-1]
    for k in range(len(FEATURES)):
        bar = "#" * int(round(max(ev_desc[k] - null[k], 0) * 40))
        print(f"  PC{k + 1:>2}  实测 {ev_desc[k]:5.2f}   噪声 {null[k]:5.2f}   {bar}")

    print("\n-- 成对冗余 |Spearman| >= 0.90 --")
    if len(pairs) == 0:
        print("  无")
    else:
        print(pairs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n-- 多元冗余：每个特征在其余 22 维上的 R^2 --")
    print(r2.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    dead = r2[r2["r2_vs_others"] > 0.95]["feature"].tolist()
    print(f"\n零新信息特征（R^2>0.95，可直接删）: {dead if dead else '无'}")
    print(f"独立维度估计: 名义 8 列 -> 派生 {len(FEATURES)} -> 有效 {eff['PR']:.1f}")

    print(f"\n-- 判重探针：3 个 FEATURES 之外的特征（10 日动量、10 日波动、5 日换手波动）--")
    for p, v in probe_r2.items():
        print(f"  {p:<16} 落在前 {eff['n90']} 个主成分上的 R^2 = {v:.3f}")
    x_all = np.column_stack([x, df[[f"r_{p}" for p in PROBES]].to_numpy()])
    pr_all = pr_of(x_all)
    print(f"  有效维度 {len(FEATURES)} 维 -> 加上 3 个探针 {pr_all:.2f}（净增 {pr_all - eff['PR']:+.2f}）")
    print("  读法: 纯噪声探针的 R^2 应 ≈0；0.6~0.7 只说明大部分被覆盖，"
          "剩下的是插值噪声还是真新维度，看 PR 净增。")
    print("  用法: basis_r2(主成分基, 候选特征) 配 PR 增量，两者都低才值得回测")

    OUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(corr, index=FEATURES, columns=FEATURES).to_csv(OUT_DIR / "capacity_corr.csv")
    pd.DataFrame({"pc": np.arange(1, len(FEATURES) + 1),
                  "eig_obs": ev_desc, "eig_null": null}).to_csv(
        OUT_DIR / "capacity_eig.csv", index=False)
    pairs.to_csv(OUT_DIR / "capacity_redundant_pairs.csv", index=False)
    r2.to_csv(OUT_DIR / "capacity_multivariate_r2.csv", index=False)
    print(f"[capacity] 明细 -> {OUT_DIR}/capacity_*.csv")


if __name__ == "__main__":
    main()
