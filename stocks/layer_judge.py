"""分层回测判据 ②③④ 的产出脚本（冻结点的可复现件）。

为什么单独存在：`alpha191.py --layers` 只落 11 列（累计/年化/月胜率/CI/月数/夏普/回撤/
换手/笔数/被挡/费用），判据 ②去均值相关 ③因子值截面秩相关 ④样本外年化 此前只出现在临时
产物 `out/_judge*.md` 里、无产出代码 —— 所以冻结点只能冻结 1/4 条判据。本脚本补上产出，
定义如下（2026-10-12 用 `Alpha191候选分层回测计划.md` §10.5 的实测值逐项反推并核对）：

  ② 去均值相关 = 5 个组各自的「月收益 − 当期 5 组均值」序列与反转基线同组序列的 Pearson，
                 取 5 组平均。对 a144 实测 0.350、a042 0.424（与计划 §10.5 逐位一致）
  ③ 秩相关均值 = 每个自然月末 spearman(因子值截面, mom20 截面) 的均值。a144 −0.090、a042 −0.241
  ④ 样本外年化 = 净值曲线月频重采样、切 date > split、last/first ** (1/年数) − 1。
                 a144 20.4205%（计划 §10.5 报 20.42%）

冻结保护（2026-10-12 补，`freeze_guard.py` + 协议 §9.1）：默认只允许在**冻结快照**上出判据结论 ——
读面板前先核对面板 sha256/字节/行数/最大日期/交易日数/标的数与**它所消费的 layers CSV 哈希**，
不符即非零退出；`--end` 越过冻结边界默认拒绝。要用越界数据/非冻结产物做研究，必须显式
`--research`，产物另落盘并标「研究数据（非独立验证）」。

用法:
    uv run stocks/layer_judge.py                       # 读 out/layers_*_5g.csv 全部候选（冻结口径）
    uv run stocks/layer_judge.py --factors alpha144,alpha042
    uv run stocks/layer_judge.py --split 2020-12-31
    uv run stocks/layer_judge.py --research --panel data/panel_2015_2026_2026-11-30.parquet  # 研究用

产物: out/layer_judge.csv（冻结口径）/ out/layer_judge_research.csv（研究口径），stdout 打表。
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import factors  # noqa: E402
import alpha191  # noqa: E402
import freeze_guard  # noqa: E402

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"

BASELINE_NAME = "reversal20"          # 反转基线 20 日，分层 CSV 与候选同口径
BASELINE_ANNUAL = 0.0760              # ① 的阈值：退市归零引擎下的基线 G5 年化
CORR_LIMIT = 0.8                      # ② ③ 的阈值
SAMPLE_OUT_SPLIT = "2020-12-31"       # ④ 的切分：训练 <= 该日，测试 >

# 自检用（计划 §10.5 的实测值，写死；改了定义就必须同步改这里）
CHECK = {"alpha144": {"去均值相关": 0.350, "样本外年化": 0.2042}}


def layer_path(name: str, groups: int, out_dir: pathlib.Path = OUT_DIR) -> pathlib.Path:
    return out_dir / f"layers_{name}_{groups}g.csv"


def monthly_returns(curves: pd.DataFrame) -> pd.DataFrame:
    """日频净值 -> 月末净值 -> 月收益。与 factors.curve_stats 的月胜率同口径。"""
    m = curves.resample("ME").last().dropna()
    return m.pct_change().dropna()


def demean_by_period(rets: pd.DataFrame) -> pd.DataFrame:
    """每期减去当期各组的横截面均值（去掉共同的市场 beta）。"""
    return rets.sub(rets.mean(axis=1), axis=0)


def corr_demeaned(cand: pd.DataFrame, base: pd.DataFrame) -> float:
    """② 去均值相关：5 组各自与本基线同组的去均值月收益相关，取 5 组平均。"""
    a, b = demean_by_period(monthly_returns(cand)), demean_by_period(monthly_returns(base))
    common = a.index.intersection(b.index)
    vals = [a.loc[common, g].corr(b.loc[common, g]) for g in a.columns if g in b.columns]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def sample_out_annual(curve: pd.Series, split: pd.Timestamp) -> float:
    """④ 样本外年化：月频，切 date > split，last/first ** (1/年数) - 1。"""
    m = curve.resample("ME").last().dropna()
    m = m[m.index > split]
    ret = m.pct_change().dropna()
    if len(ret) < 2 or m.iloc[0] <= 0:
        return float("nan")
    return float((m.iloc[-1] / m.iloc[0]) ** (1 / (len(ret) / 12)) - 1)


def g5_annual(curve: pd.Series) -> float:
    """① 全期年化（与 curve_stats 的年化同口径：月末净值，按总月数折算）。"""
    m = curve.resample("ME").last().dropna()
    ret = m.pct_change().dropna()
    if len(ret) < 2:
        return float("nan")
    return float((m.iloc[-1] / m.iloc[0]) ** (1 / (len(ret) / 12)) - 1)


def rank_corr_vs_mom(fac: pd.DataFrame, mom: pd.DataFrame, dates: pd.DatetimeIndex) -> float:
    """③ 因子值截面秩相关（对 mom20），逐月末算再取均值。"""
    me = factors.month_end_dates(dates)
    vals = [factors.spearman(fac.loc[d], mom.loc[d]) for d in me if d in fac.index]
    vals = pd.Series(vals).dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def verdict(row: dict, use_rank: bool = True) -> str:
    """4 条判据合一。任一条不成立即不通过，写明死在第几条。

    use_rank=False（--no-rank / 无面板）时不判第 3 条：没算就是缺数据，不是死因。
    """
    dead = []
    if not (np.isfinite(row["全期年化"]) and row["全期年化"] > BASELINE_ANNUAL):
        dead.append("1")
    if not (np.isfinite(row["去均值相关"]) and row["去均值相关"] < CORR_LIMIT):
        dead.append("2")
    if use_rank and not (np.isfinite(row["秩相关均值"]) and row["秩相关均值"] < CORR_LIMIT):
        dead.append("3")
    if not (np.isfinite(row["样本外年化"]) and row["样本外年化"] > 0):
        dead.append("4")
    return "通过" if not dead else "死" + "".join(dead)


def self_check() -> None:
    """① 对账：定义若被改动，a144 的 ②④ 必须仍在计划 §10.5 的实测值上。"""
    cand = pd.read_csv(layer_path("alpha144", 5), parse_dates=["date"]).set_index("date")
    base = pd.read_csv(layer_path(BASELINE_NAME, 5), parse_dates=["date"]).set_index("date")
    got = {"去均值相关": corr_demeaned(cand, base),
           "样本外年化": sample_out_annual(cand["G5"], pd.Timestamp(SAMPLE_OUT_SPLIT))}
    for k, want in CHECK["alpha144"].items():
        if abs(got[k] - want) > 5e-4:
            raise AssertionError(f"自检不过：a144 {k} 实测 {got[k]:.4f} != 计划 §10.5 的 {want:.4f}，定义被改过？")
    # 因子对自身的秩相关必须是 1（便宜的结构检查，不碰面板）
    x = pd.Series([3.0, 1.0, 2.0, np.nan], index=list("abcd"))
    if abs(factors.spearman(x, x) - 1.0) > 1e-12:
        raise AssertionError("秩相关实现在含 NaN 时不是 1")
    print("[judge] 自检通过：a144 去均值相关/样本外年化 与计划 §10.5 一致")


def main() -> None:
    ap = argparse.ArgumentParser(description="分层回测判据 ②③④ 的可复现产出")
    ap.add_argument("--factors", default="", help="逗号分隔；默认读 out/layers_*_5g.csv 里全部候选")
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--split", default=SAMPLE_OUT_SPLIT)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=freeze_guard.FREEZE_END,
                    help=f"数据上界；默认 = 冻结边界 {freeze_guard.FREEZE_END}，超过必须加 --research")
    ap.add_argument("--panel", default="", help="面板路径；默认 = 冻结快照")
    ap.add_argument("--out-dir", default="", help="读 layers CSV 的目录；默认 out/")
    ap.add_argument("--research", action="store_true",
                    help="研究模式：允许越界数据/非冻结面板；产物另落盘并标「非独立验证」")
    ap.add_argument("--out", default="", help="产物路径；默认 out/layer_judge[_research].csv")
    ap.add_argument("--no-rank", action="store_true", help="跳过 ③（不读面板，冒烟用）")
    args = ap.parse_args()

    self_check()
    split = pd.Timestamp(args.split)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else OUT_DIR
    if args.end > freeze_guard.FREEZE_END and not args.research:
        print(f"[judge] 拒绝：--end {args.end} 越过冻结边界 {freeze_guard.FREEZE_END}（协议 §9.1）；"
              f"要用越界数据做研究，加 --research（产物会标为非独立验证）", file=sys.stderr)
        sys.exit(2)
    if args.research:
        print("[judge] ⚠ 研究模式：允许非冻结/越界数据；产物标「研究数据（非独立验证）」，"
              "不得当作独立验证（协议 §9.1）", file=sys.stderr)

    if args.factors:
        names = [x.strip() for x in args.factors.split(",") if x.strip()]
    else:
        names = sorted(p.name[len("layers_"):-len(f"_{args.groups}g.csv")]
                       for p in out_dir.glob(f"layers_*_{args.groups}g.csv"))
        names = [n for n in names if n not in ("momentum20", BASELINE_NAME)]

    base = pd.read_csv(layer_path(BASELINE_NAME, args.groups, out_dir),
                       parse_dates=["date"]).set_index("date")

    wide = mom = None
    if args.panel:
        panel_path = pathlib.Path(args.panel)
    elif args.research:
        panel_path = freeze_guard.research_panel_path(args.start, args.end)
    else:
        panel_path = freeze_guard.frozen_panel_path()
    if not args.no_rank:
        # 冻结口径的入口：面板 + 它所消费的 layers CSV 一起验（只验面板挡不住「产物被换掉」）
        problems = ([] if args.research
                    else freeze_guard.verify(panel_path) + freeze_guard.verify_artifacts(out_dir))
        if problems:
            for m in problems:
                print(f"[judge] ✗ 冻结校验不过：{m}", file=sys.stderr)
            print("[judge] 拒绝在非冻结面板/非冻结产物上出判据结论（协议 §9.1）；"
                  "研究用途加 --research", file=sys.stderr)
            sys.exit(1)
        if not panel_path.exists():
            print(f"[judge] 找不到 {panel_path}，跳过 3（研究模式请显式 --panel）", file=sys.stderr)
        else:
            print(f"[judge] 读 {panel_path.name}（03 需要）")
            wide = alpha191.load_wide(panel_path, None)
            mom = factors.momentum(wide["close"], 20)

    rows = []
    for name in names:
        p = layer_path(name, args.groups, out_dir)
        if not p.exists():
            print(f"[judge] 缺 {p}，跳过", file=sys.stderr)
            continue
        cand = pd.read_csv(p, parse_dates=["date"]).set_index("date")
        r = {"factor": name,
             "全期年化": g5_annual(cand["G5"]),
             "样本外年化": sample_out_annual(cand["G5"], split),
             "去均值相关": corr_demeaned(cand, base),
             "秩相关均值": float("nan")}
        if wide is not None:
            try:
                fac = alpha191.ALPHAS[name](wide["open"], wide["high"], wide["low"], wide["close"],
                                            wide["volume"], wide["amount"], wide["returns"])
                fac = fac.replace([np.inf, -np.inf], np.nan)
                r["秩相关均值"] = rank_corr_vs_mom(fac, mom, wide["close"].index)
            except Exception as exc:      # 因子算不出来不该毁掉已算好的 ①②④
                print(f"[judge] {name} 秩相关失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        r["判定"] = verdict(r, use_rank=(wide is not None))
        rows.append(r)
        print(f"[judge] {name:<10s} 1全期 {r['全期年化']:+.2%}  2去均值 {r['去均值相关']:+.3f}  "
              f"3秩相关 {r['秩相关均值']:+.3f}  4样本外 {r['样本外年化']:+.2%}  -> {r['判定']}")

    out = pd.DataFrame(rows)
    out["证据等级"] = freeze_guard.verdict_level(args.research)
    OUT_DIR.mkdir(exist_ok=True)
    if args.out:
        dst = pathlib.Path(args.out)
    else:
        dst = OUT_DIR / ("layer_judge_research.csv" if args.research else "layer_judge.csv")
    out.to_csv(dst, index=False)
    n_pass = int((out["判定"] == "通过").sum()) if len(out) else 0
    print(f"[judge] 证据等级 {freeze_guard.verdict_level(args.research)}"
          f"（冻结 {freeze_guard.FREEZE_COMMIT[:7]} / {freeze_guard.FREEZE_TS}）")
    print(f"[judge] {len(out)} 个候选，通过 {n_pass} 个 -> {dst}")


if __name__ == "__main__":
    main()
