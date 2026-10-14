"""市值中性化（阶段⑤ 出口）：IC 层 + 分层两层。

性质：**出口**类工作（README §进度 ⑤），**不新增防线** —— 只出读数，不改变任何因子的在册状态，
也不得用市值中性化结果淘汰/纳入因子（`假设登记表.md` §0 规则 3 + §14.2：停手期 2027-04-11 前禁新增防线）。
口径先冻结在 `doc/stocks/Alpha191候选分层回测计划.md` §18（跑数前已写入）。

要回答协议 §9 挂的「等权天然偏小盘，反转在小盘上最强 —— IC 数字有水分」：
  A 层：IC 是不是市值带来的。单变量残差 IC **按变量名列报**（A1 关只按子集大小汇总，没拆到变量）。
         复用 `alpha191.build_orth_inputs` + `orth_scan`，与 A1 关同一链路 -> 读数可比。
  B 层：分层收益是不是市值带来的。score 的截面 rank(pct) 对 log_mv 的截面 rank(pct) 取 OLS 残差，
         再用残差重跑 5 层回测；名单 = `layer_judge.csv` 过线因子（不写死）+ C5（§16 口径 1）。
         分层/绩效全部复用 §16 口径 3 那一套，不另写第二份。

市值只用冻结面板已有的 `outstanding_share`：`log_mv = ln(close × outstanding_share)`
（与 `capacity.py` / `dominance.py` 同公式）—— 不改面板、不动冻结锚点。

同源硬闸（不过就非零退出）：
  ① A 层对账：每候选的「无控制」残差 IC == out/alpha191_ic.csv 的 ic_nocontrol（1e-12）；
  ② 中性化真生效：逐期残差的正交性 `|cov|/(std_cv × std_score)` 全部 < 1e-9
     （尺度稳健写法；因子全等值的退化期单列计数、不参与 —— 直接算 Pearson 会在那里变成 0/0 假相关），
     且中性化后 G5 与原口径 G5 的相对最大差 > 1e-3（接成空操作会被这条挡住）；
  ③ 中性化不许新增空仓月末（原口径本来空仓的期不算，如 alpha042 2021-06-30 因子值全 NaN）；
  ④ 月数 = 142 - 1 = 141。
前置硬闸：freeze_guard.verify + verify_artifacts；layer_judge.self_check()（a144 锚点）。

用法:
    uv run stocks/mv_neutral_check.py            # A 层 + B 层
    uv run stocks/mv_neutral_check.py --limit-a 2

产物: out/mvn_single_control.csv + out/mvn_layers_<名>_5g.csv + out/mvn_summary.csv
      （命名禁止以 `layers_` 开头 —— freeze_guard.ARTIFACT_GLOB = "layers_*_5g.csv" 会把它当冻结产物）
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import alpha191  # noqa: E402
import combo_check as cc  # noqa: E402
import dominance as dom  # noqa: E402
import factors  # noqa: E402
import freeze_guard  # noqa: E402
import impact_check as ic  # noqa: E402
import layer_judge as judge  # noqa: E402
import portfolio_check as pc  # noqa: E402
import store  # noqa: E402

OUT_DIR = ROOT / "out"
GROUPS = 5
N_MONTHS = 142
SPLIT = "2020-12-31"     # A 层正交期（与 alpha191_dominance.py 逐字相同）
MIN_ROWS = 30           # 行集守卫（与 alpha191.orth_scan 逐字相同）
MV_TOL = 1e-9           # 硬闸②：逐期 corr(残差, rank(log_mv)) 的上界
CHANGE_TOL = 1e-3       # 硬闸②：中性化后 G5 与原口径的最小相对差
C5_NAME = "C5"


def log(msg: str) -> None:
    print(f"[mvn] {msg}", flush=True)


def mv_wide(panel_path: pathlib.Path) -> pd.DataFrame:
    """log_mv 宽表（date × symbol），与 capacity.py 同公式。"""
    df = duckdb.sql(
        f"select date, symbol, ln(nullif(close * outstanding_share, 0)) as log_mv "
        f"from read_parquet('{panel_path.as_posix()}')").df()
    return df.pivot(index="date", columns="symbol", values="log_mv").sort_index()


def rank_residual(score: pd.DataFrame, mv: pd.DataFrame) -> tuple[pd.DataFrame, float, int]:
    """§18 B 口径 3/4：逐期 OLS 残差 rank(pct)(score) ~ rank(pct)(log_mv)。

    返回 (残差, 正交性最坏值, 退化期数)。正交性用**尺度稳健**写法 `|cov|/(std_cv × std_score)`：
    直接算 Pearson 在「因子值截面全等值」的退化期会变成 0/0 —— 实测 alpha139 2021-07-08 全市场
    因子值全等于 0.5000913，去均值后只剩 float 噪声（std ≈ 1e-16），Pearson 会报出 0.95 的假相关。
    退化期没有截面信息，单列计数、不参与检查（原口径在同一期同样只能按股票顺序乱分组）。
    """
    mv = mv.reindex(index=score.index, columns=score.columns)
    a_all = score.rank(axis=1, pct=True).to_numpy(dtype=float)
    c_all = mv.rank(axis=1, pct=True).to_numpy(dtype=float)
    resid = np.full_like(a_all, np.nan)
    worst, n_deg = 0.0, 0
    for i in range(a_all.shape[0]):
        a, c = a_all[i], c_all[i]
        ok = np.isfinite(a) & np.isfinite(c)
        if int(ok.sum()) < MIN_ROWS:
            continue
        av, cv = a[ok].copy(), c[ok].copy()
        sa, sc = float(av.std()), float(cv.std())
        av = av - av.mean()
        cv = cv - cv.mean()
        cc = float(cv @ cv)
        beta = float(cv @ av) / cc if cc > 0 else 0.0
        r = av - beta * cv
        resid[i, ok] = r
        if sa <= 0 or sc <= 0:
            n_deg += 1
            continue
        worst = max(worst, abs(float(cv @ r)) / (sc * sa))
    return pd.DataFrame(resid, index=score.index, columns=score.columns), worst, n_deg


def ic_single_control(panel_path: pathlib.Path, out_dir: pathlib.Path, limit: int) -> pd.DataFrame:
    """A 层：11 个单变量（含 log_mv）各自的残差 IC，按变量名列报。"""
    ref = pd.read_csv(out_dir / "alpha191_ic.csv").set_index("factor")
    names = sorted(ref.index[ref["verdict"] == "部分独立"].tolist())
    if limit:
        names = names[:limit]
    log(f"A 层候选（alpha191_ic.csv 的「部分独立」）：{len(names)} 个")

    all_dates = pd.to_datetime(duckdb.sql(
        f"select distinct date from read_parquet('{panel_path.as_posix()}') order by 1").df()["date"])
    me = factors.month_end_dates(pd.DatetimeIndex(all_dates))
    ctrl, masks = alpha191.build_orth_inputs(panel_path, me)
    # 无控制 + 11 个单变量 + 全 11 个（最后一行只为复现 r2_allctrl，与 A1 关对账）
    use = np.vstack([masks[:len(dom.CONTROLS) + 1], masks[-1:]])
    orth_dates = me[me > pd.Timestamp(SPLIT)]
    log(f"A 层月末 {len(me)} 期，正交期 {len(orth_dates)} 期，mask {len(use)} 个（1+11+1）")

    wide = ic.alpha191.load_wide(panel_path, None)
    fwd = wide["close"].reindex(me).pct_change().shift(-1)
    o, h, l, c, v, a, r = (wide["open"], wide["high"], wide["low"], wide["close"],
                           wide["volume"], wide["amount"], wide["returns"])

    rows, bad = [], []
    for i, name in enumerate(names, 1):
        st = time.time()
        fac = alpha191.ALPHAS[name](o, h, l, c, v, a, r).replace([np.inf, -np.inf], np.nan)
        ic0, ic_m, r2_m = alpha191.orth_scan(fac.loc[me], ctrl, fwd, orth_dates, use)
        ic_mean = np.nanmean(ic_m, axis=0)
        # 对账只能拿 ic0（因子自己行集，= 归档 ic_nocontrol）；ic_mean[0] 是「控制变量齐全」行集，
        # 与归档同名列不是同一个量（orth_scan 的既有约定：两个行集只比量级）
        got, want = float(np.nanmean(ic0)), float(ref.loc[name, "ic_nocontrol"])
        if not np.isclose(got, want, rtol=0, atol=1e-12, equal_nan=True):
            bad.append(f"{name}: 无控制残差 IC {got!r} != 归档 {want!r}")
        singles = ic_mean[1:1 + len(dom.CONTROLS)]
        order = np.argsort(np.abs(singles))
        rows.append({"factor": name, "ic_nocontrol": got,
                     "ic_none_ctrl_rows": float(ic_mean[0]),
                     "ic_log_mv": float(singles[dom.CONTROLS.index("log_mv")]),
                     "衰减_log_mv": float(singles[dom.CONTROLS.index("log_mv")] - ic_mean[0]),
                     "r2_allctrl": float(np.nanmean(r2_m, axis=0)[-1]),
                     "最强控制": dom.CONTROLS[int(order[0])],
                     "最强控制_IC": float(singles[int(order[0])]),
                     **{f"单控制_{c}": float(singles[j]) for j, c in enumerate(dom.CONTROLS)}})
        log(f"A {i}/{len(names)} {name} 完成（{time.time() - st:.1f}s）")
        del fac
    if bad:
        raise SystemExit("[mvn] 硬闸① 对账失败（前 5 条）：\n  " + "\n  ".join(bad[:5]))
    log(f"硬闸① 通过：{len(rows)} 候选的 ic_nocontrol 与 alpha191_ic.csv 一致（1e-12）")
    return pd.DataFrame(rows).sort_values("factor")


def layers_b(panel_path: pathlib.Path, out_dir: pathlib.Path, names: tuple[str, ...],
             mv: pd.DataFrame, problems: list[str]) -> pd.DataFrame:
    """B 层：中性化后重跑 5 层，出与前几节可比的读数。"""
    wide = ic.alpha191.load_wide(panel_path, None)
    panel = factors.load_prices(panel_path)
    delist = store.delist_dates(panel["symbol"].unique())
    dates = wide["close"].index
    me = factors.month_end_dates(dates)
    base = pc.load_curve(pc.layer_path(pc.BASELINE, GROUPS, out_dir))
    base_layers = pd.read_csv(pc.layer_path(pc.BASELINE, GROUPS, out_dir),
                              parse_dates=["date"]).set_index("date")
    yb = pc.year_returns(base)
    ye = pc.year_returns(pc.equal_weight_curve(panel_path))
    full = list(range(pc.FIRST_YEAR, pc.LAST_FULL_YEAR + 1))

    scores = {n: ic.factor_frame(n, wide) for n in names}
    scores[C5_NAME] = cc.combo_factor(cc.combo_ranks(names, wide))

    rows = []
    for cname, score in scores.items():
        resid, worst, n_deg = rank_residual(score, mv)
        if not (worst < MV_TOL):
            problems.append(f"{cname}: 残差正交性最坏值 {worst:.2e} >= {MV_TOL}")
        me_cnt = resid.reindex(me).notna().sum(axis=1)
        orig_cnt = score.reindex(me).notna().sum(axis=1)
        new_empty = (me_cnt < MIN_ROWS) & (orig_cnt >= MIN_ROWS)   # 中性化不许新增空仓期
        if bool(new_empty.any()):
            problems.append(f"{cname}: 中性化新增 {int(new_empty.sum())} 个空仓月末"
                            f"（{', '.join(str(d.date()) for d in new_empty[new_empty].index[:3])}）")
        r_score = score.rank(axis=1, pct=True).reindex(me)
        deg_me = int(((r_score.notna().sum(axis=1) > 0) & (r_score.nunique(axis=1) <= 1)).sum())

        layers, st = cc.run_layers(resid, panel, dates, delist)
        g5 = layers["G5"]
        layers.to_csv(out_dir / f"mvn_layers_{cname}_5g.csv")

        orig_path = (out_dir / f"combo_layers_{cname}_5g.csv" if cname == C5_NAME
                     else pc.layer_path(cname, GROUPS, out_dir))
        orig = pd.read_csv(orig_path, parse_dates=["date"]).set_index("date")["G5"].dropna()
        rel = float((g5.reindex(orig.index) - orig).abs().max() / orig.abs().mean())
        if not (rel > CHANGE_TOL):
            problems.append(f"{cname}: 中性化后 G5 与原口径相对差 {rel:.2e} <= {CHANGE_TOL}（空操作？）")
        if st["月数"] != N_MONTHS - 1:
            problems.append(f"{cname}: 月数 {st['月数']} != {N_MONTHS - 1}")

        yr = pc.year_returns(g5)
        fy = yr.reindex(full).dropna()
        n_a = int((fy - yb.reindex(full)).gt(0).sum())
        n_e = int((fy - ye.reindex(full)).gt(0).sum())
        exc = (fy - yb.reindex(full)).dropna()
        corr2 = judge.corr_demeaned(layers, base_layers)
        ann, ann0 = pc.g5_annual(g5), pc.g5_annual(orig)
        rows.append({
            "名称": cname, "全期年化": ann, "原口径年化": ann0, "年化差": ann - ann0,
            "年化夏普": st["年化夏普"], "最大回撤": st["最大回撤"], "月胜率": st["月胜率"],
            "年换手双边": st["年换手双边"], "月数": st["月数"],
            "超额_基线_年数": n_a, "超额_等权_年数": n_e,
            "集中度": pc.concentration(fy), "集中度_超额_基线": pc.concentration(exc),
            "去均值相关": corr2, "平均标的数": float(me_cnt.mean()),
            "退化日数": n_deg, "月末退化期数": deg_me,
            "月末空仓期数": int((me_cnt < MIN_ROWS).sum())})
        log(f"B {cname:<9s} 年化 {ann:+.2%}（原 {ann0:+.2%}，差 {ann - ann0:+.2%}）"
            f" 夏普 {st['年化夏普']:.3f} 回撤 {st['最大回撤']:.1%} "
            f"超额 A {n_a}/11 B(等权) {n_e}/11 去均值相关 {corr2:+.3f}"
            f" |r|max {worst:.1e} 退化 {n_deg}日/月末{deg_me}期")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="市值中性化：IC 层 + 分层两层")
    ap.add_argument("--limit-a", type=int, default=0, help="A 层只跑前 N 个候选（冒烟用）")
    ap.add_argument("--skip-a", action="store_true", help="跳过 A 层")
    ap.add_argument("--skip-b", action="store_true", help="跳过 B 层（只跑 A 层）")
    args = ap.parse_args()

    t0 = time.time()
    panel_path = freeze_guard.frozen_panel_path()
    problems = freeze_guard.verify(panel_path) + freeze_guard.verify_artifacts(OUT_DIR)
    if problems:
        for m in problems:
            log(f"[X] 冻结校验不过：{m}")
        sys.exit(2)
    try:
        judge.self_check()
    except AssertionError as exc:
        log(f"[X] 前置硬闸不过：{exc}")
        sys.exit(1)

    if not args.skip_a:
        a = ic_single_control(panel_path, OUT_DIR, args.limit_a)
        a.to_csv(OUT_DIR / "mvn_single_control.csv", index=False, float_format="%.6f")
        print("\nA 层：11 个单变量各自的残差 IC（列 `单控制_*`），KILL 线 "
              f"{dom.KILL:.4f}（§3.5）")
        show = a[["factor", "ic_nocontrol", "ic_none_ctrl_rows", "ic_log_mv", "衰减_log_mv",
                  "最强控制", "最强控制_IC", "r2_allctrl"]]
        print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"  ic_nocontrol = 因子自己行集（与归档一致）；ic_none_ctrl_rows = 控制变量齐全行集"
              f"（与单控制_* 同集，两者不可混比）")
        print(f"  log_mv 单变量吃掉最多的 = {a.loc[a['ic_log_mv'].abs().idxmax(), 'factor']}"
              f"（{a['ic_log_mv'].abs().max():.4f}）")

    if not args.skip_b:
        names = tuple(pc.passing_candidates(OUT_DIR))
        log(f"B 层名单（layer_judge.csv 判定==通过）+ {C5_NAME}：{', '.join(names)}, {C5_NAME}")
        mv = mv_wide(panel_path)
        summary = layers_b(panel_path, OUT_DIR, names, mv, problems)
        summary.to_csv(OUT_DIR / "mvn_summary.csv", index=False, float_format="%.6f")

        print("\nB 层：中性化后 vs 原口径（原口径 = §14 的 layers_*.csv / §16 的 combo_layers_C5_5g.csv）")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if problems:
        for m in problems:
            log(f"[X] {m}")
        sys.exit(1)
    log(f"全部同源硬闸通过（{time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
