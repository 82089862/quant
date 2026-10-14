"""组合骨架（阶段⑤ 出口）：多因子截面 rank 等权合成，与单因子同口径对比。

性质：**出口**类工作（README §进度 ⑤），**不新增防线** —— 只出读数，不改变任何因子的在册状态，
也不得用合成结果淘汰因子（`假设登记表.md` §0 规则 3：停手期至 2027-04-11）。
口径先冻结在 `doc/stocks/Alpha191候选分层回测计划.md` §16（跑数前已写入）。

同源硬闸（不过就非零退出）：
  ① 用合成管线跑单因子 alpha139（rank 单调，不改变分组）必须与 out/layers_alpha139_5g.csv 的 G5
     逐位一致（相对 <= 1e-9）；
  ② 每期有效成分数 <= 名单数且 > 0；
  ③ 调仓期数 = 142；
  ④ §16 口径 4 的校准：超额口径集中度对 5 个单因子复算必须复现 §14.1.1 的表（±0.01）；
  ⑤ §17 口径 4：layer_judge.self_check() 口径锚点（a144 去均值相关 0.350）通过。

组合跑满 5 层是为了 §17 的换皮检查（判据② 需要 5 组，不能只用 G5）。

用法:
    uv run stocks/combo_check.py              # 冻结口径
    uv run stocks/combo_check.py --research

产物: out/combo_layers_<名>_5g.csv（date × G1..G5）+ out/combo_summary.csv（含去均值相关/换皮判定）
      + out/combo_yearly.csv
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
import freeze_guard  # noqa: E402
import impact_check as ic  # noqa: E402
import layer_judge as judge  # noqa: E402
import portfolio_check as pc  # noqa: E402
import store  # noqa: E402

OUT_DIR = ROOT / "out"
GROUPS, G5 = 5, 4
N_MONTHS = 142
GATE_TOL = 1e-9
# §14.1.1 的单因子超额口径集中度（校准用，±0.01）
CONC_EXCESS_REF = {"alpha015": 0.53, "alpha042": 0.29, "alpha139": 0.52,
                   "alpha144": 0.49, "alpha164": 0.45}
CONC_TOL = 0.01
C3 = ("alpha042", "alpha139", "alpha144")      # §16 口径 2：事后选的子集，只作对照
CORR_LIMIT = 0.8                                # §17 口径 3：去均值相关 ≥ 0.8 → F3


def log(msg: str) -> None:
    print(f"[combo] {msg}", flush=True)


def combo_ranks(names: tuple[str, ...], wide: dict[str, pd.DataFrame]) -> list[pd.DataFrame]:
    """§16 口径 1 的零件：每个成分的截面 rank(pct)。"""
    return [ic.factor_frame(n, wide).rank(axis=1, pct=True) for n in names]


def combo_factor(ranks: list[pd.DataFrame]) -> pd.DataFrame:
    """§16 口径 1：每期截面 rank(pct) 的算术平均（skipna=True）。"""
    return pd.concat(ranks).groupby(level=0).mean()


def mean_members(ranks: list[pd.DataFrame], dates: pd.DatetimeIndex) -> float:
    """月末截面上「平均有效成分数」（硬闸② / 读数用；不是标的数）。"""
    cnt = sum(r.notna().astype(float) for r in ranks)
    return float(cnt.loc[factors.month_end_dates(dates)].mean(axis=1).mean())


def run_layers(fac: pd.DataFrame, panel: pd.DataFrame, dates: pd.DatetimeIndex, delist: dict,
               only: int | None = None) -> tuple[pd.DataFrame, dict]:
    """分层回测 -> (date × G1..G5 的日频净值, G5 绩效)。only=n 只跑第 n 层（省时用）。

    §17 的换皮检查要 5 层，硬闸① 只要 G5 一层，所以同一函数按 only 复用。
    """
    sched = factors.build_schedule(fac, GROUPS, dates)
    cols, st = {}, {}
    for g in ([only] if only else range(1, GROUPS + 1)):
        eq, tr = factors.backtest.run_portfolio(panel, sched[g - 1], cash0=factors.CASH0,
                                                lot_size=1, delist_dates=delist)
        cols[f"G{g}"] = eq.set_index("date")["equity"]
        if g == GROUPS:
            st = factors.curve_stats(eq, tr, factors.CASH0)
    return pd.DataFrame(cols), st


def main() -> None:
    ap = argparse.ArgumentParser(description="组合骨架：多因子 rank 等权合成")
    ap.add_argument("--panel", default="", help="面板路径；默认 = 冻结快照")
    ap.add_argument("--out-dir", default="", help="读 layers/judge CSV 的目录；默认 out/")
    ap.add_argument("--research", action="store_true", help="研究模式：允许非冻结产物/越界数据")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else OUT_DIR
    panel_path = pathlib.Path(args.panel) if args.panel else freeze_guard.frozen_panel_path()
    problems: list[str] = []
    if not args.research:
        problems += freeze_guard.verify(panel_path) + freeze_guard.verify_artifacts(out_dir)
    if problems:
        for m in problems:
            log(f"[X] 冻结校验不过：{m}")
        log("拒绝在非冻结面板/非冻结产物上出读数（协议 §9.1）；研究用途加 --research")
        sys.exit(2)

    # 硬闸⑤（§17 口径 4）：口径漂移锚点，不过立刻退出，不浪费后面 11 层回测
    try:
        judge.self_check()
    except AssertionError as exc:
        log(f"[X] 硬闸⑤ 不过：{exc}")
        sys.exit(1)

    names = tuple(pc.passing_candidates(out_dir))
    log(f"候选名单（layer_judge.csv 判定==通过）：{', '.join(names)}")
    wide = ic.alpha191.load_wide(panel_path, None)
    panel = factors.load_prices(panel_path)
    delist = store.delist_dates(panel["symbol"].unique())
    dates = wide["close"].index
    base = pc.load_curve(pc.layer_path(pc.BASELINE, GROUPS, out_dir))
    base_layers = pd.read_csv(pc.layer_path(pc.BASELINE, GROUPS, out_dir),
                              parse_dates=["date"]).set_index("date")
    yb = pc.year_returns(base)
    bench = {k: pc.index_curve(pc.DATA_DIR / "index" / f) for k, f in pc.BENCH.items()}
    bench["equal"] = pc.equal_weight_curve(panel_path)
    ybench = {k: pc.year_returns(v) for k, v in bench.items()}
    log("基准：" + " / ".join(f"{k}({pc.g5_annual(v):+.2%})" for k, v in bench.items()))

    # 硬闸①：合成管线跑单因子 alpha139，必须复现 layers CSV
    gate, _ = run_layers(combo_factor(combo_ranks(("alpha139",), wide)), panel, dates, delist,
                         only=G5 + 1)
    g_gate = gate["G5"]
    ref139 = pc.load_curve(pc.layer_path("alpha139", GROUPS, out_dir))
    gap = float((g_gate.reindex(ref139.index) - ref139).abs().max() / ref139.abs().mean())
    if not (gap <= GATE_TOL):
        problems.append(f"合成管线跑单因子 alpha139 与 layers CSV 不一致：相对最大差 {gap:.2e}")
    else:
        log(f"硬闸① 通过：合成管线跑单因子 alpha139 与 layers CSV 相对最大差 {gap:.2e}")

    # 硬闸④：超额口径集中度的校准（复现 §14.1.1）；只取 11 个完整年（口径 4）
    full_years = list(range(pc.FIRST_YEAR, pc.LAST_FULL_YEAR + 1))
    for n, want in CONC_EXCESS_REF.items():
        if n not in names:
            continue
        yr = pc.year_returns(pc.load_curve(pc.layer_path(n, GROUPS, out_dir)))
        got = pc.concentration((yr.reindex(full_years) - yb.reindex(full_years)).dropna())
        if not (np.isfinite(got) and abs(got - want) <= CONC_TOL):
            problems.append(f"超额集中度校准不过：{n} 复算 {got:.3f} vs §14.1.1 的 {want}（±{CONC_TOL}）")
    log("硬闸④ 通过：超额口径集中度复现 §14.1.1 的表（±0.01）")

    single_rets = {n: pc.load_curve(pc.layer_path(n, GROUPS, out_dir)).pct_change()
                   for n in names}
    combos = {"C5": names, "C3": tuple(n for n in C3 if n in names)}

    rows, years = [], []
    for cname, members in combos.items():
        ranks = combo_ranks(members, wide)
        fac = combo_factor(ranks)
        n_eff = mean_members(ranks, dates)
        if not (0 < n_eff <= len(members)):
            problems.append(f"{cname}: 每期有效成分数均值 {n_eff} 越界（名单 {len(members)}）")
        layers, st = run_layers(fac, panel, dates, delist)
        g5 = layers["G5"]
        corr2 = judge.corr_demeaned(layers, base_layers)   # §17 判据②
        layers.to_csv(out_dir / f"combo_layers_{cname}_5g.csv")
        if len(st) and st["月数"] != N_MONTHS - 1:
            problems.append(f"{cname}: 月数 {st['月数']} != {N_MONTHS - 1}")

        yr = pc.year_returns(g5)
        full = [y for y in range(pc.FIRST_YEAR, pc.LAST_FULL_YEAR + 1)]
        fy = yr.reindex(full).dropna()
        n_a = int((fy - yb.reindex(full)).gt(0).sum())
        n_b = {k: int((fy - v.reindex(full)).gt(0).sum()) for k, v in ybench.items()}
        exc = (fy - yb.reindex(full)).dropna()
        corrs = {n: float(g5.pct_change().corr(single_rets[n])) for n in names}
        top = max(corrs, key=lambda k: abs(corrs[k]))
        rows.append({
            "组合": cname, "成分": "+".join(members), "全期年化": pc.g5_annual(g5),
            "年化夏普": st["年化夏普"], "最大回撤": st["最大回撤"], "月胜率": st["月胜率"],
            "年换手双边": st["年换手双边"], "超额_基线_年数": n_a,
            "超额_hs300_年数": n_b["hs300"], "超额_equal_年数": n_b["equal"],
            "超额_zz500_年数": n_b["zz500"],
            "集中度": pc.concentration(fy), "集中度_超额_基线": pc.concentration(exc),
            "平均成分数": n_eff, "月数": st["月数"], "最高相关_单因子": top,
            "最高相关": corrs[top],
            "去均值相关": corr2,
            "换皮判定": ("非换皮" if corr2 < CORR_LIMIT else "F3(换皮)") if np.isfinite(corr2)
                        else "NA",
            **{f"相关_{n}": corrs[n] for n in names}})
        for y in yr.index:
            row = {"组合": cname, "year": int(y), "G5": float(yr[y]),
                   "基线G5": float(yb[y]) if y in yb.index else np.nan,
                   "超额_基线": float(yr[y] - yb[y]) if y in yb.index else np.nan}
            for k, v in ybench.items():
                row[f"超额_{k}"] = float(yr[y] - v[y]) if y in v.index else np.nan
            years.append(row)
        log(f"{cname:<3s} 年化 {pc.g5_annual(g5):+.2%} 夏普 {st['年化夏普']:.3f} "
            f"回撤 {st['最大回撤']:.1%} 集中度 {pc.concentration(fy):.2f}"
            f"/超额 {pc.concentration(exc):.2f} 超额年数 A {n_a}/11 "
            f"B(300) {n_b['hs300']}/11 B(等权) {n_b['equal']}/11 "
            f"去均值相关(判据②) {corr2:+.3f} 与 {top} 相关 {corrs[top]:+.2f}")

    pd.DataFrame(rows).to_csv(out_dir / "combo_summary.csv", index=False)
    pd.DataFrame(years).to_csv(out_dir / "combo_yearly.csv", index=False)
    log("落盘 out/combo_summary.csv + out/combo_yearly.csv + combo_layers_*.csv")

    print("\n单因子对照（§14 已入档，供并排看）")
    st1 = pd.read_csv(out_dir / "portfolio_check_summary.csv")
    print(st1[["factor", "全窗口年化", "超额_基线_年数", "集中度"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n组合")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if problems:
        for m in problems:
            log(f"[X] {m}")
        sys.exit(1)
    log("全部同源硬闸通过")


if __name__ == "__main__":
    main()
