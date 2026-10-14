"""组合体检（阶段⑤ 出口绩效归因）：基准对比 + 分年稳定性。

性质：**出口**类工作（README §进度 ⑤），**不新增防线** —— 只出绩效归因读数表，
不改变任何因子的在册状态（`假设登记表.md` §0 规则 3：停手期「精力转原料/出口」）。
三条记账线先冻结在 `doc/stocks/Alpha191候选分层回测计划.md` §14（跑数前已写入）。

同源硬闸（不过就非零退出）：
  ① 每个候选的全窗口年化必须与 out/layer_judge.csv 的「全期年化」逐位一致（1e-12）；
  ② 基线 reversal20 的 G5 年化必须等于 0.0760（计划 §10.5 的退市归零引擎值，5e-4）；
  ③ 冻结口径下先验产物哈希（freeze_guard.verify_artifacts + verify）。

用法:
    uv run stocks/portfolio_check.py              # 冻结口径
    uv run stocks/portfolio_check.py --research   # 研究口径（越界数据 / 非冻结产物）

产物: out/portfolio_check_yearly.csv + out/portfolio_check_summary.csv
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
import store  # noqa: E402

OUT_DIR = ROOT / "out"
DATA_DIR = ROOT / "data"

BASELINE = "reversal20"
BASELINE_ANNUAL = 0.0760          # 计划 §10.5：退市归零引擎下基线 G5 年化（结构自检）
FIRST_YEAR, LAST_FULL_YEAR = 2015, 2025
YEAR_LINE_A, YEAR_LINE_B, CONC_LINE = 7, 6, 0.50      # §14 记账线，先冻结
BENCH = {"hs300": "sh000300.parquet", "zz500": "sh000905.parquet"}


def layer_path(name: str, groups: int = 5, out_dir: pathlib.Path = OUT_DIR) -> pathlib.Path:
    return out_dir / f"layers_{name}_{groups}g.csv"


def passing_candidates(out_dir: pathlib.Path = OUT_DIR) -> list[str]:
    """过线名单的事实源：out/layer_judge.csv 的「判定 == 通过」（不写死）。"""
    d = pd.read_csv(out_dir / "layer_judge.csv")
    return sorted(d.loc[d["判定"] == "通过", "factor"])


def load_curve(path: pathlib.Path) -> pd.Series:
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")["G5"].dropna()


def year_returns(s: pd.Series) -> pd.Series:
    """自然年收益：该年末值 / 上年末值 - 1；首年用序列首值作基（与指数起点同日）。"""
    ye = s.resample("YE").last().dropna()
    base = pd.Series([float(s.iloc[0])], index=[pd.Timestamp(f"{s.index[0].year - 1}-12-31")])
    pts = pd.concat([base, ye])
    return pts.pct_change().dropna().rename(lambda t: t.year)


def g5_annual(s: pd.Series) -> float:
    """全窗口年化 —— 与 layer_judge.g5_annual 同口径（月末净值按总月数折算）。"""
    m = s.resample("ME").last().dropna()
    ret = m.pct_change().dropna()
    if len(ret) < 2:
        return float("nan")
    return float((m.iloc[-1] / m.iloc[0]) ** (1 / (len(ret) / 12)) - 1)


def equal_weight_curve(panel_path: pathlib.Path) -> pd.Series:
    """全市场等权再平衡指数（冻结面板现算，§14 口径 4）。

    每日收益 = 当日有价股票 close.pct_change() 的均值；退市股在退市日之后价格置 0
    （首次归零当日收益 -100%），与分层回测的退市归零引擎同口径。起点净值 1.0。
    """
    closes = factors.load_closes(panel_path).astype(float)
    for sym, dt in store.delist_dates(closes.columns).items():
        closes.loc[closes.index > pd.Timestamp(dt), sym] = 0.0
    rets = closes.pct_change()
    mkt = rets.mean(axis=1, skipna=True).fillna(0.0)
    return (1.0 + mkt).cumprod()


def index_curve(path: pathlib.Path) -> pd.Series:
    d = pd.read_parquet(path)
    return d.set_index("date")["close"].astype(float)


def concentration(yr: pd.Series) -> float:
    """集中度 = (C - C_去掉最好年) / C，C = 完整年累计收益。C <= 0 时无意义 -> NaN。"""
    c = float(np.prod(1.0 + yr.to_numpy()) - 1.0)
    if not np.isfinite(c) or c <= 0:
        return float("nan")
    i = int(np.argmax(yr.to_numpy()))
    c_wo = float(np.prod(1.0 + np.delete(yr.to_numpy(), i)) - 1.0)
    return (c - c_wo) / c


def main() -> None:
    ap = argparse.ArgumentParser(description="组合体检：基准对比 + 分年稳定性（出口绩效归因）")
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--out-dir", default="", help="读 layers CSV 的目录；默认 out/")
    ap.add_argument("--panel", default="", help="面板路径；默认 = 冻结快照")
    ap.add_argument("--research", action="store_true", help="研究模式：允许非冻结产物/越界数据")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else OUT_DIR
    panel_path = pathlib.Path(args.panel) if args.panel else freeze_guard.frozen_panel_path()

    if not args.research:
        problems = freeze_guard.verify(panel_path) + freeze_guard.verify_artifacts(out_dir)
        if problems:
            for m in problems:
                print(f"[port] [X] 冻结校验不过：{m}", file=sys.stderr)
            print("[port] 拒绝在非冻结面板/非冻结产物上出读数（协议 §9.1）；研究用途加 --research",
                  file=sys.stderr)
            sys.exit(1)
    else:
        print("[port] [!] 研究模式：产物标「研究数据（非独立验证）」，不得当独立验证（协议 §9.1）",
              file=sys.stderr)

    names = passing_candidates(out_dir)
    judge = pd.read_csv(out_dir / "layer_judge.csv").set_index("factor")
    print(f"[port] 候选名单（layer_judge.csv 判定==通过）：{', '.join(names)}")

    base = load_curve(layer_path(BASELINE, args.groups, out_dir))
    got_base = g5_annual(base)
    if abs(got_base - BASELINE_ANNUAL) > 5e-4:
        raise AssertionError(f"自检不过：基线 {BASELINE} G5 年化 {got_base:.6f} != "
                             f"{BASELINE_ANNUAL}（计划 §10.5），读错文件/列？")

    bench = {k: index_curve(DATA_DIR / "index" / f) for k, f in BENCH.items()}
    bench["equal"] = equal_weight_curve(panel_path)
    print(f"[port] 基准：{' / '.join(f'{k}({g5_annual(v):+.2%})' for k, v in bench.items())}")

    yearly_rows, summary_rows = [], []
    for name in names:
        cur = load_curve(layer_path(name, args.groups, out_dir))
        # 同源硬闸①：全窗口年化必须与冻结判据产物逐位一致
        want = float(judge.loc[name, "全期年化"])
        got = g5_annual(cur)
        if abs(got - want) > 1e-12:
            raise AssertionError(f"对账不过：{name} 年化 {got!r} != layer_judge.csv 的 {want!r}")

        yr = year_returns(cur)
        yb = year_returns(base)
        ybench = {k: year_returns(v) for k, v in bench.items()}
        full_years = [y for y in yr.index if FIRST_YEAR <= y <= LAST_FULL_YEAR]

        for y in yr.index:
            row = {"factor": name, "year": int(y), "完整年": bool(y in full_years),
                   "G5": float(yr[y]), "基线G5": float(yb.get(y, np.nan)),
                   "超额_基线": float(yr[y] - yb[y]) if y in yb.index else np.nan}
            for k, v in ybench.items():
                row[f"{k}"] = float(v[y]) if y in v.index else np.nan
                row[f"超额_{k}"] = float(yr[y] - v[y]) if y in v.index else np.nan
            yearly_rows.append(row)

        fy = yr.reindex(full_years).dropna()
        n_a = int((fy - yb.reindex(full_years)).gt(0).sum())
        n_b = {k: int((fy - v.reindex(full_years)).gt(0).sum()) for k, v in ybench.items()}
        conc = concentration(fy)
        ok_a = n_a >= YEAR_LINE_A
        ok_b = {k: v >= YEAR_LINE_B for k, v in n_b.items()}
        ok_c = bool(np.isfinite(conc) and conc <= CONC_LINE)
        summary_rows.append({
            "factor": name, "全窗口年化": got, "基线年化": got_base,
            "超额_基线_年数": n_a, f"超额_{'hs300'}_年数": n_b["hs300"],
            f"超额_{'equal'}_年数": n_b["equal"], f"超额_{'zz500'}_年数": n_b["zz500"],
            "集中度": conc,
            "达标A": ok_a, "达标B_沪深300": ok_b["hs300"], "达标B_等权": ok_b["equal"],
            "达标C": ok_c, "三条全达标": bool(ok_a and ok_b["hs300"] and ok_b["equal"] and ok_c),
        })
        flag = "." * 0
        print(f"[port] {name:<9s} 全期 {got:+.2%} | 超额年数 A(基线) {n_a}/11 "
              f"B(300) {n_b['hs300']}/11 B(等权) {n_b['equal']}/11 | 集中度 {conc:.2f} "
              f"-> {'全达标' if summary_rows[-1]['三条全达标'] else '未全达标'}{flag}")

    yd = pd.DataFrame(yearly_rows)
    sd = pd.DataFrame(summary_rows)
    OUT_DIR.mkdir(exist_ok=True)
    suffix = ".research" if args.research else ""
    y_dst = OUT_DIR / f"portfolio_check_yearly{suffix}.csv"
    s_dst = OUT_DIR / f"portfolio_check_summary{suffix}.csv"
    yd.to_csv(y_dst, index=False)
    sd.to_csv(s_dst, index=False)
    print(f"[port] 证据等级 {freeze_guard.verdict_level(args.research)}"
          f"（冻结 {freeze_guard.FREEZE_COMMIT[:7]} / {freeze_guard.FREEZE_TS}）")
    print(f"[port] {len(sd)} 个候选，三条全达标 {int(sd['三条全达标'].sum())} 个 -> {s_dst} + {y_dst}")


if __name__ == "__main__":
    main()
