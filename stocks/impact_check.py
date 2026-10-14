"""冲击成本估算（阶段⑤ 出口）：静态参与率下的未成交占比 + 容量曲线。

性质：**出口**类工作（README §进度 ⑤），**不新增防线** —— 只出容量读数，不改变任何因子的
在册状态，也不得用未成交占比淘汰因子（`假设登记表.md` §0 规则 3：停手期至 2027-04-11）。
口径先冻结在 `doc/stocks/Alpha191候选分层回测计划.md` §15（跑数前已写入）。

同源硬闸（不过就非零退出）：
  ① 重跑 G5 层的全期年化必须与 out/layer_judge.csv 的「全期年化」一致（|Δ| <= 1e-9；
     基线 reversal20 对 0.0760 用 5e-4）；
  ② 调仓期数必须等于 out/layers_<name>_5g.csv 的月份数；
  ③ 买入笔数 > 0；amount 缺失/<=0 的成交笔数按 §15 口径 1 报出，不静默丢弃。

方向偏置（§15 口径 6）：需求只统计已成交的笔数（被涨跌停挡掉的不进分母）→ 未成交占比是偏乐观
的下界；顺延成本与冲击的价格反馈不在本次估算内。

用法:
    uv run stocks/impact_check.py              # 冻结口径
    uv run stocks/impact_check.py --research   # 研究口径（非冻结产物）

产物: out/impact_check.csv + out/impact_check_summary.csv
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import alpha191  # noqa: E402
import backtest  # noqa: E402
import factors  # noqa: E402
import freeze_guard  # noqa: E402
import portfolio_check as pc  # noqa: E402
import store  # noqa: E402

OUT_DIR = ROOT / "out"
DATA_DIR = ROOT / "data"
GROUPS = 5
G5 = GROUPS - 1
RATES = (0.01, 0.05, 0.10)          # §15 口径 3：三档参与率，并列报，不当阈值
CAP_LINES = (0.01, 0.05)            # §15 口径 5：容量定义线
ANN_TOL = 1e-9
BASE_TOL = 5e-4


def log(msg: str) -> None:
    print(f"[impact] {msg}", flush=True)


def load_amount(panel_path: pathlib.Path) -> pd.DataFrame:
    """长表 date/symbol/amount（元，真实成交口径）。"""
    return duckdb.sql(
        f"select date, symbol, amount from read_parquet('{panel_path.as_posix()}')"
    ).df()


def factor_frame(name: str, wide: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """候选因子值；基线 reversal20 = -20 日动量（与 layer_judge 同口径）。"""
    if name == pc.BASELINE:
        return -factors.momentum(wide["close"], 20)
    o, h, l, c, v, a, r = (wide["open"], wide["high"], wide["low"], wide["close"],
                          wide["volume"], wide["amount"], wide["returns"])
    return alpha191.ALPHAS[name](o, h, l, c, v, a, r)


def unfilled(demand: np.ndarray, cap: np.ndarray, scale: float) -> float:
    """未成交金额 = Σ max(0, 需求×scale − 上限)。"""
    return float(np.maximum(demand * scale - cap, 0.0).sum())


def capacity(demand: np.ndarray, cap: np.ndarray, th: float) -> float:
    """未成交占比 <= th 的最大本金（未成交占比在 k 上单调不减，二分）。"""
    total = float(demand.sum())
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    lo, hi = 0.0, 1.0
    while unfilled(demand, cap, hi) / (total * hi) <= th:
        lo, hi = hi, hi * 4.0
        if hi > 1e9:
            return factors.CASH0 * hi
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if unfilled(demand, cap, mid) / (total * mid) <= th:
            lo = mid
        else:
            hi = mid
    return factors.CASH0 * lo


def one_candidate(name: str, wide: dict[str, pd.DataFrame], panel: pd.DataFrame,
                  amounts: pd.DataFrame, delist: dict, ref: float,
                  problems: list[str]) -> tuple[dict, list[dict]]:
    """跑一个因子的 G5 层，返回 (全期读数, 逐年读数)。错进 problems，不抛。"""
    fac = factor_frame(name, wide).replace([np.inf, -np.inf], np.nan)
    sched = factors.build_schedule(fac, GROUPS, wide["close"].index)[G5]
    eq, tr = backtest.run_portfolio(panel, sched, cash0=factors.CASH0, lot_size=1,
                                    delist_dates=delist)
    ann = pc.g5_annual(eq.set_index("date")["equity"])

    # 硬闸①
    tol = BASE_TOL if name == pc.BASELINE else ANN_TOL
    if not (np.isfinite(ann) and abs(ann - ref) <= tol):
        problems.append(f"{name}: 重跑 G5 年化 {ann:.8f} 与参考 {ref:.8f} 差 > {tol:g}")

    # 硬闸②：调仓期数 vs layers CSV 的月份数
    curve = pd.read_csv(OUT_DIR / f"layers_{name}_5g.csv", parse_dates=["date"])
    n_months = int(curve.set_index("date")["G5"].resample("ME").last().dropna().shape[0])
    if len(sched) != n_months:
        problems.append(f"{name}: 调仓期数 {len(sched)} != layers CSV 月份数 {n_months}")

    # 硬闸③：需求金额 = 实际成交额；amount 缺失/异常单列报出
    tr = tr.copy()
    tr["gross"] = tr["shares"].astype(float) * tr["price"].astype(float)
    tr = tr.merge(amounts, on=["date", "symbol"], how="left")
    bad = int((~np.isfinite(tr["amount"]) | (tr["amount"] <= 0)).sum())
    tr = tr[np.isfinite(tr["amount"]) & (tr["amount"] > 0)]
    n_buy = int((tr["side"] == "buy").sum())
    if n_buy == 0:
        problems.append(f"{name}: 买入笔数 0，口径接错")

    demand = tr["gross"].to_numpy(dtype=float)
    caps = tr["amount"].to_numpy(dtype=float)[:, None] * np.array(RATES)[None, :]
    total = float(demand.sum())
    row = {"factor": name, "全期年化": ann, "参考年化": ref, "调仓期数": len(sched),
           "买入笔数": n_buy, "卖出笔数": int((tr["side"] == "sell").sum()),
           "成交额_亿": total / 1e8, "缺amount笔数": bad}
    for i, r in enumerate(RATES):
        row[f"未成交占比@{r:.0%}"] = unfilled(demand, caps[:, i], 1.0) / total
    # 容量：三档参与率 × 两条线，全部出（不挑单一组合）
    for i, r in enumerate(RATES):
        for th in CAP_LINES:
            row[f"容量@{th:.0%}_参与{r:.0%}_亿"] = capacity(demand, caps[:, i], th) / 1e8

    years: list[dict] = []
    yr = tr["date"].dt.year
    for y in sorted(yr.unique()):
        m = (yr == y).to_numpy()
        d, c = demand[m], tr["amount"].to_numpy(dtype=float)[m]
        if not np.isfinite(d.sum()) or d.sum() <= 0:
            continue
        years.append({"factor": name, "year": int(y), "成交额_亿": float(d.sum()) / 1e8,
                      "未成交占比@5%": unfilled(d, c * RATES[1], 1.0) / float(d.sum())})
    log(f"{name:<10s} 年化 {ann:+.4%} 调仓 {len(sched)} 期 成交额 {total / 1e8:.1f} 亿 "
        f"缺amount {bad} 笔")
    return row, years


def main() -> None:
    ap = argparse.ArgumentParser(description="冲击成本估算：静态参与率未成交占比 + 容量曲线")
    ap.add_argument("--panel", default="", help="面板路径；默认 = 冻结快照")
    ap.add_argument("--out-dir", default="", help="读 layers CSV / layer_judge.csv 的目录；默认 out/")
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

    log(f"面板 {panel_path.name}，冻结边界 {freeze_guard.FREEZE_END}")
    judge = pd.read_csv(out_dir / "layer_judge.csv").set_index("factor")
    names = sorted(judge.index[judge["判定"] == "通过"].tolist()) + [pc.BASELINE]
    wide = alpha191.load_wide(panel_path, None)
    panel = factors.load_prices(panel_path)
    amounts = load_amount(panel_path)
    delist = store.delist_dates(panel["symbol"].unique())
    log(f"候选 {len(names) - 1} 个 + 基线 {pc.BASELINE}；退市股 {len(delist)} 只")

    rows, years = [], []
    for name in names:
        ref = pc.BASELINE_ANNUAL if name == pc.BASELINE else float(judge.loc[name, "全期年化"])
        r, y = one_candidate(name, wide, panel, amounts, delist, ref, problems)
        rows.append(r)
        years.extend(y)

    summary = pd.DataFrame(rows)
    yearly = pd.DataFrame(years)
    summary.to_csv(out_dir / "impact_check_summary.csv", index=False)
    yearly.to_csv(out_dir / "impact_check.csv", index=False)
    log(f"落盘 {out_dir / 'impact_check_summary.csv'} + impact_check.csv")

    show = ["factor", "全期年化"] + [f"未成交占比@{r:.0%}" for r in RATES]
    print(summary[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    for th in CAP_LINES:
        cols = [f"容量@{th:.0%}_参与{r:.0%}_亿" for r in RATES]
        print(f"\n容量（未成交占比 <= {th:.0%} 的最大本金，亿）")
        print(summary[["factor"] + cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if problems:
        for m in problems:
            log(f"[X] {m}")
        sys.exit(1)
    log("全部同源硬闸通过")


if __name__ == "__main__":
    main()
