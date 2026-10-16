"""两融事件口径：年度 / 时段分布落盘（证据修复阶段，只读重算）。

为什么单独存在：`margin_event_probe.py` 会把「-5% 事件按年分布」打印到 stdout，
但**没有落盘**（`stocks/out/` 下没有它的日志），所以「优势是否由少数年份主导」无法从任何产物核验。
本脚本把那部分算出来落成独立 CSV，**不改口径、不改事件定义、不动原始产物**。

严格边界（本脚本不是防线）：
  - 不做涨跌停过滤、不扣费、不引入小盘分组、不搜索参数
  - 不做显著性检验；年度拆分**不是**独立样本（exec5 是 5 日持有，相邻事件窗口重叠）
  - 不冻结任何判死判据、不写 H 卡（停手期至 2027-04-11，`假设登记表.md` §0 规则 3）

口径来源：直接 import `margin_event_probe` 的 `BASE_SQL` / `THRESHOLDS`，
保证与原始 CSV **逐位同源**；`self_check()` 用 `out/margin_event_probe.csv` 对账，不一致即非零退出。

归属与差异说明：
  - 事件按**事件日 T 的日历年份**归属；exec5 的收益窗口可能跨年，不因此改归属（与原始口径一致）
  - `n` = 行数；`n_exec5有效` = exec5 非空行数。原始 CSV 只报 `n`，
    而均值/中位数由 pandas 在**非空**子集上算 —— 两者在退市 / 停牌导致前向收益缺失时不相等
  - 前向收益缺失（`lead()` 为 null）= 该股在窗口内退出面板（退市股）或数据断档，不填补、不剔除行

用法:
    uv run stocks/margin_event_yearly.py
"""
from __future__ import annotations

import pathlib
import sys

import duckdb
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from margin_event_probe import BASE_SQL, PANEL, THRESHOLDS, STAT_COLS  # noqa: E402

OUT_DIR = ROOT / "out"
PROBE_CSV = OUT_DIR / "margin_event_probe.csv"
YEARLY_CSV = OUT_DIR / "margin_event_yearly.csv"
PERIOD_CSV = OUT_DIR / "margin_event_periods.csv"

# 固定时期划分：**跑数前写死**，不按结果调整
EARLY_YEARS = (2015, 2016)


def _stats(sub: pd.DataFrame) -> dict:
    rec = {"n": int(len(sub)), "n_exec5有效": int(sub["exec5"].notna().sum())}
    for c in STAT_COLS:
        rec[f"{c}_均值"] = float(sub[c].mean())
        rec[f"{c}_中位"] = float(sub[c].median())
    return rec


def self_check(base: pd.DataFrame) -> None:
    """与原始 margin_event_probe.csv 对账：同口径重算必须逐位一致。"""
    if not PROBE_CSV.exists():
        raise SystemExit(f"缺 {PROBE_CSV}，无法对账（先跑 margin_event_probe.py）")
    orig = pd.read_csv(PROBE_CSV)
    worst, where = 0.0, ""
    for th in THRESHOLDS:
        ev = base["chg1"] <= -th
        for label, mask in (("事件", ev), ("非事件", ~ev)):
            rec = _stats(base[mask])
            row = orig[(orig["阈值"] == -th) & (orig["组"] == label)]
            if len(row) != 1:
                raise SystemExit(f"对账失败：原 CSV 里 阈值={-th} 组={label} 命中 {len(row)} 行")
            row = row.iloc[0]
            for k, v in rec.items():
                if k == "n_exec5有效":
                    continue
                d = abs(float(row[k]) - v)
                if d > worst:
                    worst, where = d, f"阈值={-th} 组={label} {k}"
    assert worst < 1e-12, f"与原始 CSV 不一致：max|diff|={worst:.3e}（{where}）"
    print(f"[yearly] 自检通过：与 margin_event_probe.csv 逐位一致（max|diff|={worst:.1e}）")


def main() -> None:
    con = duckdb.connect()
    base = con.sql(BASE_SQL).df()
    base["date"] = pd.to_datetime(base["date"])
    base["year"] = base["date"].dt.year
    self_check(base)

    # -- 结构说明（缺失交易日 / 退市 / 跨年）--
    panel_days = con.sql(f"select count(distinct date) from read_parquet('{PANEL}')").fetchone()[0]
    margin_days = base["date"].nunique()
    miss = int(base["exec5"].isna().sum())
    print(f"\n[yearly] 面板交易日 {panel_days:,} / 事件表覆盖 {margin_days:,} 个交易日"
          f"（差 {panel_days - margin_days:,} 天 = 两融无该日数据或面板缺价）")
    print(f"[yearly] exec5 缺失（前向 5 日窗口触及面板尾部 / 退市 / 断档）："
          f"{miss:,} 行 = {miss / len(base):.3%} —— 不填补、不剔行")

    # -- 年度表（3 档阈值全出，不挑档；报告引的是 -0.05）--
    yearly_rows = []
    for th in THRESHOLDS:
        ev = base["chg1"] <= -th
        for year in sorted(base["year"].unique()):
            iny = base["year"] == year
            for label, mask in (("事件", ev), ("非事件", ~ev)):
                sub = base[iny & mask]
                rec = {"阈值": -th, "年": int(year), "组": label,
                       "占比": float(mask[iny].mean()), **_stats(sub)}
                yearly_rows.append(rec)
    yearly = pd.DataFrame(yearly_rows)

    # -- 全样本汇总（与年度表同源，供贡献分解的分母）--
    totals = {}
    for th in THRESHOLDS:
        ev = base["chg1"] <= -th
        totals[-th] = {"事件": _stats(base[ev]), "非事件": _stats(base[~ev])}

    # -- 贡献分解：mean_ev - mean_ne = Σ_y (w_y^ev·mean_ev_y - w_y^ne·mean_ne_y) --
    contrib_rows = []
    for th in THRESHOLDS:
        tot = totals[-th]
        n_ev_tot = tot["事件"]["n_exec5有效"]
        n_ne_tot = tot["非事件"]["n_exec5有效"]
        d_tot = tot["事件"]["exec5_均值"] - tot["非事件"]["exec5_均值"]
        for year in sorted(base["year"].unique()):
            y = yearly[(yearly["阈值"] == -th) & (yearly["年"] == year)]
            e, n = y[y["组"] == "事件"].iloc[0], y[y["组"] == "非事件"].iloc[0]
            c_ev = e["n_exec5有效"] / n_ev_tot * e["exec5_均值"]
            c_ne = n["n_exec5有效"] / n_ne_tot * n["exec5_均值"]
            contrib_rows.append({
                "阈值": -th, "年": int(year),
                "事件_exec5均值": e["exec5_均值"], "非事件_exec5均值": n["exec5_均值"],
                "年度均值差": e["exec5_均值"] - n["exec5_均值"],
                "事件_exec5中位": e["exec5_中位"], "非事件_exec5中位": n["exec5_中位"],
                "年度中位差": e["exec5_中位"] - n["exec5_中位"],
                "对全样本均值差的贡献": c_ev - c_ne,
                "贡献占比": (c_ev - c_ne) / d_tot,
            })
    contrib = pd.DataFrame(contrib_rows)
    assert abs(contrib.groupby("阈值")["贡献占比"].sum().sub(1.0).max()) < 1e-9, \
        "贡献分解未闭合（Σ贡献占比 应为 1）"

    # -- 固定时期对照：2015-2016 vs 其余（划分跑数前写死）--
    period_rows = []
    for th in THRESHOLDS:
        ev = base["chg1"] <= -th
        for name, mask_y in (("2015-2016", base["year"].isin(EARLY_YEARS)),
                             ("其余年份", ~base["year"].isin(EARLY_YEARS))):
            for label, mask in (("事件", ev), ("非事件", ~ev)):
                sub = base[mask_y & mask]
                period_rows.append({"阈值": -th, "时期": name, "组": label,
                                    "占比": float(mask[mask_y].mean()), **_stats(sub)})
    periods = pd.DataFrame(period_rows)

    OUT_DIR.mkdir(exist_ok=True)
    yearly.to_csv(YEARLY_CSV, index=False)
    periods.to_csv(PERIOD_CSV, index=False)

    # -- 终端摘要：只报 -0.05（报告引用的那一档），三档完整读数在 CSV --
    q = contrib[contrib["阈值"] == -0.05].sort_values("对全样本均值差的贡献", ascending=False)
    print("\n" + "=" * 78)
    print("exec5 事件-非事件 均值差：按年分解（阈值 -0.05）")
    print("=" * 78)
    print(f"{'年':>6} {'n事件':>9} {'均值差':>10} {'中位差':>10} {'贡献':>10} {'贡献占比':>9}")
    for _, r in q.iterrows():
        print(f"{r['年']:>6} {int(yearly[(yearly['阈值'] == -0.05) & (yearly['年'] == r['年']) & (yearly['组'] == '事件')]['n'].iloc[0]):>9,} "
              f"{r['年度均值差']:>+10.4f} {r['年度中位差']:>+10.4f} "
              f"{r['对全样本均值差的贡献']:>+10.4f} {r['贡献占比']:>8.1%}")
    d = totals[-0.05]["事件"]["exec5_均值"] - totals[-0.05]["非事件"]["exec5_均值"]
    print(f"\n全样本均值差 = {d:+.6f}（{d * 100:+.4f}pp）")
    print("\n-- 固定时期对照（-0.05，exec5）--")
    for name in ("2015-2016", "其余年份"):
        e = periods[(periods["阈值"] == -0.05) & (periods["时期"] == name) & (periods["组"] == "事件")].iloc[0]
        n = periods[(periods["阈值"] == -0.05) & (periods["时期"] == name) & (periods["组"] == "非事件")].iloc[0]
        print(f"  {name:<10} 事件 n={int(e['n']):>8,} 均值{e['exec5_均值']:+.4f} 中位{e['exec5_中位']:+.4f} | "
              f"非事件 均值{n['exec5_均值']:+.4f} 中位{n['exec5_中位']:+.4f} | "
              f"均值差{e['exec5_均值'] - n['exec5_均值']:+.4f} 中位差{e['exec5_中位'] - n['exec5_中位']:+.4f}")

    print(f"\n[yearly] 年度表 -> {YEARLY_CSV}")
    print(f"[yearly] 时期表 -> {PERIOD_CSV}")
    print("  证据等级：研究数据（非独立验证）；性质：证据修复（年度分布披露），"
          "不是判据，不得用于淘汰/纳入候选")
    print("  未做：涨跌停过滤、扣费、小盘分组、显著性检验、样本内外切分 —— 要当判据必须先补这些")


if __name__ == "__main__":
    main()
