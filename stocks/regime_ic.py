"""H12 外部状态 -> 因子 IC 条件回归（消费 stocks/fetch_regime.py 产出的 regime 表）。

对 14 条逐月 IC 序列（13 个 Alpha191 候选 + 反转基线 reversal20）× 6 个外部状态变量，
逐对做 OLS：IC_t ~ a + b * regime_t。判据在跑之前已写死
（doc/stocks/外部状态条件回归计划.md §5 + doc/stocks/假设登记表.md H12）：

  N = 84 次检验，Bonferroni：双侧 p <= 0.05/84（等价写法：正态置信 >= 1 - 0.05/84 = 0.999405）
  五档判定（任一成立即证伪该 (因子, 状态) 对）：
    F1  双侧 p > 0.05/84                    没过多重检验
    F5  |b_full| <= surrogate 95 分位        幅度不超噪声底
    F2  p_train < 0.05 且 |t_test| < 1      样本外归零
    F4  sign(b_train) != sign(b_test)       跨期符号相反
    存活 = 其余
  多档同中时的归档优先级（只影响计数口径，不改判据本身）：F1 > F5 > F2 > F4。

分列报告（计划 §7.2 拍板，fetch_regime.py 落地为「陈旧丢弃 + 不截齐」）：
  main  家族 = 5 个状态 x 14 因子 = 70 对（状态非空 >= 130 月）
  cu_au 家族 = cu_au_ratio_chg x 14 = 14 对（HG/GC 只有 2016-10 起，约 119 月）
  两个家族共用同一预登记门槛 0.05/84 —— 判据不许事后改，分列只分样本窗、不分阈值。

surrogate：在该对的共同窗口内循环平移状态序列 300 次（保自相关、破与 IC 的真实对应），
同套回归得 |b| 零分布取 95 分位。方法同 leadlag.py / timing.py。

口径声明：IC 序列来自冻结面板（协议 §9.1），本计划冻结前已启动，
结论一律算「研究数据（非独立验证）」，不算干净窗口。

用法:
    uv run stocks/regime_ic.py          # 写 stocks/out/regime_ic.csv
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"
DATA_DIR = ROOT / "data"
REGIME_CSV = DATA_DIR / "regime_monthly.csv"
IC_SERIES_CSV = OUT_DIR / "alpha191_ic_series.csv"
BASELINE_CSV = OUT_DIR / "ic_reversal20.csv"
IC_TABLE_CSV = OUT_DIR / "alpha191_ic.csv"
RESULT_CSV = OUT_DIR / "regime_ic.csv"

SPLIT = pd.Timestamp("2020-12-31")   # 该日含在训练段（与 alpha191.py --split 默认一致）
N_TOTAL = 84                          # 14 因子 x 6 状态
P_BONF = 0.05 / N_TOTAL               # 5.952e-4，双侧
N_SURROGATE = 300
SEED = 20261013
CU_AU = "cu_au_ratio_chg"
REGIMES = ["cftc_usd_net_ratio", "lme_cu_cancel_ratio", "us_10y2y",
           "us_10y_chg", "wti_ret", CU_AU]
FAMILY = {r: ("cu_au" if r == CU_AU else "main") for r in REGIMES}


def load_ic_frames() -> dict[str, tuple[pd.Series, str]]:
    """13 候选（alpha191_ic_series.csv）+ 基线（ic_reversal20.csv）-> {名: (序列, 角色)}。"""
    cands = pd.read_csv(IC_SERIES_CSV, index_col=0, parse_dates=True)
    base = pd.read_csv(BASELINE_CSV, index_col=0, parse_dates=True)["ic"]
    if len(cands.columns) != 13:
        raise SystemExit(f"候选序列应为 13 列，实得 {len(cands.columns)}")
    frames = {c: (cands[c], "候选") for c in cands.columns}
    frames["reversal20"] = (base, "基线")
    return frames


def cross_check(frames: dict[str, tuple[pd.Series, str]]) -> None:
    """验收步 1：ic_series 测试段均值与 alpha191_ic.csv 的 ic_test 逐位相等（容差 1e-12）。"""
    ref = pd.read_csv(IC_TABLE_CSV).set_index("factor")
    bad = []
    for name, (ic, _) in frames.items():
        if name == "reversal20":
            continue
        seg = ic.dropna()
        seg = seg[seg.index > SPLIT]
        got, want = float(seg.mean()), float(ref.loc[name, "ic_test"])
        if not abs(got - want) <= 1e-12:
            bad.append((name, got, want))
    if bad:
        for name, got, want in bad:
            print(f"[对账] 不符 {name}: ic_series 测试段 {got!r} vs ic_test {want!r}")
        raise SystemExit("[对账] 失败：IC 序列导出与 147 行筛选表不一致，先查导出口径")
    print("[对账] 13 个候选测试段 IC 均值与 alpha191_ic.csv 的 ic_test 全部一致（容差 1e-12）")


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """IC ~ a + b*x 的 (b, t, 双侧p)。p 用正态近似（math.erfc），不引 scipy。"""
    n = len(x)
    xc, yc = x - x.mean(), y - y.mean()
    sxx = float((xc * xc).sum())
    b = float((xc * yc).sum()) / sxx
    resid = y - (float(y.mean()) - b * float(x.mean())) - b * x
    se = math.sqrt(float((resid * resid).sum()) / (n - 2) / sxx)
    t = b / se
    return b, t, math.erfc(abs(t) / math.sqrt(2.0))


def pair_stats(ic: pd.Series, reg: pd.Series, rng: np.random.Generator) -> dict:
    """单对 (因子 IC, 状态)：全段 + 训练/测试 OLS + 循环平移 surrogate 的 |b| 95 分位。"""
    j = pd.concat({"ic": ic, "x": reg}, axis=1, join="inner").dropna()
    x = j["x"].to_numpy(float)
    y = j["ic"].to_numpy(float)
    n = len(j)
    m_tr = np.asarray(j.index <= SPLIT)
    if n < 36 or m_tr.sum() < 12 or (~m_tr).sum() < 12:
        raise SystemExit(f"样本不足：n={n} train={int(m_tr.sum())} test={int((~m_tr).sum())}")
    b, t, p = ols(x, y)
    b_tr, t_tr, p_tr = ols(x[m_tr], y[m_tr])
    b_te, t_te, p_te = ols(x[~m_tr], y[~m_tr])
    bs = np.empty(N_SURROGATE)
    for i in range(N_SURROGATE):
        bs[i] = ols(np.roll(x, int(rng.integers(1, n))), y)[0]
    p95 = float(np.percentile(np.abs(bs), 95))
    sign_same = (b_tr > 0) == (b_te > 0)
    if p > P_BONF:
        verdict = "F1"
    elif abs(b) <= p95:
        verdict = "F5"
    elif p_tr < 0.05 and abs(t_te) < 1.0:
        verdict = "F2"
    elif not sign_same:
        verdict = "F4"
    else:
        verdict = "存活"
    return {
        "n": n, "n_train": int(m_tr.sum()), "n_test": int((~m_tr).sum()),
        "first": j.index.min().date(), "last": j.index.max().date(),
        "b_full": b, "t_full": t, "p_full": p, "bonferroni": int(p <= P_BONF),
        "b_train": b_tr, "t_train": t_tr, "p_train": p_tr,
        "b_test": b_te, "t_test": t_te, "p_test": p_te,
        "sign_same": int(sign_same), "surrogate_p95": p95, "verdict": verdict,
    }


def main() -> None:
    reg = pd.read_csv(REGIME_CSV, index_col=0, parse_dates=True)
    missing = [c for c in REGIMES if c not in reg.columns]
    if missing:
        raise SystemExit(f"regime_monthly.csv 缺列：{missing}")
    for r in REGIMES:
        if FAMILY[r] == "main" and int(reg[r].notna().sum()) < 130:
            raise SystemExit(f"主家族状态 {r} 非空 {int(reg[r].notna().sum())} < 130")

    frames = load_ic_frames()
    cross_check(frames)
    print(f"[H12] 因变量 {len(frames)} 个（13 候选 + 基线）x 自变量 {len(REGIMES)} 个 = {N_TOTAL} 对")

    rng = np.random.default_rng(SEED)
    rows = []
    for fname, (ic, role) in frames.items():
        for rname in REGIMES:
            st = pair_stats(ic, reg[rname], rng)
            rows.append({"family": FAMILY[rname], "factor": fname, "role": role,
                         "regime": rname, **st})
    res = pd.DataFrame(rows)
    if len(res) != N_TOTAL:
        raise SystemExit(f"行数 {len(res)} != {N_TOTAL}")
    counts = res["verdict"].value_counts().to_dict()
    if sum(counts.values()) != N_TOTAL:
        raise SystemExit("五档计数合计 != 84")
    res.to_csv(RESULT_CSV, float_format=None)
    print(f"[H12] {len(res)} 行 -> {RESULT_CSV}")

    print("\n===== 五档计数 =====")
    for fam in ("main", "cu_au"):
        sub = res[res["family"] == fam]
        vc = sub["verdict"].value_counts().to_dict()
        print(f"{fam:6s} {len(sub)} 对: " +
              " ".join(f"{k}={vc.get(k, 0)}" for k in ("存活", "F1", "F5", "F2", "F4")))
    vc = res["verdict"].value_counts().to_dict()
    print(f"合计   {len(res)} 对: " +
          " ".join(f"{k}={vc.get(k, 0)}" for k in ("存活", "F1", "F5", "F2", "F4")))

    surv = res[res["verdict"] == "存活"]
    print(f"\n===== 存活对（{len(surv)}）=====")
    if len(surv):
        for _, r in surv.iterrows():
            print(f"{r['factor']:>10s}({r['role']}) x {r['regime']:20s} "
                  f"b={r['b_full']:+.4f} t={r['t_full']:+.2f} n={r['n']} "
                  f"p95={r['surrogate_p95']:.4f} [{r['first']}~{r['last']}]")
    else:
        print("（无）")

    base = res[res["role"] == "基线"]
    print("\n===== 基线 reversal20 的 6 对（方法校准项，不计候选发现）=====")
    for _, r in base.iterrows():
        print(f"{r['regime']:20s} 判定={r['verdict']:3s} b={r['b_full']:+.4f} "
              f"t={r['t_full']:+.2f} |b|/p95={r['b_full'] / r['surrogate_p95']:.2f}")
    if (base["verdict"] == "存活").sum() == 0:
        print("基线 6 对全灭：读数说明方法在这批状态上没有功效，而非『因子都健康』（计划 §5.2）")


if __name__ == "__main__":
    main()
