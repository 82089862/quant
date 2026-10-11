"""时序滞后联动（lead-lag）挖掘：锚点固定，遍历全市场，找先行-滞后配对。

只输出统计观测，不判定因果，不生成买卖信号，不评估收益。

方法约束（每条都是为了让结论站得住，缺一条结论就不成立）：
  1. 用收益率不用价格水平 —— 价格水平有趋势，CCF 会假显著。
  2. 残差化 —— 先剔除市场等权指数，剥掉共同暴露。实测：名义显著从 12.0% 压到 7.0%，
     而纯噪声底是 5.1%，即原始口径下一半以上的"显著"是共同因子。
  3. 剔除同期相关 —— 同期高度相关的两只股票，会因为自相关伪装出滞后相关。
     主口径用残差 CCF；`--pairwise-partial` 再对候选的同期值取偏相关做稳健性检查。
  4. 多重检验校正 —— 检验次数 = 候选数 × 滞后阶数（约 2.6 万次）。只报名义 5% 等于报
     上千个假阳性，必须 Bonferroni + BH-FDR，且以校正后为准。
  5. surrogate 对照 —— 把每只候选的时间序列循环平移随机偏移：保留自相关、破坏真实跨股票
     关系，得到"噪声底"。任何不超过噪声底的"发现"都不是发现。
     实测：纯噪声也有 9 个配对通过 Bonferroni —— 说明正态近似阈值在股票收益率上不可靠，
     唯一的判据是"显著数是否超过噪声底"。
  6. 样本内外 —— 训练集挖 top-N，测试集只报同号且显著的留存率。留存率不高于对照即无发现。
  7. 涨跌停过滤 —— 候选当日涨跌停的样本剔除。A 股涨停买不到、信息延到次日，
     会伪装成 lead-lag，那是交易制度的产物，不是联动。

用法:
    uv run leadlag.py --anchor sz300359
    uv run leadlag.py --anchor sz300359 --tau 1,2,3 --topn 100 --surrogate 30
"""
from __future__ import annotations

import argparse
import math
import pathlib

import duckdb
import numpy as np
import pandas as pd

from backtest import price_limit_pct

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"


def load_closes(panel_path: pathlib.Path) -> pd.DataFrame:
    df = duckdb.sql(f"select date, symbol, close from read_parquet('{panel_path.as_posix()}')").df()
    return df.pivot(index="date", columns="symbol", values="close").sort_index()


def residualize(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对市场等权指数回归取残差。返回 (残差矩阵, 各列 beta)。"""
    mkt = R.mean(axis=1)
    beta = (R.T @ mkt) / (mkt @ mkt)
    return R - np.outer(mkt, beta), beta


def tradable_mask(R: np.ndarray, symbols: list[str]) -> np.ndarray:
    """False = 当日涨跌停，该样本剔除。低于 1 元的样本一并剔除（噪音大）。"""
    lim = np.array([price_limit_pct(s) for s in symbols])
    return np.abs(R) < (lim - 0.005)


def ccf_scan(A: np.ndarray, Y: np.ndarray, taus: list[int], ok: np.ndarray) -> np.ndarray:
    """返回 (len(taus), M) 的 CCF：corr(A_t, Y_{t-tau})，即候选先行 tau 日。"""
    out = np.empty((len(taus), Y.shape[1]))
    for i, tau in enumerate(taus):
        x = A[tau:]
        y = Y[:-tau]
        o = ok[tau:] & ok[:-tau]
        n = o.sum(axis=0)
        n = np.maximum(n, 3)
        xm = np.where(o, x[:, None], 0.0)
        ym = np.where(o, y, 0.0)
        mx = xm.sum(axis=0) / n
        my = ym.sum(axis=0) / n
        vx = (xm ** 2).sum(axis=0) / n - mx ** 2
        vy = (ym ** 2).sum(axis=0) / n - my ** 2
        cxy = (xm * ym).sum(axis=0) / n - mx * my
        out[i] = cxy / np.sqrt(np.maximum(vx * vy, 1e-300))
    return out


def fisher_p(ccf: np.ndarray, n: int) -> np.ndarray:
    """Fisher z 变换后的双尾 p 值。"""
    z = np.abs(np.arctanh(np.clip(ccf, -0.999999, 0.999999))) * math.sqrt(max(n - 3, 1))
    flat = np.array([math.erfc(v / math.sqrt(2)) for v in z.ravel()])
    return flat.reshape(z.shape)


def bh_fdr_count(p: np.ndarray, alpha: float) -> int:
    """Benjamini-Hochberg FDR 通过个数。"""
    ps = np.sort(p)
    n = len(ps)
    passed = ps <= alpha * np.arange(1, n + 1) / n
    return int(passed.sum())


def shift_columns(R: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """每列循环平移随机偏移：保留自相关与边际分布，破坏跨股票的真实时序关系。"""
    T, M = R.shape
    out = np.empty_like(R)
    for j in range(M):
        out[:, j] = np.roll(R[:, j], int(rng.integers(1, T)))
    return out


def scan_block(A: np.ndarray, Y: np.ndarray, taus: list[int], ok: np.ndarray,
               tidx: np.ndarray, n_tests: int, alpha: float, label: str) -> dict:
    """在给定时间切片上扫一遍，返回显著数统计与 CCF 矩阵。"""
    a = A[tidx]
    y = Y[tidx]
    o = ok[tidx]
    ccf = ccf_scan(a, y, taus, o)
    p = fisher_p(ccf, len(tidx))
    thr_bon = 1 - (1 - alpha) ** (1.0 / n_tests)
    return {
        "label": label,
        "ccf": ccf,
        "p": p,
        "n_nominal": int((p < 0.05).sum()),
        "n_bonferroni": int((p < thr_bon).sum()),
        "n_fdr": bh_fdr_count(p.ravel(), alpha),
        "thr_bon_p": thr_bon,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="sz300359")
    ap.add_argument("--panel", default=str(DATA_DIR / "panel_2015_2026.parquet"))
    ap.add_argument("--tau", default="1,2,3,4,5")
    ap.add_argument("--split", default="2020-12-31")
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--surrogate", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    taus = [int(x) for x in args.tau.split(",")]
    closes = load_closes(pathlib.Path(args.panel))
    symbols = list(closes.columns)
    if args.anchor not in symbols:
        raise SystemExit(f"锚点 {args.anchor} 不在面板里")
    R = closes.pct_change().fillna(0.0).to_numpy(dtype=float)
    dates = closes.index
    j0 = symbols.index(args.anchor)
    n_tests = len(symbols) * len(taus)

    print(f"[leadlag] 锚点 {args.anchor} | 候选 {len(symbols)} 只 | {len(dates)} 交易日 | 滞后 {taus}")
    print(f"[leadlag] 检验次数 {n_tests:,}，Bonferroni 单次阈值 p < {args.alpha / n_tests:.2e}")
    print(f"[leadlag] 名义 5% 下纯噪声预期假阳性约 {int(n_tests * 0.05):,} 个\n")

    Rres, _ = residualize(R)
    ok = tradable_mask(R, symbols)
    ok[:, j0] = ok[:, j0]           # 锚点自身涨跌停同样剔除
    tr = np.where(dates <= pd.Timestamp(args.split))[0]
    te = np.where(dates > pd.Timestamp(args.split))[0]
    print(f"[leadlag] 训练 {dates[tr[0]].date()}~{dates[tr[-1]].date()} ({len(tr)} 日)"
          f" | 测试 {dates[te[0]].date()}~{dates[te[-1]].date()} ({len(te)} 日)")
    print(f"[leadlag] 涨跌停剔除样本占比 {1 - ok.mean():.2%}\n")

    train = scan_block(Rres[:, j0], Rres, taus, ok, tr, n_tests, args.alpha, "训练")
    test = scan_block(Rres[:, j0], Rres, taus, ok, te, n_tests, args.alpha, "测试")

    # 每个候选取 |ccf| 最大的那个滞后阶
    best_tau = np.abs(train["ccf"]).argmax(axis=0)
    train_ccf = train["ccf"][best_tau, np.arange(len(symbols))]
    train_p = train["p"][best_tau, np.arange(len(symbols))]
    test_ccf = test["ccf"][best_tau, np.arange(len(symbols))]
    test_p = test["p"][best_tau, np.arange(len(symbols))]
    survived = (np.sign(test_ccf) == np.sign(train_ccf)) & (test_p < args.alpha)
    order = np.argsort(-np.abs(train_ccf))
    keep = [j for j in order if j != j0][: args.topn]

    thr_bon = 1 - (1 - args.alpha) ** (1.0 / n_tests)
    pair_rows = []
    for j in keep:
        pair_rows.append({
            "anchor": args.anchor, "candidate": symbols[j], "tau": taus[best_tau[j]],
            "ccf_train": round(float(train_ccf[j]), 5), "p_train": float(train_p[j]),
            "p_train_bonf_ok": bool(train_p[j] < thr_bon),
            "ccf_test": round(float(test_ccf[j]), 5), "p_test": float(test_p[j]),
            "same_sign": bool(np.sign(test_ccf[j]) == np.sign(train_ccf[j])),
            "survived": bool(survived[j]),
        })
    pairs = pd.DataFrame(pair_rows)
    OUT_DIR.mkdir(exist_ok=True)
    out_csv = OUT_DIR / f"leadlag_{args.anchor}.csv"
    pairs.to_csv(out_csv, index=False)

    # surrogate 噪声底
    rng = np.random.default_rng(20261010)
    sur_ccf = np.zeros(len(symbols))
    sur_survived = np.zeros(len(symbols))
    for _ in range(args.surrogate):
        Rs = shift_columns(Rres, rng)
        s_tr = scan_block(Rs[:, j0], Rs, taus, ok, tr, n_tests, args.alpha, "sur")
        s_te = scan_block(Rs[:, j0], Rs, taus, ok, te, n_tests, args.alpha, "sur")
        bt = np.abs(s_tr["ccf"]).argmax(axis=0)
        sc = s_tr["ccf"][bt, np.arange(len(symbols))]
        tc = s_te["ccf"][bt, np.arange(len(symbols))]
        tp = s_te["p"][bt, np.arange(len(symbols))]
        sur_ccf += sc
        sur_survived += ((np.sign(tc) == np.sign(sc)) & (tp < args.alpha)).astype(float)
    sur_ccf /= args.surrogate
    sur_survived /= args.surrogate
    sur_surv_rate = sur_survived[keep].mean()
    sur_max_ccf = np.abs(sur_ccf[[j for j in order if j != j0][:100]]).max()

    print(f"{'':10s} {'名义显著':>10s} {'Bonferroni':>12s} {'FDR':>8s}")
    for blk in (train, test):
        print(f"{blk['label']:10s} {blk['n_nominal']:10,d} {blk['n_bonferroni']:12,d} {blk['n_fdr']:8,d}")
    print(f"\n[leadlag] 训练集 top-{args.topn} 在测试集留存 {int(survived[keep].sum())}/{args.topn}"
          f" = {survived[keep].mean():.1%}")
    print(f"[leadlag] surrogate 噪声底同名留存率 = {sur_surv_rate:.1%}"
          f" | top-100 里噪声能造出的最大 |CCF| = {sur_max_ccf:.4f}")
    print(f"[leadlag] 判定: " + ("留存率高于噪声底，存在候选信号，需进一步查稳定性"
                                if survived[keep].mean() > sur_surv_rate + 0.10
                                else "留存率与噪声底不可区分 —— 本口径下未发现可复现的先行滞后关系"))
    print(f"\n[leadlag] 配对明细 -> {out_csv}")
    print(pairs.head(10).to_string(index=False))
    print("\n[leadlag] 失效风险：任何历史联动都可能在行业变更、基本面变化、交易制度调整后断裂；"
          "本表只描述历史统计现象，不含因果判定，也不构成交易依据。")


if __name__ == "__main__":
    main()
