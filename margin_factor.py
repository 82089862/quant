"""融资融券月度因子：先过验证协议，再谈收益。

与 factors.py 同口径：信号日 = 月末最后交易日收盘，执行日 = 下一交易日开盘。
融资余额/买入额是 T 日盘后数据，T+1 才可交易 —— 月末信号天然满足，无前视。

验证协议（缺一条结论就不成立）：
  1. 训练 2015-2020 / 测试 2021-2026，只在测试集看留存。
  2. 多重检验：5 个因子，Bonferroni 校正。
  3. surrogate 噪声底：每月内自行打乱因子值（保留因子边际分布、破坏因子-收益关系），
     跑同一套 IC 计算。任何不超过噪声底的"发现"都不是发现。
  4. 涨跌停剔除：信号日涨跌停的样本剔除（次日买不到）。
  5. 正交性预检已在 margin_probe.py 做过：m_buy_int 落在价格基上的 R²=0.014。

用法:
    uv run margin_factor.py
    uv run margin_factor.py --surrogate 500
"""
from __future__ import annotations

import argparse
import pathlib

import duckdb
import numpy as np
import pandas as pd

import factors
import margin_probe
import capacity
from backtest import price_limit_pct

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
CACHE_DIR = DATA_DIR / "margin_cache"

FACTORS = ["m_buy_int", "m_bal_chg", "m_short_ratio", "m_shortsell_int", "m_bal_ratio"]


def build_factor_panel(me: pd.DatetimeIndex, panel_path: pathlib.Path) -> pd.DataFrame:
    """抓月末两融数据，换算成 5 个因子（date, symbol, FACTORS...）。"""
    print(f"[mfac] 抓 {len(me)} 个月末的两融明细")
    frames = []
    for i, d in enumerate(me, 1):
        frames.append(margin_probe.fetch_one(d, CACHE_DIR))
        if i % 20 == 0 or i == len(me):
            print(f"[mfac] {i}/{len(me)}")
    margin = pd.concat(frames, ignore_index=True)

    lit = ", ".join(f"date '{pd.Timestamp(d).date()}'" for d in me)
    raw = duckdb.sql(f"""
        select date, symbol, close, volume, amount, outstanding_share
        from read_parquet('{panel_path.as_posix()}')
        where date in ({lit})
    """).df()

    m = margin.merge(raw, on=["date", "symbol"], how="inner")
    print(f"[mfac] 月末两融覆盖 {len(m):,} 行 / {m['symbol'].nunique()} 只"
          f" / {m['date'].nunique()} 个月末")
    mv = m["close"] * m["outstanding_share"]
    m["m_bal_ratio"] = m["融资余额"] / mv
    m["m_buy_int"] = m["融资买入额"] / m["amount"]
    m["m_short_ratio"] = m["融券余量"] / m["outstanding_share"]
    m["m_shortsell_int"] = m["融券卖出量"] / m["volume"]
    # shortcut: 个别月缺数据时 pct_change 会跨月，非严格月环比
    m = m.sort_values(["symbol", "date"])
    m["m_bal_chg"] = m.groupby("symbol")["融资余额"].pct_change(fill_method=None)
    return m[["date", "symbol"] + FACTORS]


def prep_blocks(fac: pd.DataFrame, fwd: pd.DataFrame, ok: pd.DataFrame,
                min_n: int = 30) -> list[tuple]:
    """每个月末预算 rank 并去均值。置换只改配对不改 rank，故 surrogate 只是重排一个向量。"""
    blocks = []
    for d in fwd.index:
        if d not in fac.index or d not in ok.index:
            continue
        f = fac.loc[d].dropna()
        r = fwd.loc[d].dropna()
        common = f.index.intersection(r.index)
        common = common[ok.loc[d].reindex(common).fillna(False).to_numpy()]
        if len(common) < min_n:
            continue
        ra = pd.Series(f.reindex(common).to_numpy()).rank().to_numpy()
        rb = pd.Series(r.reindex(common).to_numpy()).rank().to_numpy()
        blocks.append((d, ra - ra.mean(), rb - rb.mean()))
    return blocks


def ic_series(blocks: list[tuple], rng: np.random.Generator | None = None) -> np.ndarray:
    """逐期 Spearman IC。rng 非空时每期在截面内打乱因子配对。"""
    out = np.empty(len(blocks))
    for i, (_, ra, rb) in enumerate(blocks):
        x = rng.permutation(ra) if rng is not None else ra
        out[i] = (x @ rb) / np.sqrt((x @ x) * (rb @ rb))
    return out


def layered_returns(fac: pd.DataFrame, fwd: pd.DataFrame, ok: pd.DataFrame,
                    dates: pd.DatetimeIndex, n: int = 5) -> tuple[dict, float]:
    """n 层等权下一期收益（未扣成本），及各层均值、多空 t 值。"""
    buckets = {g: [] for g in range(n)}
    for d in dates:
        if d not in fac.index or d not in fwd.index or d not in ok.index:
            continue
        f = fac.loc[d].dropna()
        r = fwd.loc[d].dropna()
        common = f.index.intersection(r.index)
        common = common[ok.loc[d].reindex(common).fillna(False).to_numpy()]
        if len(common) < n * 10:
            continue
        lab = pd.qcut(f.reindex(common).rank(method="first"), n, labels=False)
        rr = r.reindex(common)
        for g in range(n):
            buckets[g].append(rr[lab == g].mean())
    means = {g: float(np.nanmean(v)) for g, v in buckets.items()}
    diff = np.array(buckets[n - 1]) - np.array(buckets[0])
    t = float(np.nanmean(diff) / (np.nanstd(diff, ddof=1) / np.sqrt(len(diff))))
    return means, t


def orthogonal_ic(fp: pd.DataFrame, px: pd.DataFrame, fwd: pd.DataFrame, ok: pd.DataFrame,
                  name: str, dates: pd.DatetimeIndex, px_cols: list[str],
                  min_n: int = 30) -> tuple[np.ndarray, float]:
    """把因子对价格因子截面正交化，重算 IC。

    共同暴露检验：若残差 IC 掉到 0，说明这个因子只是价格因子的换皮。
    返回 (逐期残差 IC, 因子被价格因子解释的平均 R²)。
    """
    ics, r2s = [], []
    for d in dates:
        if d not in fwd.index or d not in ok.index:
            continue
        r = fwd.loc[d].dropna()
        sub = fp.loc[fp["date"] == d, ["symbol", name]].merge(
            px.loc[px["date"] == d, ["symbol"] + px_cols], on="symbol", how="inner")
        sub = sub.dropna().reset_index(drop=True)
        if len(sub) < min_n:
            continue
        keep = ok.loc[d].reindex(sub["symbol"]).fillna(False).astype(bool).to_numpy()
        sub = sub[keep].reset_index(drop=True)
        sub = sub[sub["symbol"].isin(r.index)].reset_index(drop=True)
        if len(sub) < min_n:
            continue
        y = sub[name].to_numpy(dtype=float)
        mat = np.column_stack([np.ones(len(sub))]
                              + [sub[c].to_numpy(dtype=float) for c in px_cols])
        beta, *_ = np.linalg.lstsq(mat, y, rcond=None)
        resid = y - mat @ beta
        r2s.append(1.0 - resid.var() / y.var())
        ra = pd.Series(resid).rank().to_numpy()
        rb = pd.Series(r.loc[sub["symbol"].to_numpy()].to_numpy()).rank().to_numpy()
        ics.append(float(np.corrcoef(ra, rb)[0, 1]))
    return np.array(ics), float(np.mean(r2s))


CONTROL_SETS: dict[str, list[str]] = {
    "无控制": [],
    "仅规模": ["log_mv"],
    "仅流动性": ["turnover_lv"],
    "规模+换手(2个)": ["log_mv", "turnover_lv"],
    "规模+流动性": ["log_mv", "log_amt", "turnover_lv", "illiq20"],
    "动量+波动": ["ret1", "mom5", "mom20", "mom60", "vol20", "vol60", "amp"],
    "全部11个": ["ret1", "mom5", "mom20", "mom60", "vol20", "vol60",
                  "log_mv", "log_amt", "turnover_lv", "amp", "illiq20"],
}

PX_COLS = [f"r_{c}" for c in ("ret1", "mom5", "mom20", "mom60", "vol20", "vol60",
                             "log_mv", "log_amt", "turnover_lv", "amp", "illiq20")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(DATA_DIR / "panel_2015_2026.parquet"))
    ap.add_argument("--split", default="2020-12-31")
    ap.add_argument("--surrogate", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    panel_path = pathlib.Path(args.panel)
    closes = factors.load_closes(panel_path)
    me = factors.month_end_dates(closes.index)
    print(f"[mfac] 月末信号日 {len(me)} 个：{me[0].date()} ~ {me[-1].date()}")

    fp = build_factor_panel(me, panel_path)

    last = closes.reindex(me)
    fwd = last.pct_change().shift(-1)          # 下期收益
    lim = pd.Series([price_limit_pct(s) for s in closes.columns], index=closes.columns)
    ok = last.pct_change().abs().lt(lim - 0.005)   # 信号日涨跌停剔除

    split = pd.Timestamp(args.split)
    n_tr = int((me <= split).sum())
    print(f"[mfac] 训练 {me[0].date()}~{me[n_tr - 1].date()} ({n_tr} 期)"
          f" | 测试 {me[n_tr].date()}~{me[-1].date()} ({len(me) - n_tr} 期)")
    print(f"[mfac] 检验 {len(FACTORS)} 个因子，Bonferroni 单次阈值 p < {args.alpha / len(FACTORS):.4f}\n")

    rng = np.random.default_rng(20261011)
    rows, curves = [], {}
    for name in FACTORS:
        fac = fp.pivot(index="date", columns="symbol", values=name)
        blk = prep_blocks(fac, fwd, ok)
        blk_tr = [b for b in blk if b[0] <= split]
        blk_te = [b for b in blk if b[0] > split]
        ic_tr, ic_te = ic_series(blk_tr), ic_series(blk_te)
        null = np.array([ic_series(blk_te, rng).mean() for _ in range(args.surrogate)])
        obs = float(ic_te.mean())
        p_perm = float((np.abs(null) >= abs(obs)).mean())
        p_bon = min(1.0, p_perm * len(FACTORS))
        sign_ok = bool(np.sign(obs) == np.sign(ic_tr.mean()))
        rows.append({
            "factor": name, "n_train": len(blk_tr), "n_test": len(blk_te),
            "ic_train": ic_tr.mean(), "icir_train": ic_tr.mean() / ic_tr.std(),
            "ic_test": obs, "icir_test": obs / ic_te.std(), "sign_ok": sign_ok,
            "null_abs_mean": float(np.abs(null).mean()),
            "null_p95": float(np.quantile(np.abs(null), 0.95)),
            "p_perm": p_perm, "p_bonferroni": p_bon,
            "pass": bool(p_bon < args.alpha and sign_ok),
        })
        curves[name] = ic_te
    res = pd.DataFrame(rows)

    print(f"{'因子':<16s} {'IC训练':>9s} {'ICIR':>7s} {'IC测试':>9s} {'ICIR':>7s}"
          f" {'符号':>4s} {'噪声底':>9s} {'p':>7s} {'Bonf':>7s} {'过检':>4s}")
    for _, r in res.iterrows():
        print(f"{r['factor']:<16s} {r['ic_train']:>+9.4f} {r['icir_train']:>+7.2f}"
              f" {r['ic_test']:>+9.4f} {r['icir_test']:>+7.2f}"
              f" {'一致' if r['sign_ok'] else '相反':>4s} {r['null_abs_mean']:>9.4f}"
              f" {r['p_perm']:>7.3f} {r['p_bonferroni']:>7.3f} {'是' if r['pass'] else '否':>4s}")

    passed = res[res["pass"]]["factor"].tolist()
    print(f"\n[mfac] 过检因子: {passed if passed else '无'}")

    # 共同暴露检验：两融因子会不会只是价格因子的换皮
    px = capacity.build_features(panel_path, 25.0, dates=me)
    px_cols = PX_COLS
    px = px[["date", "symbol"] + px_cols]
    print("\n-- 共同暴露检验：对 11 个价格因子截面正交化后的 IC（测试期）--")
    print(f"{'因子':<16s} {'原 IC':>9s} {'残差 IC':>9s} {'残差 ICIR':>10s}"
          f" {'被价格解释':>10s} {'结论':>16s}")
    orth_rows = []
    for name in FACTORS:
        raw_te = float(res.loc[res["factor"] == name, "ic_test"].iloc[0])
        ic_r, r2 = orthogonal_ic(fp, px, fwd, ok, name, me[me > split], px_cols)
        icir_r = ic_r.mean() / ic_r.std()
        verdict = ("换皮 —— 残差 IC 归零" if abs(ic_r.mean()) < 0.005
                   else "残差仍显著" if abs(ic_r.mean()) > 0.02
                   else "部分独立")
        print(f"{name:<16s} {raw_te:>+9.4f} {ic_r.mean():>+9.4f} {icir_r:>+10.2f}"
              f" {r2:>10.1%} {verdict:>16s}")
        orth_rows.append({"factor": name, "ic_raw": raw_te, "ic_resid": ic_r.mean(),
                          "icir_resid": icir_r, "r2_vs_price": r2})
    orth = pd.DataFrame(orth_rows)
    orth.to_csv(OUT_DIR / "margin_factor_orth.csv", index=False)

    # 拆解：哪一类控制变量吃掉了信号（顺便检验是否过控制）
    print("\n-- 拆解：不同控制集下的残差 IC（测试期）--")
    print(f"{'因子':<16s}" + "".join(f"{k:>12s}" for k in CONTROL_SETS))
    diag_rows = []
    for name in FACTORS:
        vals = {}
        for label, cols in CONTROL_SETS.items():
            ic_r, _ = orthogonal_ic(fp, px, fwd, ok, name, me[me > split],
                                    [f"r_{c}" for c in cols])
            vals[label] = float(ic_r.mean())
            vals[f"_icir_{label}"] = float(ic_r.mean() / ic_r.std())
        raw = vals["无控制"]
        cells = "".join(f"{vals[k]:>+9.4f}{'!' if np.sign(vals[k]) != np.sign(raw) else ' '}"
                        f"{'':>2s}" for k in CONTROL_SETS)
        print(f"{name:<16s}{cells}")
        diag_rows.append({"factor": name, **vals})
    pd.DataFrame(diag_rows).to_csv(OUT_DIR / "margin_factor_control_diag.csv", index=False)
    print("  ! = 相对「无控制」符号翻转；完全翻转为 0 说明该控制集就是原始 IC 的来源")
    print("  「仅规模」/「仅流动性」单变量就能吃掉 -> IC 主要来自那一维，不是新信息")
    print("  单变量吃不掉但「全部11个」吃掉 -> 可能过控制，需看更少控制变量的稳健性")

    print("\n-- 分层收益（测试期，5 层等权，未扣成本）--")
    for name in FACTORS:
        fac = fp.pivot(index="date", columns="symbol", values=name)
        means, t = layered_returns(fac, fwd, ok, me[me > split])
        spread = means[4] - means[0]
        print(f"  {name:<16s} G1 {means[0]:+.3%}  G5 {means[4]:+.3%}"
              f"  多空 {spread:+.3%}/月  t={t:+.2f}   {'<- 过检' if name in passed else ''}")

    OUT_DIR.mkdir(exist_ok=True)
    res.to_csv(OUT_DIR / "margin_factor_ic.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT_DIR / "margin_factor_ic_series.csv")
    print(f"\n[mfac] 明细 -> {OUT_DIR}/margin_factor_ic*.csv")
    print("[mfac] 失效风险：两融标的名单会变（增删标的），非两融股票历史上没有数据，"
          "该因子天然只覆盖大盘中盘；名单变动后结论可能失效。")


if __name__ == "__main__":
    main()
