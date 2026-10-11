import numpy as np, pandas as pd, pathlib, duckdb, warnings
import factors
warnings.filterwarnings("ignore")
WIN, MIN_AMT = 250, 1e8

closes = factors.load_closes(pathlib.Path("data/panel_2015_2026.parquet"))
amt = duckdb.sql("select date, symbol, amount from read_parquet('data/panel_2015_2026.parquet')").df()
amt = amt.pivot(index="date", columns="symbol", values="amount").sort_index().tail(WIN)
R = closes.pct_change().tail(WIN)
keep = R.notna().all() & (amt.mean() > MIN_AMT).reindex(R.columns).fillna(False)
R = R.loc[:, keep]
mkt = R.mean(axis=1).to_numpy()
X = R.to_numpy(float)
beta = (X.T @ mkt) / (mkt @ mkt)
Z = X - np.outer(mkt, beta)
Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
C = Z.T @ Z / len(Z)
np.fill_diagonal(C, 0.0)
sym = list(R.columns)
u = pd.read_parquet("data/universe.parquet").set_index("symbol")["code_name"].to_dict()
j0 = sym.index("sz002507")
print("涪陵榨菜 sz002507 的残差相关 Top-15（阈值 0.4 下它有 290 个邻居）:")
for j in np.argsort(-C[j0])[:15]:
    print(f"   {sym[j]:10s} {u.get(sym[j],'?'):10s} 相关 {C[j0,j]:.3f}")

print("\n阈值敏感性 —— 同一个指标，只改阈值，Top-10 名单:")
for thr in (0.3, 0.4, 0.5, 0.6):
    deg = (C > thr).sum(axis=1)
    names = [u.get(sym[j], '?') for j in np.argsort(-deg)[:10]]
    print(f"  >{thr}: 中位邻居 {int(np.median(deg)):4d} | {', '.join(names)}")
