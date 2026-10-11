# quant

A 股日频回测最小系统 + 电力交易研究骨架。命令行工具，无 GUI、无回测框架依赖。

**按域分目录**：`stocks/`（股票）、`power/`（电力交易）。两域共用 `doc/研究方法与验证协议.md`（怎么判）与 `doc/死因码表.md`（为什么失败）。

## 快速开始

```bash
uv sync
uv run stocks/universe.py                                            # 证券档案（一次，约 30s）
uv run stocks/fetch.py --start 2015-01-01                            # 全A行情（一次，约 70 分钟）
uv run stocks/check.py                                               # 数据层自检（4 项断言）
uv run stocks/test_backtest.py                                       # 引擎自检（5 项断言）
uv run stocks/backtest.py                                            # 示例：茅台买入持有
uv run stocks/factors.py --factor reversal --window 20 --groups 5    # 分层回测 + IC
```

`stocks/factors.py` 首次运行会自动合成面板（DuckDB，约 6 秒）。

中断后重跑 `stocks/fetch.py` 即可，已拉的分片会跳过；`stocks/fetch.py --retry` 只重试 `stocks/data/failed.txt` 里的。

导一份到本机 MySQL（可选旁路副本，见「MySQL 旁路副本」）：

```bash
uv run stocks/import_mysql.py ddl      # 建库 quant + 表 daily / universe
uv run stocks/import_mysql.py probe    # 先测写路径（1 万行，自动清理）
uv run stocks/import_mysql.py load     # 全量导入（约 5 分钟）
uv run stocks/import_mysql.py verify   # 行数 + 分区 + 抽样比对 parquet
```

密码走 `QUANT_MYSQL_PASSWORD` 环境变量或 `--password`，不写进任何文件。

## 目录

```
quant/
  doc/                               全域通用
    研究方法与验证协议.md             证伪型测试的共同逻辑（五环节 + 已实测失败案例总表）
    死因码表.md                       F1~F5 死因码，两域共用、跨域查重
    stocks/假设登记表.md              股票域假设记录块（ID 用 H）
    power/假设登记表.md               电力域假设记录块（ID 用 P）
  stocks/                            股票域
    universe.py  fetch.py  store.py  backtest.py  factors.py  check.py  test_backtest.py
    capacity.py  margin_probe.py  margin_factor.py  dominance.py  leadlag.py
    downturn.py  import_mysql.py
    data/
      universe.parquet               8992 条证券档案（含 1192 只已退市）
      daily/{symbol}.parquet         5223 个分片，1102 万行，后复权日线
      panel_2015_2026.parquet        合成面板 303MB，横截面查询走它
      failed.txt                     拉取失败清单
    out/                             净值曲线与 IC 的 CSV
  power/                             电力交易域（骨架，见 power/README.md）
    data/
    out/
```

`stocks/data/` 合计 853MB，不进 git。

## 文件（stocks/）

| 文件 | 职责 |
|---|---|
| `stocks/universe.py` | 拉 baostock 全量证券档案 -> `stocks/data/universe.parquet` |
| `stocks/fetch.py` | 拉新浪后复权日线 -> `stocks/data/daily/{symbol}.parquet`，原子写、断点续传 |
| `stocks/store.py` | 读数据的唯一入口。`load_daily` 单只 / `load_panel` 横截面 / `build_panel` 合成面板 |
| `stocks/backtest.py` | 日频撮合引擎。`run_single` 单标的 / `run_portfolio` 多标的组合 |
| `stocks/factors.py` | 单因子（动量/反转）+ 月度分层回测 + IC |
| `stocks/leadlag.py` | 时序滞后联动挖掘：锚点遍历全市场 + 多重检验校正 + surrogate 噪声底 + 样本内外分割 |
| `stocks/check.py` | 数据层断言（复权可复现 / 与因子表自洽 / 收益在涨跌停内 / 分片结构） |
| `stocks/test_backtest.py` | 引擎断言（对账 / 涨跌停 / 调仓 / 费用） |
| `stocks/import_mysql.py` | 把 `stocks/data/` 的 parquet 导入本机 MySQL（`ddl` / `probe` / `load` / `verify`） |
| `stocks/import_hypothesis.py` | 把 `doc/` 的假设登记表 + 死因码表镜像进 MySQL（`ddl` / `load` / `verify`），单向、重跑覆盖 |
| `stocks/capacity.py` | 信息容量盘点。8 个数值列 -> 有效维度 7.5，新想法投影算 R² 判重 |
| `stocks/margin_probe.py` | 新数据源信息含量实测。融资融券 PR 净增 +1.95 |
| `stocks/margin_factor.py` | 两融因子验证：IC + surrogate + 训练/测试 + 7 档正交性阶梯 |
| `stocks/dominance.py` | 控制变量子集穷举（2^11）+ 优势分析：排除"过控制"、量化共线分摊 |
| `stocks/downturn.py` | 尾部传染挖掘：锚点下跌日 -> 候选次日下跌 lift + 多重检验校正 + 样本外 |
| `doc/研究方法与验证协议.md` | 上述证伪型测试的**共同方法论**（跨域通用） |
| `doc/死因码表.md` | F1~F5 死因码，**两域共用**，失败后跨域查重 |
| `doc/stocks/假设登记表.md` | 股票域假设记录块（ID 用 `H`） |
| `doc/power/假设登记表.md` | 电力域假设记录块（ID 用 `P`） |

## 数据口径

- **价格一律是后复权价**，可直接算收益率、无除权跳变。已验证：不同起始日拉同一段逐点相同（差 0.0），且 `hfq/raw` 等于 `hfq-factor` 表的累积因子。
- `volume` / `amount` 是真实成交口径，算流动性约束用它。
- 数据源：行情用新浪（akshare `stock_zh_a_daily`），证券档案用 baostock。
- baostock 的 `query_hs300_stocks(date=)` 支持历史指数成分，本项目尚未使用。
- **东财接口在本机不可达**（`push2his.eastmoney.com` 的 `/api/` 路径被拦），故不用它。

## 引擎规则

`backtest.run_single(daily, target, symbol, ...)` —— 单标的

- T 日收盘出目标权重，**T+1 开盘价成交**。
- 开盘触及涨停不买、跌停不卖，记在 `equity.blocked` 列。
- 同一开盘时点先卖后买，保证当日买入当日不卖（T+1）。
- 费用双边收：佣金（含最低值）+ 卖出印花税 + 过户费。
- `target` 里 **NaN = 该日不调仓**；`hold_all()` 是买入持有，全填 1.0 是每日再平衡，两者不同。

`backtest.run_portfolio(panel, rebalance, ...)` —— 组合

- `rebalance` 的 index 是**执行日**（开盘成交那天），columns 是 symbol，值是目标权重；调用方负责把信号日 shift 到执行日（`factors.build_schedule` 做这件事）。
- 成交规则与 `run_single` 一致；`equity.blocked` 是当日没成交的**笔数**，这个数大了回测和实盘就会分家。

## 已知简化

- 不区分 ST（5% 涨跌幅）与新股上市首日（无涨跌幅）。历史 ST 状态无法从当前数据源还原。
- 停牌日在日线里直接缺失，持仓按上一收盘估值，不交易。
- 退市股（255 只）未拉，回测存在幸存者偏差。档案已留，后续可补。
- 北交所不在数据源覆盖内。
- `sh689009`（科创板 CDR）新浪无数据，留在 `stocks/data/failed.txt`。

## 回测陷阱（实测踩过的，都静默失败）

1. **`0 股 × NaN 价 = NaN`**。组合估值里停牌/未上市价是 NaN，直接 `(shares * prices).sum()` 会把总资产污染成 NaN，之后每期目标股数都算不出来，回测全程空仓、收益 0，**不报错**。已封在 `backtest.portfolio_value` 里。
2. **目标权重每天填 1.0 = 每日再平衡**，不是买入持有。只有 `NaN` 表示不调仓。第一次对账差 9185 元才暴露。
3. **整手约束会吃光小资金回测**。100 万分给 1000 只股票 = 每只 1000 元，按 100 股整手全部被 `floor` 成 0，11 年只成交 50 笔。分层研究阶段用 `lot_size=1`。
4. **后复权价常有几千上万**（茅台新浪 hfq 11218），资金不够时高价股同样被 `floor` 成 0 股。分层研究用 1 亿本金。
5. `pandas.Series.corr(method="spearman")` 需要 scipy，本项目不装，用 rank 后的 Pearson 等价实现。

## 因子结果（2015-01 ~ 2026-10，月度调仓，5 层等权，1 亿本金）

`uv run stocks/factors.py --factor momentum --window 20 --groups 5`（reversal 结果与之对称）：
20 日动量 / 反转，140 期：

| 组 | 动量累计 | 反转累计 |
|---|---|---|
| G1（因子最小） | +212.87% | -55.79% |
| G2 | +230.61% | +125.94% |
| G3 | +210.79% | +210.17% |
| G4 | +125.25% | +230.68% |
| G5（因子最大） | -55.79% | +212.88% |

IC 均值：动量 -0.0704 / 反转 +0.0704，ICIR 0.46。**A 股月频存在显著反转效应**，方向对称验证通过（reversal G5 = momentum G1，差 0.01%）。

**这些数字有水分，别当实盘预期**：等权天然偏小盘股，而反转在小盘股上最强；未扣冲击成本（未成交 2.6% 说明流动性已在挡路）；未做市值/流动性中性化；退市股缺失。分层也非严格单调（G4/G5 轻微倒挂）。

## 时序滞后联动（lead-lag）实测

`uv run stocks/leadlag.py --anchor sz300359` —— 锚点全通教育，候选 5223 只，滞后 1~5 日，共 26,115 次检验：

| 切片 | 名义显著(5%) | Bonferroni | FDR |
|---|---|---|---|
| 训练 2015-2020（1462 日） | 3,171（12.1%） | 7 | 28 |
| 测试 2021-2026（1395 日） | 1,667（6.4%） | **0** | **0** |

训练集 top-50 配对在测试集留存 **3/50 = 6.0%**，surrogate 噪声底 **2.9%**。测试集内独立做的 Bonferroni 与 FDR 显著数**都是 0**。

**结论：本口径下未发现可复现的先行-滞后关系。** 训练集最强配对（ccf 0.158，p=1.2e-9）在测试集归零（ccf 0.0019，p=0.945）。

方法约束，缺一条结论就不成立：

1. 用收益率不用价格水平（价格水平有趋势，CCF 会假显著）
2. 剔除市场等权指数（实测名义显著 12.0% → 7.0%，纯噪声底 5.1%，即原始口径下一半以上是共同因子）
3. 剔除同期相关（同期高相关的两只股票会因自相关伪装出滞后）
4. 多重检验校正（26,115 次检验，只报名义 5% 等于报 1,305 个假阳性）
5. surrogate 噪声底（候选序列循环平移：保留自相关、破坏真实跨股票关系）——实测纯噪声也能有 9 个通过 Bonferroni，说明正态近似阈值不可靠，唯一判据是「显著数是否超过噪声底」
6. 样本内外分割，只报留存率
7. 涨跌停过滤（实测剔除 1.52% 样本）——涨停买不到、信息延到次日，会伪装成 lead-lag，那是交易制度产物

输出 `stocks/out/leadlag_{anchor}.csv`：配对、滞后阶、训练/测试 ccf、p 值、是否同号、是否留存。只输出统计观测，不判定因果、不生成买卖信号。

## MySQL 旁路副本（可选）

回测链路不读 MySQL，这是给外部工具 / BI 用的一份镜像。

- 实例：本机 WSL Ubuntu-22.04 里的 MySQL 8.0.46，Windows 侧 `127.0.0.1:3306` 可直连。
- 库 `quant`，四张表 + 一个视图：

| 表 | 行数 | 结构 |
|---|---|---|
| `daily` | 11,022,483 | `panel_*.parquet` 的镜像。`PRIMARY KEY(symbol, date)` + `KEY(date)`，9 个数值列全 `DOUBLE`（与 parquet 同精度），`date` 是 `DATE` |
| `universe` | 8,992 | `universe.parquet` 的镜像。`PRIMARY KEY(code)`，`code_name` 是 utf8mb4 中文名 |
| `hypothesis` | 9 | `doc/stocks/假设登记表.md` 的镜像。`PRIMARY KEY(id)`，字段与登记表记录块一一对应；`death_codes` 是逗号串（多码如 `F4,F3`，`FIND_IN_SET` 可直接用） |
| `cause_code` | 6 | `doc/死因码表.md` 的镜像。`PRIMARY KEY(code)`，`status` = 在用 / 已废弃 |
| `v_cause_stat`（视图） | — | `cause_code` LEFT JOIN `hypothesis`，直接出死因分布 + 命中哪些假设号 |

- **`hypothesis` / `cause_code` 是单向只读镜像**：事实源永远是 `doc/` 那两个 md，导入工具 `stocks/import_hypothesis.py`（口令同样走 `QUANT_MYSQL_PASSWORD`）。`load` 先 `TRUNCATE` 再整表覆盖，**在库里手改的数据下次 load 会被冲掉**。

```sql
-- 死因分布（F3 占 5/8 = 62.5%）
SELECT * FROM quant.v_cause_stat ORDER BY n_failed DESC;
```

- **`daily` 按年分区**（`PARTITION BY RANGE COLUMNS(date)`，p2015..p2026，12 个）。按日期查已实测裁剪到单个分区；归档旧年份用 `DROP PARTITION` 而不是 DELETE 千万行。
- 导入走 DuckDB 的 `mysql` 扩展直连（`ATTACH ... (TYPE mysql)` + `INSERT INTO ... SELECT`），**没有中间文件**；实测约 3.2 万行/s，全量 4.5 分钟。按年分批、一年一个 INSERT（单语句即单事务），重跑会跳过已有年份。
- 整库约 2.7GB。

**维护点**：分区没做 `MAXVALUE` 兜底——将来出现 2027 年及以后的数据时 INSERT 会直接报 1526 停下，先加分区再导：

```sql
ALTER TABLE quant.daily ADD PARTITION (PARTITION p2027 VALUES LESS THAN ('2028-01-01'));
```

`verify` 核对：行数与 parquet 相等、分区数 12、3 只股票各抽 5 行逐值比对（实测最大差 0）。

## 进度

- ① 数据层 — 完成。5223 只 / 1102 万行 / 2015-01-05 起，4 项断言通过。
- ② 撮合引擎 — 完成。5 项断言通过（对账 / 涨跌停 / 调仓 / 费用）。
- ③ 因子研究 — 完成。动量与反转、分层回测、IC。
- ④ MySQL 旁路副本 — 完成。`quant.daily` 1102 万行 / `quant.universe` 8992 行，按年分区，抽样与 parquet 完全一致；`quant.hypothesis` 9 行 / `quant.cause_code` 6 行 + 视图 `v_cause_stat`，与 md 逐值一致。
- ⑤ 组合与实盘 — 未做。下一步按性价比排序：基准对比（全市场等权 / 沪深300）> 分年稳定性 > 市值中性化 > 冲击成本估算。实盘通道（QMT/Ptrade）更远。
- 旁支 A 时序 lead-lag — 已完成并**证伪**，见上节（训练集 7 个 Bonferroni 显著，测试集 0 个）。
- 旁支 B 邻居效应 — 已探并放弃，脚本已删。方法：月末用过去 60 日收益率算残差相关，每只股票取 top-20 邻居；A 与邻居的同步 IC +0.39 是共同暴露、不可交易；A 相对邻居的超额反转 IC -5.5%、IR -0.58（137 期），比绝对反转因子（IR -0.46）更稳，但未继续验证。
