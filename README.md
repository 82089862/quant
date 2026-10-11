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
| `stocks/alpha191.py` | GTJA Alpha191 因子库（自写实现）：147 个公式 + 自写向量化算子，月度 IC + 训练/测试 + 11 控制变量正交化 + Bonferroni/FDR |
| `stocks/index_fetch.py` | 拉宽基指数日线（新浪）-> `stocks/data/index/`。股票 panel 里没有指数，择时类策略要先跑它 |
| `stocks/timing.py` | 指数择时策略池：RSRS / QRS修正 / ICU均线 / LLT / 鳄鱼线 / 滚动夏普，含训练测试切分与循环平移 surrogate |
| `stocks/import_screen.py` | 把择时/因子筛选结果镜像进 MySQL（`ddl` / `load` / `verify`，`--table timing\|factor\|all`），单向、重跑覆盖 |
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

## Alpha191 因子库（自写实现，147/191）

公式集来自公开的国泰君安 Alpha191（短周期价量因子），参照实现是开源项目 [popbo/alphas](https://github.com/popbo/alphas)。**只借公式，不引框架**：算子（Tsrank / Decaylinear / Wma / Regbeta / Lowday / Highday 等）全部自写且**全向量化** —— 参考实现用 `rolling.apply` 会变成 1460 万次 python 回调，跑成小时级；这里用 shift 累加改写成 numpy 级运算，全量 147 个因子 1184s 跑完。

```bash
uv run stocks/alpha191.py                          # 全量，写 out/alpha191_ic.csv
uv run stocks/alpha191.py --only alpha006,alpha014 # 冒烟
```

### 排除的 44 个（`SKIP` 里逐条写了理由，跑完会输出在 CSV 的「跳过」行）

| 类 | 个数 | 原因 |
|---|---|---|
| 依赖 `vwap` | 36 | 我们的 `amount/volume` 是**真实成交价**、`close` 是**后复权价**，实测 sh600519 2026-10-09 差 8.85 倍（1268.34 vs 11218.61）。同日纯比值能约掉复权因子，但 `VWAP-CLOSE` 这类差值、`SUM(VWAP,10)` 这类跨日运算全部失真 |
| 依赖 `benchmark` | 3 | 没有指数面板（75/181/182） |
| 公式恒为 0 | 4 | 原集合里的占位（30/143/149/190） |
| 含前视 | 1 | α054 的 `STD(ABS(CLOSE-OPEN))` 无窗口 = 全样本标准差，用了未来数据 |

### 实测结果（2015-01 ~ 2026-10，142 期月频，训练 ≤2020-12-31 / 测试 >）

- 检验 147 个，Bonferroni 阈值 `p < 3.40e-04`：**过 Bonferroni 94 个，过 FDR 131 个**，训练/测试同号 143 个。
- 但那 94 个**绝大多数是换皮**。加做共同暴露正交化（对 dominance.py 那 11 个价格控制变量做截面回归取残差，只在测试段判读）：

| 判读 | 个数 | 含义 |
|---|---|---|
| F3 确凿换皮 | **87** | 有单个控制变量就能把它打回量级线下 |
| 被多变量联手吃掉 | 33 | 单控最弱仍在线下，全 11 控制后归零 |
| 部分独立 | 14 | 全 11 控制后残差 IC 仍超量级线 0.0141 |
| F5 本来就无信号 | 13 | 无控制 IC 本身就在量级线下 |

- **过 Bonferroni + 训练测试同号 + 全控制后仍超量级线的：13 个**

| 因子 | 无控制 IC | 全 11 控制后残差 IC | 被价格解释比例 R² |
|---|---|---|---|
| `alpha070`（成交额 6 日标准差） | -0.1202 | +0.0399 | 0.287 |
| `alpha081`（成交量 21 日 Sma） | -0.0691 | +0.0405 | 0.252 |
| `alpha042`（最高价 10 日波动 × 量价相关） | +0.0903 | +0.0359 | 0.340 |
| `alpha107`（开盘相对前日高低收） | -0.0376 | -0.0323 | 0.124 |
| `alpha144`（下跌日跌幅/成交额） | +0.0742 | -0.0290 | 0.487 |
| `alpha015`（隔夜跳空） | -0.0250 | +0.0273 | 0.141 |
| `alpha135`、`alpha088`、`alpha035`、`alpha078`、`alpha139`、`alpha164`、`alpha184` | — | 0.0143 ~ 0.0226 | 0.11 ~ 0.74 |

（`out/alpha191_ic.csv` 是全量 191 行，含每个因子的 IC/ICIR/t/训练测试/残差 IC/判读标签。）

### 三个必须一起看的点

1. **94 个「过 Bonferroni」里 87 个是 F3 换皮。** 这与本项目已有的结论一致（两融 5 个因子、邻居同步都死在同一处），也是为什么 IC 单独过关不能算数。
2. **残差 IC 的符号翻转是好几个因子的特征**（α070 从 -0.12 翻成 +0.04）。原 IC 是规模/流动性暴露，剔掉之后剩下的是方向相反的另一个东西 —— 不能当「原信号加强了」。
3. **正交化后能留下 0.014~0.040 的残差 IC，量级很小**，且只在测试段判读（训练段用于避免泄漏）。这 13 个还没过协议的后两关：**控制集 2^11 穷举**（dominance.py 那套）和 **surrogate 噪声底**。在跑完这两关之前，「13 个」只能叫候选。

### 口径与已知缺口

- 正交化的对账：`--only` 单跑时 `ic_nocontrol` 必须与 `factors.ic_series` 的测试段 IC **逐位相等**，不等就 raise 停下（这条断言把「控制集接错行」这类错误挡在入库前）。
- 正交化**不剔除涨跌停** = 信息口径，不是可交易口径（与 `dominance.py` 的差别写在这里，免得以后对不上数）。
- **幸存者偏差**：255 只退市股未拉，而这里 IC 最大的几个都是小盘/流动性类因子，真实值会比现在差。
- 参考实现在 `alpha028` / `alpha059` 与研报原文写法不一致（疑笔误），本轮**照拄参考实现、未“修正”**，避免把改错当成改进。

## 指数择时策略池（借外部思想，核心自写）

思路来自开源项目 [QuantsPlaybook](https://github.com/hugo2046/QuantsPlaybook) 复现的券商金工研报，**只借规则、不引它的框架**（它依赖 qlib / backtrader / jqdata，我们不装）。

前提：我们 panel 里**没有指数**（只有 5223 只股票），所以先 `uv run stocks/index_fetch.py` 把 5 个宽基指数的日线拉到 `stocks/data/index/`。

```bash
uv run stocks/index_fetch.py
uv run stocks/timing.py --symbols sh000300,sh000905,sh000016,sh000001,sz399006 --out stocks/out/timing_pool.csv
```

| 策略 | 来源研报 | 规则 |
|---|---|---|
| `rsrs` | 光大证券 RSRS | 滚动 18 日 high/low 回归 beta（相关系数×标准差比），600 日 zscore，±0.7 阈值 |
| `rsrs_r2` | 中金 QRS 修正版 | 同上，但 beta 乘 R² |
| `icu` | 中泰证券 ICU 均线 | Siegel 重复中位数稳健回归均线，收盘价穿越 |
| `icu2` | 同上 | 快慢双均线（20/120） |
| `llt` | 广发证券低延迟趋势线 | LLT 递推线，α=0.05 |
| `alligator` | 招商证券鳄鱼线 | SMA(13/8/5) 各延迟 8/5/3，多头排列新形成进场 |
| `tsharpe` | 滚动夏普 | 近 60 日夏普 > 0 持仓（**简化版**，不是 Whitelaw 回归口径） |

### 实测结果（2015-01 ~ 2026-10，训练 2015-2020 / 测试 2021-2026）

35 次检验（5 指数 × 7 策略），门槛 = Bonferroni 1 - 0.05/35 = **0.9986**，**通过 0 个**。

| 最好的一批 | 测试累计 | 测试夏普 | 测试分位（循环平移零分布） |
|---|---|---|---|
| `sh000905` rsrs_r2 | +49.97% | 0.52 | 0.947 |
| `sz399006` rsrs | +58.53% | 0.47 | 0.923 |
| `sh000300` alligator | +6.43% | 0.15 | 0.913 |
| `sz399006` alligator | +35.16% | 0.37 | 0.810 |

三个必须同时看的点：

1. **没有一个过多重检验门槛。** 最好的 rsrs_r2 分位 0.947，离 0.9986 差得远。挑最好的那个 = 我们自己死因码表里的 F1。
2. **测试期两极分化不是策略有效，是指数性质不同。** 中小盘（中证500 / 创业板）测试段多个策略双正；大盘（沪深300 / 上证50 / 上证）测试段全灭，多数策略比随机还差（分位 0.23~0.63）。同期买入持有的沪深300 测试段也是 -18.04%。即：那张表测的是「2021-2026 中小盘有趋势」，不是「策略有预测力」。
3. **训练段几乎全正、测试段两极**是过拟合的典型形状（与 `downturn.py` / `leadlag.py` 的 F2 同源）。

`surrogate` 用的是 `leadlag.py` 那套循环平移：保住持仓的自相关与仓位占比，只破坏「哪天持仓」与「哪天上涨」的对应关系；300 次平移出零分布，看实际收益的分位。

**未做的（已知缺口）**：指数只做多、满仓/空仓，不做空不做杠杆；未扣指数无法直接交易的跟踪误差；择时是市场级信号，**不进个股截面因子链路**（协议 §11.1 的 S 层口径）。

## MySQL 旁路副本（可选）

回测链路不读 MySQL，这是给外部工具 / BI 用的一份镜像。

- 实例：本机 WSL Ubuntu-22.04 里的 MySQL 8.0.46，Windows 侧 `127.0.0.1:3306` 可直连。
- 库 `quant`，六张表 + 一个视图：

| 表 | 行数 | 结构 |
|---|---|---|
| `daily` | 11,022,483 | `panel_*.parquet` 的镜像。`PRIMARY KEY(symbol, date)` + `KEY(date)`，9 个数值列全 `DOUBLE`（与 parquet 同精度），`date` 是 `DATE` |
| `universe` | 8,992 | `universe.parquet` 的镜像。`PRIMARY KEY(code)`，`code_name` 是 utf8mb4 中文名 |
| `hypothesis` | 9 | `doc/stocks/假设登记表.md` 的镜像。`PRIMARY KEY(id)`，字段与登记表记录块一一对应；`death_codes` 是逗号串（多码如 `F4,F3`，`FIND_IN_SET` 可直接用） |
| `cause_code` | 6 | `doc/死因码表.md` 的镜像。`PRIMARY KEY(code)`，`status` = 在用 / 已废弃 |
| `timing_screen` | 40 | 择时策略池的筛选结论，一行 = 一个「指数 × 策略」。源是 `stocks/out/timing_pool_summary.csv`。`passed` 在导入时按 1-0.05/检验次数 现算 |
| `factor_screen` | 191 | Alpha191 因子库的筛选结论，一行 = 一个因子（已算 147 / 跳过 44）。源是 `stocks/out/alpha191_ic.csv`。含正交化对账列 `ic_nocontrol`（必须逐位等于 `ic_test`）与 `verdict` 判读 |
| `v_cause_stat`（视图） | — | `cause_code` LEFT JOIN `hypothesis`，直接出死因分布 + 命中哪些假设号 |

- **`hypothesis` / `cause_code` 是单向只读镜像**：事实源永远是 `doc/` 那两个 md，导入工具 `stocks/import_hypothesis.py`（口令同样走 `QUANT_MYSQL_PASSWORD`）。`load` 先 `TRUNCATE` 再整表覆盖，**在库里手改的数据下次 load 会被冲掉**。
- `timing_screen` 同理是单向镜像，源是 `stocks/out/timing_pool_summary.csv`；`factor_screen` 源是 `stocks/out/alpha191_ic.csv`。两者的导入工具都是 `stocks/import_screen.py`（`--table timing|factor|all`）。

```sql
-- 死因分布（F3 占 5/8 = 62.5%）
SELECT * FROM quant.v_cause_stat ORDER BY n_failed DESC;

-- Alpha191：过 Bonferroni + 训练测试同号 + 全控制后仍独立的 13 个候选
SELECT factor, ic_allctrl, r2_allctrl, t FROM quant.factor_screen
WHERE status='已算' AND bonferroni=1 AND sign_same=1 AND verdict='部分独立'
ORDER BY abs(ic_allctrl) DESC;

-- 择时池：测试分位最高的几个（passed 全为 0，一个都没过 Bonferroni 门槛）
SELECT symbol, strategy, test_cum, test_sharpe, test_pct, passed
FROM quant.timing_screen WHERE strategy <> '买入持有'
ORDER BY test_pct DESC LIMIT 5;
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
- ④ MySQL 旁路副本 — 完成。`quant.daily` 1102 万行 / `quant.universe` 8992 行，按年分区，抽样与 parquet 完全一致；`quant.hypothesis` 9 行 / `quant.cause_code` 6 行 + 视图 `v_cause_stat`，`quant.timing_screen` 40 行 / `quant.factor_screen` 191 行（已算 147），都与事实源逐值一致。
- ⑥ 外部策略池 — 择时部分完成：5 指数 × 7 策略 = 35 次检验，Bonferroni 门槛下**通过 0 个**。
- ⑦ Alpha191 因子库 — 147 个公式自写实现并跑完。裸 IC 过 Bonferroni 94 个，但正交化后判为「部分独立」的只有 **14 个**，其中同时过 Bonferroni 与训练测试同号的 **13 个候选**；剩余两关（控制集 2^11 穷举、surrogate 噪声底）未跑。
- ⑤ 组合与实盘 — 未做。下一步按性价比排序：基准对比（全市场等权 / 沪深300）> 分年稳定性 > 市值中性化 > 冲击成本估算。实盘通道（QMT/Ptrade）更远。
- 旁支 A 时序 lead-lag — 已完成并**证伪**，见上节（训练集 7 个 Bonferroni 显著，测试集 0 个）。
- 旁支 B 邻居效应 — 已探并放弃，脚本已删。方法：月末用过去 60 日收益率算残差相关，每只股票取 top-20 邻居；A 与邻居的同步 IC +0.39 是共同暴露、不可交易；A 相对邻居的超额反转 IC -5.5%、IR -0.58（137 期），比绝对反转因子（IR -0.46）更稳，但未继续验证。
