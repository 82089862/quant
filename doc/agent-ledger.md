# agent 意图台账（全域）

> 建立 2026-10-14。**全域一份**，不分域；域用 `domain` 字段区分。
> 事实源 = `doc/agent-ledger.db`（SQLite，本机，**不进 git**）；读写工具 = `ledger.py`（仓库根）。
> **记的不是用户原话，是 agent 对用户消息的理解。**

## 1. 记什么（两类）

| 类型 | 什么时候出现 | 记什么 |
|---|---|---|
| `目标` | 用户发起一项工作 | agent 理解的**目标内容 + 边界**（不记执行方案） |
| `断点` | 工作卡住，要用户拍板 | 断点 + 选项 + 用户的选择 + agent 理解的**执行线路** |

粒度：以**条目**为单位。一条用户消息可以产生 0 到 N 条；条目上带用户**消息时刻** `ts_msg`。

判定判据（2026-10-14 定，回溯补录时生效）：

- `目标` 只记**用户发起、有产物**的作业；信息询问不记（「这个目录下是什么」「你是用什么逻辑在跑」），追问细节也不记。
- `断点` 必须**我先给出明确选项集、用户明确拍了**才记；我给了选项但用户没答（改题、不回）也记一条，`chosen` 写「未选 + 实际走向」。
- 一条用户消息可以同时产出一条 `目标` 和一条 `断点`（用户改题：既发起新任务，又把前面的选项挂起）。

## 2. 什么时候写

- `目标`：理解完目标、准备动手前写。
- `断点`：**用户拍板之后写**（不做「先记待定、事后回填」）。
- `断点` 的事后实况追加进 `outcome`，状态转 `已回填`。`目标` 条不回填。

## 3. 字段

| 键 | 必填 | 说明 |
|---|---|---|
| `id` | 自动 | 主键，对外显示成 `L12` |
| `ts_msg` | 是 | 用户发消息的时刻（ISO 带时区）；不传则取落盘时刻 |
| `ts_record` | 自动 | agent 落盘时刻 |
| `type` | 是 | `目标` / `断点` |
| `domain` | 是 | `stocks` / `power` / `global` |
| `title` | 是 | 一句话标题 |
| `understood` | 是 | agent 的理解：目标条写目标+边界；断点条写上下文+执行线路 |
| `options` | 断点 | A/B/C，每个带代价与后果 |
| `chosen` | 断点 | 用户选的 |
| `rationale` | 否 | agent 理解的「用户为什么这么选」 |
| `line` | 断点 | 执行线路，一句话方向 |
| `impact` | 否 | 可逆 / 不可逆 + 影响面 |
| `code` | 断点 | D 码，见 §4 |
| `outcome` | 否 | 事后实况（回填） |
| `status` | 自动 | `已定` / `已回填` / `已更正`（回溯补录的条目落 `已回填`） |
| `corrected_by` | 否 | 更正条目的 id |
| `src` | 否 | 来源锚点 `<会话 id 前 13 位>#<消息序号>`，供去重与追溯 |

约定：每个值**单行**，要换行用 `<br>`（与 `stocks/假设登记表.md` 同约定）。

## 4. 决策类型码（只给断点用）

| 码 | 决策类型 | 什么时候用 |
|---|---|---|
| D1 | 口径分歧 | 同一事实两套算法或口径，必须选一套 |
| D2 | 数据源取舍 | 接哪个源，或降级、砍掉哪个源 |
| D3 | 资源投入 | 要不要继续投、投多少、做到哪一步停手 |
| D4 | 优先级 | 多条线先做哪条、排队顺序 |
| D5 | 停手 | 死因码同码即停，或不再加防线 |
| D6 | 不可逆动作授权 | 删库、强推、动生产数据前的显式授权 |
| D7 | 实现形态选型 | 存储、工具、接口长什么样 |
| D8 | 语义与契约 | 记什么、什么时候记、字段怎么定义 |

码表**只增不改**；删码不删行，把说明标成「(已废弃)」。命令 `ledger.py codes` 会打印同一张表。

## 5. agent 理解错了怎么办

**追加**一条更正条目，再把原条标成已更正：

```bash
uv run ledger.py add --type 断点 --domain stocks --title "更正 L7：口径不是 X 而是 Y" --understood "..."
uv run ledger.py update L7 --status 已更正 --corrected-by 12
```

原条保留不删，读的时候看到 `已更正` 就跳去看 `corrected_by` 指的那条。

## 6. 命令

```bash
uv run ledger.py add --type 断点 --domain stocks --code D1 \
    --title "..." --understood "..." --options "A / B" --chosen "A" --line "..." --impact "可逆"
uv run ledger.py list [--type 断点] [--domain stocks] [--code D1] [--since 2026-10-01]
uv run ledger.py show L12
uv run ledger.py update L12 --outcome "..." --status 已回填
uv run ledger.py codes
uv run ledger.py export --csv doc/agent-ledger.csv
uv run ledger.py extract --session 01a12402 --compact   # 从会话库抽历史消息供判读
uv run ledger.py import --jsonl entries.jsonl          # 批量补录（默认 status=已回填）
uv run ledger.py selftest
```

长文本用 `@stdin` 从标准输入读：`--understood @stdin`。

## 7. 历史回溯补录

历史会话库（`~/.pi/agent/pi-hermes-memory/sessions.db`）只有 user 原话 + assistant 回复，**没有 agent 理解字段**。所以回填不是导数据，是**重新判读**。

流程：

1. `ledger.py extract --root D:/quant --min-user 5` → 每个会话一份 jsonl（消息走 stdout，摘要走 stderr）
2. `--compact` 换成便于判读的 md：user 全留 + 紧邻其前的 assistant（选项集都在这条里）
3. agent 逐会话判读 → 写 jsonl（键名同 `add` 参数）
4. `ledger.py import --jsonl entries.jsonl`
5. 抽查：`ledger.py list --status 已回填`

规则（2026-10-14 拍板，用户「都按默认走」）：

- `understood` 前缀标 `[回溯]`；理解以**当时的 assistant 回复**为准
- `src` 指向会话库原始消息；本轮会话自己不入回填（已由 L1-L5 覆盖）
- 回填条目 `status=已回填`，`ts_msg` 用**用户消息的真实时刻**（不是补录时刻）

## 8. 边界

- `doc/agent-ledger.db` 被 `.gitignore` 排除：二进制文件 + 本仓有并发写手，进 git 必冲突。
- 本台账记的是 **agent 工作过程**，不是研究结论。研究结论仍归 `假设登记表.md`、协议、死因码表。
- 不进 MySQL 镜像（与 `hypothesis` 不同，没有跨会话的机械消费方）。
