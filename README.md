# DataAgent

**不是让 LLM 写 SQL，而是让它在写错的时候知道自己是错的。**

一个带护栏的自然语言查数 Agent。数据用 Olist 巴西电商库（9 张表 / 99,441 单 / 198 MB SQLite），
通过 function calling 让模型自己决定查什么、遇到错误怎么修，最后用一个可审计的 trace 记录全过程。

## 为什么做这个

第一次跑通就有答案，而且是错的：

| 轮次 | 模型给出的答案 | 实际值 | 错在哪 |
|---|---|---|---|
| 1 | 取消率 **6.25%** | 0.63% | 拿到正确的 625 和 99441 之后**心算相除错了一个数量级** |
| 2（加了数字溯源后） | 取消率 **62500%** | 0.63% | 改写 SQL 时 `count(*)` 漏了 `FROM`，SQLite 里恒等于 1，分母变成 1 |
| 3（加了聚合自检后） | "0.63%，**即 63%**" | 0.63% | 单位自相矛盾 |

三次都不是 SQL 语法错，SQL 全都能跑通、都有数据返回。这就是 LLM 最危险的失败模式：
**不报错、很自信、结果错。** 所以这个项目的重心不在"能不能生成 SQL"，而在
"生成之后能不能发现自己是错的"。

## 五道防线

| 防线 | 位置 | 挡住什么 | 实测 |
|---|---|---|---|
| 1. AST 校验 | `sandbox.validate_sql` | 写操作、多语句、未知表、`load_extension` 类函数 | DELETE/UPDATE/PRAGMA 全部拒绝 |
| 2. 只读连接 | `mode=ro` + `PRAGMA query_only` | 校验被绕过时的兜底写入 | 单测覆盖 |
| 3. 执行期中断 | `set_progress_handler` | 无 where 的笛卡尔积（geolocation 100 万行） | 递归 CTE 0.3s 被中断，连接仍可用 |
| 4. 结果自检 | `tools.self_check` | `empty_result` / `suspicious_zero` / `aggregate_without_from` / `suspicious_ratio` | 抓住第 2 轮那个 `count(*)` 分母 |
| 5. 数字溯源 | `verify.unsourced_numbers` | 答案里出现但查询结果中没有出处的数字 | 抓住第 1 轮的心算，打回重算一次 |

关键取舍：**`ok=True` 而 `error_kind='empty_result'`**。
"SQL 能跑通"和"答案是对的"是两件事，混在一起就失去了诊断价值。

## 架构

```mermaid
flowchart LR
    Q[问题] --> A[Agent loop<br/>max_steps=8]
    A -->|function call| T[ToolBox]
    T --> L[list_tables]
    T --> D[describe_table]
    T --> S[sample_rows]
    T --> R[run_sql]
    R --> G{沙箱三道闸门}
    G -->|通过| DB[(SQLite<br/>mode=ro)]
    G -->|拦截| E[结构化 error_kind]
    DB --> C[结果自检]
    C --> A
    E --> A
    A -->|最终答案| V[数字溯源]
    V -->|有未溯源数字| A
    V -->|通过| F[答案 + trace]
    A -.每一步.-> TR[(trace.sqlite)]
```

trace 从第一行代码就落库（`runs` / `steps` 两张表，含 `error_kind`、token、耗时、成本），
因为下一个项目 AgentEval 要直接吃这张表做采样噪声与显著性分析。

## 快速开始

```powershell
uv venv .venv --python 3.13
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

copy .env.example .env      # 填 DATAAGENT_API_KEY 和 DATAAGENT_DB

# 不消耗 token 的两个命令
.\run.cmd schema
.\run.cmd sql "select order_status, count(*) n from orders group by 1 order by n desc limit 5"

# agent
.\run.cmd ask "被取消的订单占全部订单的比例是多少？" -v
```

`python evals/run_eval.py --gold-only` 会现场执行 12 条参考 SQL 生成金标准（不花钱、不调模型）。

## 当前结果

- 12 题评测集（easy 3 / medium 4 / hard 3 / ambiguous 2），参考 SQL 全部可执行
- 沙箱与自检：25 个单测全绿（`python -m pytest tests -q`）
- 单题端到端：7 步、25.9k token、¥0（glm-4-flash 免费额度）
- **尚未跑全量评测**：准确率、平均修复轮次、成本三项指标等补完 60 题后再填，见 Roadmap

## Decision Log

- **schema 全塞进上下文，不做向量检索**：9 张表的 DDL + 列注释 + 样例值只有 5.1 KB，
  检索反而会漏表。表数上到几十张、schema 过百 KB 时才值得上 RAG。
- **不引入 LangChain/LangGraph**：loop 只有一百多行，控制流自己握着才能定位
  "模型在哪一步走错"。框架对比实验留到后期用 LangGraph 复刻同一套评测。
- **数字溯源只重试一次**：第二次仍失败就如实标记 `unsourced` 并输出，
  而不是无限重试——无限重试会把失败藏起来，还会烧钱。
- **trace 用 SQLite 而不是接 Langfuse**：本地可查询、零依赖，
  后续需要可视化时再导出 OTel 格式。
- **prompt_version 参与缓存 key**：改提示词必须同步升版本号，
  否则只会读回旧缓存，实验数据全部作废（这个坑在之前的 LLM 标注项目里踩过）。

## 已知局限

1. `glm-4-flash` 心算与单位换算能力弱，三次失败案例都出在这一层；
   换 `glm-4-plus`/`deepseek-chat` 的对照还没跑。
2. 数字溯源只能验证"有没有出处"，验证不了"这条 SQL 语义对不对"——
   第 2 轮 62500% 有出处但仍是错的，靠的是防线 4 而不是防线 5。
3. 评测集的判对口径偏宽松（多行题覆盖 50% 即算对），需要人工抽检校准。
4. 没有并发、没有服务化、没有 CI。

## Roadmap

- [ ] 评测集扩到 60 题，四种难度各 15 题，人工抽检 20 题校准判对口径
- [ ] 三个 baseline 对照：一次性 prompt / few-shot / 当前 agent loop
- [ ] 补齐指标：执行准确率、平均修复轮次、p95 延迟、单题成本
- [ ] FastAPI 服务化 + Docker
- [ ] 打包成 MCP server，在 Cursor / Reasonix 里直接查数
- [ ] 失败模式分类报告（数据源就是 trace 表的 `error_kind`）
