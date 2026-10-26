# 实验：小模型 + 强护栏 vs 大模型 + 弱护栏

> 状态：**已搭好代码与方案，未下载模型**（2026-10-26）。
> 想跑的时候按下面第二节执行即可，不需要再改代码。

## 1. 为什么做这个实验

DeepSeek-R1 论文（Nature 645, 633–638, 2025）的核心发现是：**大规模强化学习能让模型
自己涌现出自我验证、反思、回溯这些行为**。但那是训练期的手段——要重训，成本极高。

litmus 走的是另一条路：**不重训模型，而是在推理期用外部护栏强制它自我验证**。
五道防线（AST 校验 / 只读连接 / 超时 / 结果自检 / 答案数字溯源）加上 `FORCE_REFLECT`，
本质上是把 R1 靠 RL 才长出来的行为，用代码从外面逼出来。

**这个实验要回答的问题是**：护栏能替代多少模型能力？

- 如果「7B + 全套护栏」能追上「大模型 + 无护栏」，说明**护栏是可迁移的能力放大器**；
- 如果追不上，说明**某些推理缺口不是靠外部检查能补的**——那也是有价值的结论。

两种结果都值得写进作品集，因为**它是一次有明确对照的消融实验**，而不是一次跑分。

## 2. 怎么跑

### 2.1 前置条件

| 项 | 要求 | 本机现状 |
|---|---|---|
| 推理运行时 | [Ollama](https://ollama.com/download) | ❌ 未安装 |
| 模型 | `deepseek-r1:7b`（约 4.7 GB） | ❌ 未下载 |
| 显存 | ≥ 6 GB 最好 | ⚠️ RTX 4050 Laptop 只有 4 GB，会 offload 到 CPU |
| 内存 | ≥ 16 GB | ✅ 31.6 GB |
| 磁盘 | ≥ 10 GB 空闲 | ✅ D 盘 303.8 GB |

### 2.2 安装与拉模型

```powershell
# 装 ollama（安装包默认装到 %LOCALAPPDATA%\Programs\Ollama）
# 下载页：https://ollama.com/download/windows

# 拉模型（4.7 GB，走官方 CDN，本机实测可达）
ollama pull deepseek-r1:7b

# 验证服务和模型都在
ollama list
```

### 2.3 ⚠️ 必须调大上下文窗口

**ollama 默认 `num_ctx=2048`，而 litmus 的 system prompt + 9 张表 schema 就接近 3000 token。**
不调的话模型根本看不到 schema，实验结果会全是噪声。

```powershell
# 方法一：环境变量（推荐，全局生效）。设置后需重启 ollama 服务
setx OLLAMA_NUM_CTX 16384

# 方法二：临时起一个调好参数的实例
$env:OLLAMA_CONTEXT_LENGTH = 16384
ollama serve
```

### 2.4 先跑连通性（3 题，几分钟）

```powershell
cd D:\Projects\agentlab\dataagent
$env:PYTHONPATH = "src"
$env:PYTHONIOENCODING = "utf-8"

.\.venv\Scripts\python.exe -m litmus.cli ask "一共有多少条订单记录？" `
    --model deepseek-r1:7b --prompt v0.4.0 -v --no-cache
```

看到 `--- 状态: answer` 且数字正确，说明工具调用链路通了（这是最大的未知数，见第 4 节）。

### 2.5 跑评测

```powershell
# 先 15 题试水，测出每题平均耗时
.\.venv\Scripts\python.exe evals\run_eval.py `
    --model deepseek-r1:7b --prompt v0.4.0 `
    --limit 15 --tag local-r1-7b-lim15 --work-dir .litmus-local

# 确认没问题再跑全量
.\.venv\Scripts\python.exe evals\run_eval.py `
    --model deepseek-r1:7b --prompt v0.4.0 `
    --tag local-r1-7b --work-dir .litmus-local
```

结果落在 `evals/results/run-<时间戳>-<tag>.json`，进度同时逐题写进同名 `.jsonl`，
中途 Ctrl+C 也不会丢已完成的部分。

## 3. 预期耗时

按 RTX 4050（4 GB，7B Q4 需要 offload）估算，**实际以第一步的连通性测试为准**：

| 场景 | 单题耗时（估） | 60 题总耗时（估） |
|---|---|---|
| 7B，大部分层在 GPU | 20–40 s | 20–40 分钟 |
| 7B，主要跑在 CPU | 60–150 s | **1–2.5 小时** |
| 1.5B，全在 GPU | 5–15 s | 5–15 分钟 |

API 版（`glm-4-flash`）实测 60 题约 20 分钟，可作基线参照。

## 4. ⚠️ 四个已知风险

### 4.1 工具调用（function calling）可能不被支持——**最大的风险**

litmus 完全依赖 function calling（`run_sql` / `describe_table` …）。
`deepseek-r1` 蒸馏自 Qwen2.5，而 Qwen2.5 本身支持工具调用，但**蒸馏后该能力是否保留是未知的**，
ollama 的 R1 版本也**没有在模型卡片里承诺 tool calling**。

**先跑 2.4 的连通性测试**。如果模型只输出文本、不产生 `tool_calls`，这个实验就得换模型：

| 备选 | 说明 |
|---|---|
| `qwen2.5:7b` | 原生支持 function calling，最稳的退路（已在 `config.py` 的 local 白名单里） |
| `qwen2.5-coder:7b` | 写 SQL 更强，工具调用同样支持 |
| `llama3.1:8b` | Meta 官方支持 tool use |

换模型不用改代码——`config.py` 里已经把 `qwen2.5:7b` 注册进 local provider 了。

### 4.2 思考块会污染答案溯源

R1 蒸馏版会输出 `<think>…</think>` 包起来的草稿，里面全是中间算式（"625/99441 ≈ 6.25%"）。
如果不剥掉，`verify.py` 会去校验草稿里的数字，把本来正确的答案判成 `unverified`。

**已在 `llm.py:_clean_content()` 里处理**（支持 `think`/`thinking`/`reasoning`、大小写、
闭合与未闭合两种情形），并有 20 个测试守着（`tests/test_local_model.py`）。

### 4.3 上下文窗口

见 2.3。**不设 `OLLAMA_NUM_CTX` 这个实验一定是废的**，因为模型看不到表结构。

### 4.4 步数预算会被二次挤压

v0.4.0 的 `plan` 平均多吃 1 步，`max_steps=8` 下实际执行只有 5–6 步。
小模型本来就更容易卡住，建议这次同时跑一组 `--max-steps 10` 做对照：

```powershell
.\.venv\Scripts\python.exe evals\run_eval.py `
    --model deepseek-r1:7b --prompt v0.4.0 --max-steps 10 `
    --tag local-r1-7b-s10 --work-dir .litmus-local
```

## 5. 怎么读结果

现有基线（`glm-4-flash`，同 60 题，同库）：

| prompt | 总分 | easy | medium | hard | ambiguous | avg_steps |
|---|---|---|---|---|---|---|
| v0.2.0 | 0.483 | 0.93 | 0.47 | 0.33 | 0.20 | 5.20 |
| v0.3.1 | 0.500 | 1.00 | 0.47 | 0.40 | 0.13 | 4.42 |
| v0.4.0 | 0.517 | 0.93 | 0.80 | 0.27 | 0.07 | 5.42 |

跑完本地版本后，按这几个维度对比，别只看总分：

1. **难度分布**：如果小模型在 hard 上崩得更狠，说明推理缺口靠护栏补不上；
2. **`error_kinds` 频次**：`repeated_call` / `force_reflect` 的次数直接反映"护栏有多忙"；
3. **`unsourced` 数量**：小模型是否更多次交付无法溯源的数字（说明护栏在起作用）；
4. **未收敛题数**（`steps >= max_steps`）：区分"答错"和"没跑完"两类失败。

**注意**：单次 60 题的 ±1~2 题在噪声范围内，别过度解读总分微差。

## 6. 如果实验做成了，结论可以这么写

> 在 Olist 电商库的 60 题评测集上，7B 级开源模型配合 litmus 的五道护栏达到 X 分
> （对照：大模型基线 0.517）。护栏把 A% 的错误阻挡在交付之前，其中 B% 被模型自行修复。
> 这说明**结构化护栏可以部分替代模型规模**，但 hard 层仍有 C 点的差距，无法靠外部检查弥补。

（A/B/C 用实际跑出来的数字填。三个数字都来自 trace，可复核。）

## 7. 相关文件

| 文件 | 作用 |
|---|---|
| `src/litmus/config.py` | `PROVIDERS["local"]`：ollama 端点与模型白名单 |
| `src/litmus/llm.py` | `_is_loopback()`：本地端点免 key；`_clean_content()`：剥思考块 |
| `tests/test_local_model.py` | 20 个用例守上述逻辑（不需要真实模型即可跑） |
| `evals/run_eval.py` | `--model` / `--max-steps` / `--tag` / `--work-dir` |
