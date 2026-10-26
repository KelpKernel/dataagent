"""最小 agent 循环（单 agent + function calling）。

刻意不引入 LangChain/LangGraph：这个 loop 只有一百多行，
把控制流握在自己手里才能看清「模型在哪一步走错、为什么」。
框架对比实验留到 README 附录（用 LangGraph 复刻一版跑同一套评测）。

循环里有两个非标准但关键的设计：
1. 答案数字溯源（verify）：模型拿到正确 SQL 结果后仍会心算错
   （实测把 625/99441 算成 6.25%），所以答案里的数字必须在查询结果里找到出处。
2. 强制反思（force_reflect，v0.3.0 起）：trace 统计显示护栏命中里 61.6% 是
   repeated_call——模型卡在重复调用上，而不是 SQL 写错。只提示"换写法"无效，
   所以连续重复两次后直接注入一条强制指令，逼它先说明根因再换路径。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from litmus.config import Config
from litmus.llm import LLMClient, Usage
from litmus.prompts import FORCE_REFLECT, TURN_REMINDER, system_prompt
from litmus.sandbox import Sandbox
from litmus.tools import TOOLS, ToolOutcome, ToolBox, default_schema_text
from litmus.trace import TraceStore
from litmus.verify import collect_from_values, unsourced_numbers

_RETRY_TEMPLATE = """你上一条回答里有这些数字无法在查询结果中找到出处：{missing}。

请用 run_sql 把这些数字真正查出来（比率请让 SQL 自己算，用 round(100.0 * a / b, 2) 的形式），
然后基于查询结果重新给出结论。不要保留任何心算或估算的数字。"""


_MAX_PLANS = 2

# plan: 行被 loop 剥离后，用这句让它继续干活（v0.4.0 起）
_PLAN_ACK = "计划已记录。现在按计划执行，需要时调用工具。"

# 超过 _MAX_PLANS 后改成这句：实测 glm-4-flash 会每轮都想重出计划，
# 必须明确禁止，否则步数全耗在写计划上。
_PLAN_OVERFLOW = "你已经输出过计划了，不需要再输出 plan。现在直接调用工具执行。"

_FINAL_STATES = ("answer", "unverified", "failed", "plan")


def parse_final(text: str) -> tuple[str, str]:
    """解析最终消息的显式状态前缀，返回 (state, payload)。

    为什么要有这个东西（v0.4.0 引入）：
    v0.3.1 的最终答案是自由文本，机器分不出「查完了」「查了但不敢保证」「根本没查成」
    三种情况——它们都长成一段中文。参考 Claude Code 的做法（result: / needs input: /
    failed: 是唯一的完成信号，散文不算），把状态提到文本最前面，程序就能直接判定。

    没有前缀（v0.2.0~v0.3.1）时返回 ("", 原文)，调用方按旧行为走自由文本逻辑，
    所以老版本提示词仍可复现。

    容错：允许前后有 markdown 装饰（`**answer:**`）和中文冒号（`answer：`），
    实测模型偶尔会给前缀加粗。
    """
    stripped = text.strip()
    # 只剥掉开头可能存在的 markdown 装饰与空白，再用 startswith 判定。
    # 早先用 find() + 位置阈值，正文里提一句 "answer:" 就会被误判成状态。
    decorated = stripped.lstrip("*_`# \t")
    lowered = decorated.lower()
    for state in _FINAL_STATES:
        for sep in (":", "："):
            prefix = state + sep
            if lowered.startswith(prefix):
                payload = decorated[len(prefix) :].lstrip("*_` :：").strip()
                return state, payload
    return "", stripped


@dataclass(slots=True)
class AgentResult:
    question: str
    answer: str
    ok: bool
    steps: int
    # v0.4.0：answer / unverified / failed。空串表示用的是 v0.4.0 之前的提示词，
    # 状态由 ok + unsourced 推断（旧行为）。
    state: str = ""
    usage: Usage = field(default_factory=Usage)
    error_kinds: list[str] = field(default_factory=list)
    unsourced: list[str] = field(default_factory=list)
    interventions: int = 0
    run_id: str = ""

    @property
    def tokens(self) -> int:
        return self.usage.prompt_tokens + self.usage.completion_tokens


class LitmusAgent:
    def __init__(
        self,
        config: Config,
        *,
        sandbox: Sandbox | None = None,
        tracer: TraceStore | None = None,
        llm: LLMClient | None = None,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.verbose = verbose
        self.sandbox = sandbox or Sandbox(
            config.db_path,
            row_limit=config.row_limit,
            timeout_s=config.query_timeout_s,
        )
        self.tracer = tracer or TraceStore(config.trace_db)
        self.llm = llm or LLMClient(config)
        self.schema_text = default_schema_text(self.sandbox)
        self.toolbox = ToolBox(
            self.sandbox, schema_text=self.schema_text, row_limit=config.row_limit
        )
        self.rules = system_prompt(config.prompt_version)
        # v0.4.0 起在问题后追加一次格式提醒；旧版本没有这个键，取到空串
        self.reminder = TURN_REMINDER.get(config.prompt_version, "")

    def run(self, question: str) -> AgentResult:
        run_id = self.tracer.start(
            question, self.config.model, self.config.prompt_version
        )
        messages: list[dict] = [
            {
                "role": "system",
                "content": f"{self.rules}\n\n【数据库结构】\n{self.schema_text}",
            },
            {"role": "user", "content": question},
        ]
        # v0.4.0 起在问题后追加一次格式提醒：只写 system prompt 实测不够
        # （glm-4-flash 会忽略），靠近调用点再提醒一次。旧版本 reminder 为空，消息序列不变。
        if self.reminder:
            messages.append({"role": "user", "content": self.reminder})

        total = Usage()
        error_kinds: list[str] = []
        source_values: list[float] = []  # 所有查询结果里的数值，用于答案溯源
        answer = ""
        unsourced: list[str] = []
        verify_retried = False
        stuck = 0
        interventions = 0
        step = 0
        final_state = ""  # v0.4.0：answer / unverified / failed；空串表示按旧行为
        plans = 0  # 已剥离的 plan: 行数，上限 _MAX_PLANS

        for step in range(1, self.config.max_steps + 1):
            turn = self.llm.chat(messages, tools=TOOLS)
            total.add(turn.usage)
            self.tracer.add_step(
                kind="assistant",
                name="llm",
                result_preview=turn.content,
                ok=True,
                tokens_in=turn.usage.prompt_tokens,
                tokens_out=turn.usage.completion_tokens,
                cost=turn.usage.cost,
            )
            if self.verbose:
                tag = (
                    "缓存"
                    if turn.cached
                    else f"{turn.usage.prompt_tokens}+{turn.usage.completion_tokens} tok"
                )
                calls = ",".join(c.name for c in turn.tool_calls) or "-"
                print(f"  [{step}] llm ({tag}) tools={calls}")

            if not turn.wants_tools:
                state, payload = parse_final(turn.content)
                # plan: 只是计划，不算答案——记下来让它继续（最多两次，防死循环）
                if state == "plan" and plans < _MAX_PLANS:
                    plans += 1
                    # 实测模型会把计划和结论写在同一条消息里（"plan: ...\n\nanswer: ..."），
                    # 这里只取第一行当计划，否则半截答案会被当成计划记进 trace。
                    plan_line = payload.splitlines()[0].strip() if payload else ""
                    self.tracer.add_step(
                        kind="plan", name="plan", result_preview=plan_line, ok=True
                    )
                    if self.verbose:
                        print(f"      📋 plan: {plan_line[:88]}")
                    messages.append(_assistant_message(turn))
                    messages.append({"role": "user", "content": _PLAN_ACK})
                    continue
                # 超过 plan 上限：不再 ack，改成强提醒并继续（max_steps 兜底）。
                # 不能把它当普通文本，否则 "plan: ..." 会被当成最终答案、任务提前结束。
                if state == "plan":
                    error_kinds.append("plan_overflow")
                    messages.append(_assistant_message(turn))
                    messages.append({"role": "user", "content": _PLAN_OVERFLOW})
                    continue
                # failed: 模型明确说搞不定，不再溯源、不再重试
                if state == "failed":
                    answer, final_state = payload, "failed"
                    break
                # unverified: 模型自己声明有数字没出处，采纳但保留标记
                if state == "unverified":
                    answer, final_state = payload, "unverified"
                    break
                # answer: 或无前缀（v0.4.0 之前）→ 走溯源 gate
                answer = payload if state == "answer" else turn.content.strip()
                unsourced = unsourced_numbers(answer, source_values)
                # 数字溯源失败 → 打回重算，但只重试一次，避免死循环
                if unsourced and not verify_retried and step < self.config.max_steps:
                    verify_retried = True
                    error_kinds.append("unsourced_number")
                    self.tracer.add_step(
                        kind="verify",
                        name="answer_grounding",
                        result_preview="未溯源数字: " + "、".join(unsourced),
                        ok=False,
                        error_kind="unsourced_number",
                    )
                    if self.verbose:
                        print(f"      ↻ 数字未溯源 {unsourced}，要求重算")
                    messages.append(_assistant_message(turn))
                    messages.append(
                        {
                            "role": "user",
                            "content": _RETRY_TEMPLATE.format(
                                missing="、".join(unsourced)
                            ),
                        }
                    )
                    continue
                # 重试后仍有未溯源数字：答案照给，但状态降级为 unverified
                # 只有带 answer: 前缀的才写显式状态；无前缀（v0.4.0 之前的提示词）
                # 保持空串，这样「按 version 复现旧实验」时状态列不会被污染。
                if state == "answer":
                    final_state = "unverified" if unsourced else "answer"
                break

            messages.append(_assistant_message(turn))
            for call in turn.tool_calls:
                outcome: ToolOutcome = self.toolbox.call(call.name, call.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": outcome.text,
                    }
                )
                if outcome.error_kind:
                    error_kinds.append(outcome.error_kind)
                if outcome.result is not None and outcome.result.ok:
                    for row in outcome.result.rows:
                        source_values.extend(collect_from_values(list(row)))
                stuck = stuck + 1 if outcome.error_kind == "repeated_call" else 0
                self.tracer.add_step(
                    kind="tool",
                    name=call.name,
                    args=_safe_args(call.arguments),
                    sql=outcome.sql,
                    result_preview=outcome.text,
                    error_kind=outcome.error_kind,
                    ok=outcome.ok,
                )
                if self.verbose:
                    flag = f"[{outcome.error_kind}]" if outcome.error_kind else ""
                    head = outcome.text.splitlines()[0][:88] if outcome.text else ""
                    print(f"      → {call.name} {flag} {head}")

            # 连续两次重复 → 强制反思，而不是继续提示
            if self.config.force_reflect and stuck >= 2:
                interventions += 1
                error_kinds.append("force_reflect")
                messages.append(
                    {"role": "user", "content": FORCE_REFLECT.format(n=stuck)}
                )
                self.tracer.add_step(
                    kind="intervention",
                    name="force_reflect",
                    result_preview=f"连续重复 {stuck} 次，注入反思指令",
                    ok=True,
                    error_kind="force_reflect",
                )
                if self.verbose:
                    print(f"      ⚡ 强制反思（连续重复 {stuck} 次）")
                stuck = 0
        else:
            answer = (
                f"（已达最大步数 {self.config.max_steps}，未能收敛。"
                f"最后一步模型输出：{self._last_content(messages) or '空'}）"
            )

        ok = (
            bool(answer)
            and not answer.startswith("（已达最大步数")
            and final_state != "failed"
        )
        # v0.4.0 起 trace 的 status 用显式状态，便于按状态聚合统计
        self.tracer.finish(
            status=final_state or ("ok" if ok else "budget_exhausted"),
            answer=answer,
            tokens_in=total.prompt_tokens,
            tokens_out=total.completion_tokens,
            cost=total.cost,
        )
        return AgentResult(
            question=question,
            answer=answer,
            ok=ok,
            steps=step,
            state=final_state,
            usage=total,
            error_kinds=error_kinds,
            unsourced=unsourced,
            interventions=interventions,
            run_id=run_id,
        )

    def close(self) -> None:
        self.sandbox.close()
        self.tracer.close()

    @staticmethod
    def _last_content(messages: list[dict]) -> str:
        for message in reversed(messages):
            if message.get("role") == "assistant":
                return (message.get("content") or "").strip()
        return ""


def _assistant_message(turn) -> dict:
    """构造 assistant 消息。

    tool_calls 为空时必须**不带这个字段**：智谱容忍 `tool_calls: []`，
    但 DeepSeek 会直接 400（"empty array. Expected an array with minimum length 1"）。
    多 provider 兼容的第一课：别把宽松实现的容忍当成标准。
    """
    message: dict = {"role": "assistant", "content": turn.content or ""}
    if turn.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in turn.tool_calls
        ]
    return message


def _safe_args(raw: str):
    import json

    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return raw
