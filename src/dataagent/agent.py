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

from dataagent.config import Config
from dataagent.llm import LLMClient, Usage
from dataagent.prompts import FORCE_REFLECT, system_prompt
from dataagent.sandbox import Sandbox
from dataagent.tools import TOOLS, ToolOutcome, ToolBox, default_schema_text
from dataagent.trace import TraceStore
from dataagent.verify import collect_from_values, unsourced_numbers

_RETRY_TEMPLATE = """你上一条回答里有这些数字无法在查询结果中找到出处：{missing}。

请用 run_sql 把这些数字真正查出来（比率请让 SQL 自己算，用 round(100.0 * a / b, 2) 的形式），
然后基于查询结果重新给出结论。不要保留任何心算或估算的数字。"""


@dataclass(slots=True)
class AgentResult:
    question: str
    answer: str
    ok: bool
    steps: int
    usage: Usage = field(default_factory=Usage)
    error_kinds: list[str] = field(default_factory=list)
    unsourced: list[str] = field(default_factory=list)
    interventions: int = 0
    run_id: str = ""

    @property
    def tokens(self) -> int:
        return self.usage.prompt_tokens + self.usage.completion_tokens


class DataAgent:
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

        total = Usage()
        error_kinds: list[str] = []
        source_values: list[float] = []  # 所有查询结果里的数值，用于答案溯源
        answer = ""
        unsourced: list[str] = []
        verify_retried = False
        stuck = 0
        interventions = 0
        step = 0

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
                answer = turn.content.strip()
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

        ok = bool(answer) and not answer.startswith("（已达最大步数")
        self.tracer.finish(
            status="ok" if ok else "budget_exhausted",
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
    return {
        "role": "assistant",
        "content": turn.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in turn.tool_calls
        ],
    }


def _safe_args(raw: str):
    import json

    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return raw
