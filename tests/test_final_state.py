"""最终答案三态（v0.4.0）的回归测试。

为什么单独立一个文件守这层：v0.3.1 的最终答案是自由文本，机器分不出
「查完了」「查了但不敢保证」「根本没查成」三种情况——它们都长成一段中文，
统计和审计只能靠人读。v0.4.0 把状态提到文本最前面（answer: / unverified: /
failed:，参考 Claude Code 的 result: / needs input: / failed:），解析一旦出错，
trace 里的护栏统计会全部失真，所以这一层必须有测试钉住。

同时守一条兼容底线：v0.2.0~v0.3.1 没有前缀，行为必须保持原样（state 为空），
否则老版本提示词就复现不了了。
"""

from __future__ import annotations

import pytest

from litmus.agent import parse_final
from litmus.config import DEMO_DB, Config
from litmus.llm import AssistantTurn, Usage


class ScriptedLLM:
    """按脚本顺序返回 AssistantTurn 的假客户端。

    用假的而不是真调模型：这里要验证的是**控制流**（什么时候剥计划、
    什么时候打回、什么时候直接采纳），跟模型说什么无关。
    """

    def __init__(self, turns: list[AssistantTurn]) -> None:
        self._turns = list(turns)
        self.calls = 0

    def chat(self, messages, tools=None):  # noqa: ANN001 - 对齐 LLMClient 签名
        self.calls += 1
        if not self._turns:
            raise AssertionError("脚本已用尽：agent 比预期多跑了一步")
        return self._turns.pop(0)


def _turn(content: str = "") -> AssistantTurn:
    return AssistantTurn(content=content, tool_calls=[], usage=Usage())


def _run(tmp_path, turns: list[AssistantTurn], *, prompt_version: str = "v0.4.0"):
    from litmus.agent import LitmusAgent

    config = Config(
        db_path=DEMO_DB,
        work_dir=tmp_path,
        use_cache=False,
        prompt_version=prompt_version,
    )
    agent = LitmusAgent(config, llm=ScriptedLLM(turns))
    try:
        return agent.run("测试问题")
    finally:
        agent.close()


# ---------------------------------------------------------------- parse_final


@pytest.mark.parametrize(
    "text,expected",
    [
        ("answer: 取消率 0.63%", "answer"),
        ("**answer:** 加粗前缀也认", "answer"),
        ("answer：全角冒号也认", "answer"),
        ("unverified: 约 6%，但 6 没有出处", "unverified"),
        ("failed: 目标表不存在", "failed"),
        ("plan: 用 orders join customers", "plan"),
        ("没有前缀的自由文本", ""),
        ("", ""),
    ],
)
def test_parse_final_state(text: str, expected: str) -> None:
    state, _ = parse_final(text)
    assert state == expected


def test_parse_final_strips_payload() -> None:
    state, payload = parse_final("answer:   前后有空格   ")
    assert state == "answer"
    assert payload == "前后有空格"


def test_parse_final_ignores_midtext_prefix() -> None:
    """正文中间出现 "answer:" 不该被当成状态前缀（前缀必须在开头附近）。"""
    state, payload = parse_final("下面讨论 answer: 这个词本身")
    assert state == ""
    assert payload == "下面讨论 answer: 这个词本身"


# --------------------------------------------------------------- run() 三态


def test_plan_line_is_stripped_and_loop_continues(tmp_path) -> None:
    """plan: 不是答案——要被记录并让循环继续，且不能混进最终答案。"""
    result = _run(
        tmp_path,
        [
            _turn("plan: 查 orders 表，按 order_status 分组"),
            _turn("answer: 数据里没有取消订单。"),  # 无数字，不触发溯源
        ],
    )
    assert result.steps == 2, "第一轮的 plan 应当被剥离，循环继续到第二轮"
    assert result.state == "answer"
    assert result.ok is True
    assert result.answer == "数据里没有取消订单。"
    assert "plan:" not in result.answer


def test_failed_state_short_circuits(tmp_path) -> None:
    """failed: 直接采纳，不重试、不溯源。"""
    result = _run(tmp_path, [_turn("failed: 目标表不存在，无法查询。")])
    assert result.steps == 1
    assert result.state == "failed"
    assert result.ok is False
    assert result.answer == "目标表不存在，无法查询。"


def test_unverified_state_accepted_without_retry(tmp_path) -> None:
    """模型自己声明有数字没出处时，直接采纳并保留标记，不再打回。"""
    result = _run(tmp_path, [_turn("unverified: 大约 6%，但这个 6 没有出处。")])
    assert result.steps == 1, "unverified: 是最终态，不应触发 verify 重试"
    assert result.state == "unverified"
    assert result.ok is True


def test_legacy_free_text_keeps_old_behaviour(tmp_path) -> None:
    """兼容底线：没有前缀的旧版提示词必须按原逻辑走，state 保持为空。"""
    result = _run(tmp_path, [_turn("数据里没有取消订单。")], prompt_version="v0.3.1")
    assert result.state == ""
    assert result.ok is True
    assert result.answer == "数据里没有取消订单。"


def test_plan_and_answer_in_one_message(tmp_path) -> None:
    """实测坑：模型会把计划和结论写在同一轮（"plan: ...\\n\\nanswer: ..."）。

    这时 parse_final 只认第一个前缀（plan），如果整段都当计划记进 trace，
    那么写在同一轮的答案就被丢掉了，白白多跑一步。所以计划只取第一行。
    """
    result = _run(
        tmp_path,
        [
            _turn("plan: 查 orders 表，按 order_id 聚合\n\nanswer: 100 单"),
            _turn("answer: 数据里没有取消订单。"),
        ],
    )
    assert result.steps == 2, "第一轮只应被当成计划，不能当作结束"
    assert result.answer == "数据里没有取消订单。"
    assert "100" not in result.answer, "写在同一轮的半截答案不能漏进最终答案"


def test_plan_overflow_keeps_looping(tmp_path) -> None:
    """连续出计划时不再 ack，而是换成强提醒继续，直到模型真正去查。

    早先的实现把超限的 plan 当普通文本处理，结果 "plan: ..." 被当成最终答案，
    任务在第一轮就"成功"结束了（假完成）。所以超限必须继续循环。
    """
    result = _run(
        tmp_path,
        [
            _turn("plan: 第一条计划"),
            _turn("plan: 第二条计划"),
            _turn("plan: 第三条计划"),  # 超过 _MAX_PLANS → 强提醒
            _turn("answer: 数据里没有取消订单。"),
        ],
    )
    assert result.steps == 4, "超限的 plan 不能让循环结束"
    assert "plan_overflow" in result.error_kinds
    assert result.state == "answer"


def test_unsourced_answer_is_downgraded_to_unverified(tmp_path) -> None:
    """溯源失败两次后，答案照样交付，但状态降级为 unverified（而不是静默通过）。"""
    result = _run(
        tmp_path,
        [
            _turn("answer: 取消率 6.25%"),  # 6.25 没有查询出处
            _turn("answer: 取消率 6.25%"),  # 打回后仍没补上
        ],
    )
    assert result.state == "unverified"
    assert "6.25" in result.unsourced
    assert "unsourced_number" in result.error_kinds
