"""assistant 消息构造的兼容性测试。

踩过的坑：智谱容忍 `tool_calls: []`，DeepSeek 直接 400。
"""

from __future__ import annotations

from litmus.agent import _assistant_message
from litmus.llm import AssistantTurn, ToolCall


def test_no_tool_calls_omits_field():
    turn = AssistantTurn(content="最终答案", tool_calls=[])
    message = _assistant_message(turn)
    assert "tool_calls" not in message
    assert message["content"] == "最终答案"


def test_with_tool_calls_includes_field():
    turn = AssistantTurn(
        content="",
        tool_calls=[ToolCall(id="call_1", name="run_sql", arguments='{"sql":"select 1"}')],
    )
    message = _assistant_message(turn)
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["function"]["name"] == "run_sql"


def test_empty_tool_calls_roundtrip_is_serializable():
    """核心断言：空 tool_calls 不能被序列化成 [] 发出去。"""
    import json

    payload = _assistant_message(AssistantTurn(content="x", tool_calls=[]))
    assert "tool_calls" not in json.loads(json.dumps(payload))
