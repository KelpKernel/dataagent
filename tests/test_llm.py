"""LLM 响应解析的回归测试（智谱的 content 污染是真实踩到的坑）。"""

from __future__ import annotations

from litmus.llm import _clean_content


def test_strips_stream_delta_residue():
    dirty = '我认为卡住的原因是 X。{"index":0,"finish_reason":"tool_calls","delta":null}'
    assert _clean_content(dirty) == "我认为卡住的原因是 X。"


def test_keeps_normal_text():
    assert _clean_content("取消率是 0.63%") == "取消率是 0.63%"


def test_keeps_text_when_json_is_at_position_zero():
    """整段都是 JSON 时不清空（宁可保留，也不要变成空答案）。"""
    assert _clean_content('{"index":0,"finish_reason":"stop"}') == '{"index":0,"finish_reason":"stop"}'


def test_strips_call_id_residue():
    dirty = '答案是 42{"id":"call_abc","type":"function"}'
    assert _clean_content(dirty) == "答案是 42"
