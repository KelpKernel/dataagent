"""本地推理支持（Ollama / R1 蒸馏版）的回归测试。

这条链路没接模型也能测——要守的是三件纯逻辑的事：
1. 推理模型的思考块必须剥掉，否则数字溯源会去校验草稿里的中间算式；
2. 回环地址不该被强制要求 API key（ollama 不校验）；
3. provider 表里 local 的端点与模型名要和文档一致。

背景见 docs/experiment-local-model.md。
"""

from __future__ import annotations

import pytest

from litmus.config import PROVIDERS, Config, resolve_endpoint
from litmus.llm import LLMClient, _clean_content, _is_loopback


# --------------------------------------------------------------- 思考块剥离


def test_strips_closed_think_block():
    """R1 风格的闭合思考块要被整段剥掉。"""
    text = "<think>先算 625/99441 ≈ 6.25%，再核对</think>answer: 取消率 0.63%"
    assert _clean_content(text) == "answer: 取消率 0.63%"


def test_strips_unclosed_think_block():
    """被 max_tokens 截断、没有闭合标签时，只能砍到末尾。"""
    text = "answer: 0.63%\n<think>让我再想想 625 除以 99441"
    assert _clean_content(text) == "answer: 0.63%"


@pytest.mark.parametrize("tag", ["think", "thinking", "reasoning", "THINK", "Think"])
def test_strips_all_think_tag_variants(tag):
    assert _clean_content(f"<{tag}>草稿 6.25%</{tag}>answer: 0.63%") == "answer: 0.63%"


def test_strips_multiple_think_blocks():
    text = "<think>第一步</think>中间<think>第二步 6.25%</think>answer: 0.63%"
    assert _clean_content(text) == "中间answer: 0.63%"


def test_think_digits_do_not_survive():
    """核心断言：思考块里的数字不能留在内容里（那是误报的根源）。"""
    cleaned = _clean_content("<think>625 / 99441 ≈ 6.25%</think>answer: 0.63%")
    assert "6.25" not in cleaned
    assert "625" not in cleaned
    assert "0.63" in cleaned


# ------------------------------------------------------- 原有清洗行为不能退化


def test_still_strips_zhipu_stream_residue():
    """智谱把流式 delta 结构塞进 content 的旧坑，行为必须保持。"""
    text = '答案在这里{"index":0,"finish_reason":"stop"}'
    assert _clean_content(text) == "答案在这里"


def test_plain_text_passes_through():
    assert _clean_content("  answer: 取消率 0.63%  ") == "answer: 取消率 0.63%"


# --------------------------------------------------------------- 本地 provider


def test_local_provider_registered():
    spec = PROVIDERS["local"]
    assert spec["base_url"].startswith("http://127.0.0.1")
    assert "deepseek-r1:7b" in spec["models"]


def test_resolve_endpoint_routes_local_model():
    base_url, _key = resolve_endpoint("deepseek-r1:7b")
    assert base_url == PROVIDERS["local"]["base_url"]


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://127.0.0.1:11434/v1", True),
        ("http://localhost:8080/v1", True),
        ("HTTP://LOCALHOST:1234/v1", True),
        ("https://api.deepseek.com", False),
        ("https://open.bigmodel.cn/api/paas/v4", False),
        ("", False),
    ],
)
def test_is_loopback(url, expected):
    assert _is_loopback(url) is expected


def test_local_endpoint_needs_no_api_key():
    """本地端点没有 key 也必须能构造客户端；远程端点仍然要 key。"""
    config = Config(db_path=":memory:", api_key="", base_url="http://127.0.0.1:11434/v1")
    client = LLMClient(config)
    client._client.close()

    remote = Config(db_path=":memory:", api_key="", base_url="https://api.deepseek.com")
    with pytest.raises(Exception, match="缺少 API key"):
        LLMClient(remote)
