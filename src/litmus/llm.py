"""OpenAI 兼容的 chat 客户端。

只做四件事，但每件都是 agent 上线必需的：
- 重试：429 / 5xx / 网络抖动 → 指数退避，最多 4 次
- 限流：串行调用 + 最小间隔，避免把免费额度打出 429
- 缓存：key 含 prompt_version，改提示词不会读回旧结果
- 记账：每次调用的 token 与金额都返回，让"成本"成为可比较指标
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from litmus.config import Config

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}

# 本地推理端点（ollama / llama.cpp / LM Studio）共用的回环地址前缀
_LOOPBACK_PREFIXES = ("http://127.0.0.1", "http://localhost", "http://[::1]")


def _is_loopback(base_url: str) -> bool:
    """本地端点不校验 key，据此跳过强制 key 检查。"""
    return (base_url or "").lower().startswith(_LOOPBACK_PREFIXES)


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cost += other.cost


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(slots=True)
class AssistantTurn:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cached: bool = False

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, config: Config, *, min_interval_s: float = 0.0) -> None:
        # 本地推理端点（Ollama 等）不校验 key，用回环地址判断而不是给模型名开白名单，
        # 这样以后换 llama.cpp / LM Studio 只要也是 127.0.0.1 就不用改代码。
        if not config.api_key and not _is_loopback(config.base_url):
            raise LLMError(
                "缺少 API key：在 .env 里设置 LITMUS_API_KEY（或换 --model 前先配好 key）"
            )
        self.config = config
        self.min_interval_s = min_interval_s
        self._last_call = 0.0
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=httpx.Timeout(120.0, connect=15.0),
        )
        config.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 缓存 ----------
    def _cache_key(self, messages: list[dict], tools: list[dict] | None) -> str:
        payload = {
            "prompt_version": self.config.prompt_version,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": messages,
            "tools": tools or [],
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str):
        return self.config.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> AssistantTurn | None:
        if not self.config.use_cache:
            return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        turn = _parse_turn(data, self.config)
        turn.cached = True
        return turn

    def _write_cache(self, key: str, data: dict) -> None:
        if not self.config.use_cache:
            return
        self._cache_path(key).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )

    # ---------- 主入口 ----------
    def chat(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AssistantTurn:
        key = self._cache_key(messages, tools)
        hit = self._read_cache(key)
        if hit is not None:
            return hit

        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": False,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        data = self._post_with_retry("/chat/completions", body)
        self._write_cache(key, data)
        return _parse_turn(data, self.config)

    def _post_with_retry(self, path: str, body: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(4):
            self._throttle()
            try:
                response = self._client.post(path, json=body)
            except httpx.HTTPError as exc:  # 网络层抖动
                last_error = exc
            else:
                if response.status_code < 400:
                    return response.json()
                if response.status_code not in _RETRY_STATUS:
                    raise LLMError(
                        f"HTTP {response.status_code}: {response.text[:500]}"
                    )
                last_error = LLMError(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
            time.sleep(min(2**attempt, 8) + random.random())
        raise LLMError(f"重试 4 次仍失败：{last_error}")

    def _throttle(self) -> None:
        if self.min_interval_s <= 0:
            return
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()


# 推理模型（R1 蒸馏版、QwQ 等）会在正文前输出思考块，形式有  thinking、<thinking>、
# <reasoning> 几种。必须剥掉再进入答案，否则 litmus 的数字溯源会把思考过程里的
# 中间算式（"625/99441 ≈ 6.25%"）当成答案内容，一路误报 unverified。
_THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)
# 少数情况块没闭合（被 max_tokens 截断），只能砍到末尾
_THINK_OPEN = re.compile(r"<(think|thinking|reasoning)>.*\Z", re.S | re.I)


def _clean_content(text: str) -> str:
    """清洗两类脏数据。两个坑都是实测踩出来的。

    1. 推理模型的思考块：R1 蒸馏版会输出 `<think>` 包起来的草稿推理。直接保留会让
       答案溯源去校验草稿里的数字，把本来正确的答案判成不可信。
    2. 疑似流式 delta 残渣：智谱在 tool_calls 响应里会把
       `{"index":0,"finish_reason":...}` 这类流式结构塞进 message.content，
       实测这条脏数据能把答案变成 "…我接下来要走的是{"index":0,...}"。
    """
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_OPEN.sub("", text)
    for marker in ('{"index":', '{"id":"call', '{"id": "call'):
        position = text.find(marker)
        if position > 0:
            return text[:position].strip()
    return text.strip()


def _parse_turn(data: dict, config: Config) -> AssistantTurn:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    raw_calls = message.get("tool_calls") or []
    calls = [
        ToolCall(
            id=call.get("id") or os.urandom(6).hex(),
            name=(call.get("function") or {}).get("name", ""),
            arguments=(call.get("function") or {}).get("arguments") or "{}",
        )
        for call in raw_calls
    ]

    usage_raw = data.get("usage") or {}
    prompt_tokens = int(usage_raw.get("prompt_tokens") or 0)
    completion_tokens = int(usage_raw.get("completion_tokens") or 0)
    price_in, price_out = config.price()
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost=prompt_tokens / 1000 * price_in + completion_tokens / 1000 * price_out,
    )

    return AssistantTurn(
        content=_clean_content(message.get("content") or ""),
        tool_calls=calls,
        usage=usage,
    )
