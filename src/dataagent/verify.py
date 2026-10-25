"""答案数字溯源（output grounding）。

来由是实测：MVD 第一次跑通，模型答"取消率 625/99441 ≈ 6.25%",
实际是 0.63%——它把 99441 当成 10000 心算了，错 10 倍。
SQL 对了、行数对了，最后一步心算错了，这是 LLM 最典型的静默失败：
不报错、看着很自信、结果错一个数量级。

所以最终答案里的每个数字，都必须能在查询结果里找到出处。
找不到的不一定是错，但必须被标记——这是可审计性的下限。

换算规则（踩过坑才定成这么严的）：
第一版在"收集"和"匹配"两头都做 ×100/÷100，四层叠加后 625 能被凑成 6.25，
校验形同虚设。现在只允许一次换算，且方向由答案的写法决定：
- 答案写 "6.25%" → 允许与结果里的 0.0625 匹配
- 答案写 "0.0625" → 允许与结果里的 6.25 匹配
"""

from __future__ import annotations

import math
import re

_NUMBER = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:[.,]\d+)?")


def collect_from_values(values: list) -> list[float]:
    """收集查询结果里的原始数值，不做任何换算（换算只在匹配时做一次）。"""
    collected: list[float] = []
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            collected.append(float(value))
        elif isinstance(value, str):
            collected.extend(_parse_numbers(value))
    return collected


def unsourced_numbers(
    answer: str, source_values: list[float], *, rel_tol: float = 1e-2
) -> list[str]:
    """返回答案里无法溯源到查询结果的数字（按出现顺序）。"""
    missing: list[str] = []
    for match in _NUMBER.finditer(answer):
        token = match.group(0)
        value = _parse_token(token)
        if value is None:
            continue
        if not _matches_nearby(answer, match, value, source_values, rel_tol):
            missing.append(token)
    return missing


def _matches_nearby(
    answer: str,
    match: re.Match[str],
    value: float,
    source_values: list[float],
    rel_tol: float,
) -> bool:
    tail = answer[match.end() : match.end() + 2]
    candidates = [value]
    if "%" in tail or "％" in tail:
        candidates.append(value / 100)  # 6.25% ↔ 0.0625
    elif value != int(value):
        candidates.append(value * 100)  # 0.0625 ↔ 6.25
    return any(
        math.isclose(candidate, source, rel_tol=rel_tol, abs_tol=1e-9)
        for candidate in candidates
        for source in source_values
    )


def numbers_in(text: str) -> list[float]:
    """抽取文本里的所有数值（正确处理多级千分位）。"""
    out: list[float] = []
    for token in _NUMBER.findall(text):
        value = _parse_token(token)
        if value is not None:
            out.append(value)
    return out


def _parse_numbers(text: str) -> list[float]:
    return numbers_in(text)


def _parse_token(token: str) -> float | None:
    """区分千分位与小数。

    必须处理多级千分位：`12,542,084.19` 早期版本会被切成 `12,542` 和 `084.19`
    两个数，导致正确答案被判成错（DeepSeek 在 q22 上就栽在这）。
    """
    cleaned = token
    if token.count(",") >= 2:
        cleaned = token.replace(",", "")
    elif "," in token:
        head, _, tail = token.partition(",")
        cleaned = head + tail if len(tail) == 3 else head + "." + tail
    try:
        return float(cleaned)
    except ValueError:
        return None
