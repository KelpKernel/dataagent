"""数字解析的回归测试。

多级千分位是真实踩到的坑：DeepSeek 在 q22 答对 12,542,084.19，
早期版本的正则把它切成 `12,542` 和 `084.19` 两个数，正确答案被判成错。
"""

from __future__ import annotations

import pytest

from dataagent.verify import numbers_in, unsourced_numbers


@pytest.mark.parametrize(
    "text,expected",
    [
        ("12,542,084.19", [12542084.19]),
        ("共 99,441 单", [99441.0]),
        ("5,998,226.96 元", [5998226.96]),
        ("625", [625.0]),
        ("0.0625", [0.0625]),
        ("6.25%", [6.25]),
        ("41,746", [41746.0]),
        ("1,234 和 5.5", [1234.0, 5.5]),
    ],
)
def test_numbers_in(text, expected):
    assert numbers_in(text) == pytest.approx(expected)


def test_multilevel_thousands_token_is_sourced():
    """千分位写法的正确答案必须能溯源。"""
    pool = numbers_in("12542084.19")
    assert unsourced_numbers("信用卡总金额 12,542,084.19", pool) == []


def test_mental_arithmetic_still_flagged():
    pool = numbers_in("625 99441")
    assert "6.25" in unsourced_numbers("625 / 99441 ≈ 6.25%", pool)
