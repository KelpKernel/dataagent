"""数字溯源的回归测试。这层逻辑的假阳性会直接让护栏失效，必须有测试守着。"""

from __future__ import annotations

import pytest

from dataagent.verify import collect_from_values, unsourced_numbers


def test_flags_mental_arithmetic():
    """真实失败案例：625/99441 被心算成 6.25%（实际 0.63%）。"""
    pool = collect_from_values([625, 99441])
    missing = unsourced_numbers("625 / 99441 ≈ 0.0625，即 6.25%", pool)
    assert "0.0625" in missing
    assert "6.25" in missing
    # 两个原始数字有出处，不应被误报
    assert "625" not in missing
    assert "99441" not in missing


def test_accepts_sourced_ratio():
    """SQL 算出来的比率要能通过校验（百分比写法与小数写法都算命中）。"""
    pool = collect_from_values([0.6286])
    assert unsourced_numbers("取消率约 0.6286", pool) == []
    assert unsourced_numbers("取消率约 62.86%", pool) == []


def test_no_double_conversion_false_positive():
    """池里有 625 不代表答案里的 6.25 就有出处（第一版的四层换算 bug）。"""
    pool = collect_from_values([625])
    assert unsourced_numbers("约 6.25%", pool) == ["6.25"]


def test_thousands_separator_is_not_decimal():
    pool = collect_from_values([99441])
    assert unsourced_numbers("共 99,441 单", pool) == []


@pytest.mark.parametrize(
    "answer,pool,expected",
    [
        ("共 625 单", [625], []),
        ("来自 2018 年的 3000 单", [3000], ["2018"]),
    ],
)
def test_basic_cases(answer, pool, expected):
    assert unsourced_numbers(answer, collect_from_values(pool)) == expected
