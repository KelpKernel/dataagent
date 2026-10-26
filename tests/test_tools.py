"""结果自检的回归测试。误报会污染失败率统计，必须以测试锁死。"""

from __future__ import annotations

from litmus.sandbox import QueryResult
from litmus.tools import self_check


def _ok(rows, columns=("v",)) -> QueryResult:
    return QueryResult(ok=True, columns=list(columns), rows=rows)


def test_aggregate_with_from_is_not_flagged():
    """带 FROM 的 count(*) 不能被误报（sqlglot 30 的键名是 from_）。"""
    sql = "select count(*) from orders where order_status = 'canceled'"
    assert self_check(_ok([(625,)], ("COUNT(*)",)), sql) is None


def test_aggregate_without_from_is_flagged():
    sql = "select round(100.0 * (select count(*) from orders) / count(*), 2)"
    assert self_check(_ok([(62500.0,)]), sql) == "aggregate_without_from"


def test_suspicious_ratio():
    sql = "select 100.0 * count(*) / (select count(*) from orders) from orders"
    assert self_check(_ok([(6250.0,)]), sql) == "suspicious_ratio"


def test_zero_flagged():
    assert self_check(_ok([(0,)], ("n",)), "select count(*) from orders") == "suspicious_zero"


def test_empty_flagged():
    assert self_check(_ok([]), "select 1") == "empty_result"


def test_normal_value_passes():
    assert self_check(_ok([(625,)], ("n",)), "select count(*) from orders") is None
