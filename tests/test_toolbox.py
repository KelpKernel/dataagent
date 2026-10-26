"""ToolBox 行为测试：重复调用检测。

这条规则不是拍脑袋设计的：真实 trace 里模型对着 unknown_column 把同一条错 SQL
原样提交了 3 次，还连续 3 次 describe_table 同一张表，最后打满步数不收敛。
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from litmus.sandbox import Sandbox
from litmus.tools import ToolBox


@pytest.fixture()
def toolbox(tmp_path):
    db = tmp_path / "t.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        create table orders(order_id text, customer_id text);
        create table customers(customer_id text, customer_unique_id text);
        insert into orders values ('o1','c1'),('o2','c1'),('o3','c2');
        insert into customers values ('c1','u1'),('c2','u2');
        """
    )
    conn.commit()
    conn.close()
    box = Sandbox(db, row_limit=50, timeout_s=5)
    yield ToolBox(box)
    box.close()


def _sql(box: ToolBox, sql: str):
    return box.call("run_sql", json.dumps({"sql": sql}))


def test_repeated_sql_upgrades_error_kind(toolbox):
    sql = "select count(*) from orders where customer_unique_id in (select customer_unique_id from customers)"
    assert _sql(toolbox, sql).error_kind == "unknown_column"

    second = _sql(toolbox, sql)
    # 重复比"单次报错"更值得告诉模型：它卡住了
    assert second.error_kind == "repeated_call"
    assert "完全相同" in second.text


def test_repeated_ok_sql_flagged(toolbox):
    assert _sql(toolbox, "select count(*) from orders").error_kind is None
    assert _sql(toolbox, "select count(*) from orders").error_kind == "repeated_call"


def test_whitespace_and_case_dont_defeat_detection(toolbox):
    _sql(toolbox, "select count(*) from orders")
    assert _sql(toolbox, "  SELECT   COUNT(*)   FROM orders  ").error_kind == "repeated_call"


def test_repeated_describe_table_flagged(toolbox):
    assert toolbox.call("describe_table", '{"table": "orders"}').error_kind is None
    second = toolbox.call("describe_table", '{"table": "orders"}')
    assert second.error_kind == "repeated_call"
    assert "不要再重复获取" in second.text


def test_different_calls_not_flagged(toolbox):
    assert _sql(toolbox, "select count(*) from orders").error_kind is None
    assert _sql(toolbox, "select count(*) from customers").error_kind is None
    assert toolbox.call("describe_table", '{"table": "orders"}').error_kind is None
    assert toolbox.call("describe_table", '{"table": "customers"}').error_kind is None
