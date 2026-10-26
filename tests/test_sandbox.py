"""护栏与沙箱的回归测试（不依赖外部数据库，也不消耗 token）。"""

from __future__ import annotations

import sqlite3

import pytest

from litmus.sandbox import Sandbox, SQLBlocked, validate_sql


@pytest.fixture()
def sandbox(tmp_path):
    db = tmp_path / "t.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        create table orders(order_id text primary key, order_status text, ts text);
        create table customers(customer_id text primary key, city text);
        insert into orders values ('o1','delivered','2018-01-01 10:00:00');
        insert into orders values ('o2','shipped','2018-01-02 10:00:00');
        insert into customers values ('c1','sao paulo');
        """
    )
    conn.commit()
    conn.close()
    box = Sandbox(db, row_limit=1, timeout_s=1.0)
    yield box
    box.close()


@pytest.mark.parametrize(
    "sql",
    [
        "delete from orders",
        "update orders set order_status='x'",
        "drop table orders",
        "insert into orders values ('o3','x','y')",
        "select 1; select 2",
        "pragma table_info(orders)",
        "attach database 'x.db' as x",
    ],
)
def test_blocks_writes_and_multi(sql):
    with pytest.raises(SQLBlocked):
        validate_sql(sql, {"orders", "customers"})


def test_blocks_unknown_table():
    with pytest.raises(SQLBlocked) as exc:
        validate_sql("select * from orders_full", {"orders"})
    assert "表不存在" in str(exc.value)


def test_allows_cte_and_limits_rows(sandbox):
    result = sandbox.run(
        "with t as (select * from orders) select * from t order by order_id"
    )
    assert result.ok, result.error
    assert len(result.rows) == 1
    assert result.truncated is True


def test_unknown_column_is_classified(sandbox):
    result = sandbox.run("select nope from orders")
    assert not result.ok
    assert result.error_kind == "unknown_column"


def test_syntax_error_is_classified(sandbox):
    result = sandbox.run("select from where")
    assert not result.ok
    assert result.error_kind in {"syntax", "sqlite_error"}


def test_timeout_interrupts_runaway_query(sandbox):
    sandbox.timeout_s = 0.3
    result = sandbox.run(
        "with recursive c(x) as (select 1 union all select x+1 from c)"
        " select count(*) from c"
    )
    assert not result.ok
    assert result.error_kind == "timeout"
    # 中断后连接仍可用
    assert sandbox.run("select count(*) from orders").ok


def test_readonly_connection_blocks_writes(sandbox):
    conn = sandbox.connect()
    with pytest.raises(sqlite3.Error):
        conn.execute("create table hack(x)")
