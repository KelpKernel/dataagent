"""MCP server 的端到端测试：真的启动子进程、走 stdio、发 JSON-RPC。"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "t.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        create table orders(order_id text primary key, order_status text, ts text);
        create table customers(customer_id text primary key, city text);
        insert into orders values ('o1','delivered','2018-01-01 10:00:00');
        insert into orders values ('o2','canceled','2018-01-02 10:00:00');
        insert into orders values ('o3','delivered','2018-02-01 10:00:00');
        insert into customers values ('c1','sao paulo');
        """
    )
    conn.commit()
    conn.close()
    return path


def run_session(db_path: Path, messages: list[dict], *, with_api_key: bool = False):
    """把整段会话喂给子进程，一次性收集所有响应。"""
    payload = "\n".join(json.dumps(m, ensure_ascii=False) for m in messages) + "\n"
    env = {
        **os.environ,
        "PYTHONPATH": str(SRC),
        "PYTHONIOENCODING": "utf-8",
        "LITMUS_DB": "",
        "LITMUS_API_KEY": "test-key" if with_api_key else "",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "litmus.mcp_server", str(db_path)],
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        env=env,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return [json.loads(line) for line in lines], proc


def init_msg() -> dict:
    return {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}


def call_tool(request_id: int, name: str, arguments: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def test_initialize_and_tools_list(db):
    responses, proc = run_session(
        db, [init_msg(), {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
    )
    assert proc.returncode == 0, proc.stderr

    init = responses[0]["result"]
    assert init["protocolVersion"] == "2024-11-05"
    assert init["serverInfo"]["name"] == "litmus"
    assert "tools" in init["capabilities"]

    names = {tool["name"] for tool in responses[1]["result"]["tools"]}
    assert {"list_tables", "describe_table", "sample_rows", "run_sql", "schema_overview"} <= names
    # 没有 API key 时不暴露 ask（省一次 LLM 调用，也不误导宿主）
    assert "ask" not in names


def test_ask_only_with_api_key(db):
    responses, _ = run_session(
        db, [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}], with_api_key=True
    )
    names = {tool["name"] for tool in responses[0]["result"]["tools"]}
    assert "ask" in names


def test_run_sql_returns_rows(db):
    responses, proc = run_session(
        db, [call_tool(1, "run_sql", {"sql": "select count(*) as n from orders"})]
    )
    assert proc.returncode == 0, proc.stderr
    result = responses[0]["result"]
    assert "isError" not in result or result["isError"] is False
    text = result["content"][0]["text"]
    assert "3" in text


def test_list_tables_and_describe(db):
    responses, _ = run_session(
        db,
        [
            call_tool(1, "list_tables", {}),
            call_tool(2, "describe_table", {"table": "orders"}),
        ],
    )
    assert "orders" in responses[0]["result"]["content"][0]["text"]
    described = responses[1]["result"]["content"][0]["text"]
    assert "order_status" in described and "样例" in described


def test_write_is_blocked(db):
    responses, _ = run_session(db, [call_tool(1, "run_sql", {"sql": "delete from orders"})])
    result = responses[0]["result"]
    assert result["isError"] is True
    assert "blocked" in result["content"][0]["text"]


def test_unknown_tool_is_flagged(db):
    responses, _ = run_session(db, [call_tool(1, "drop_everything", {})])
    result = responses[0]["result"]
    assert result["isError"] is True
    assert "未知工具" in result["content"][0]["text"]


def test_unknown_method_returns_error(db):
    responses, _ = run_session(db, [{"jsonrpc": "2.0", "id": 1, "method": "does_not_exist"}])
    assert responses[0]["error"]["code"] == -32601


def test_notifications_get_no_reply(db):
    responses, _ = run_session(
        db,
        [
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 9, "method": "ping"},
        ],
    )
    # 通知不该回包，所以只有 ping 一条响应
    assert len(responses) == 1
    assert responses[0]["id"] == 9
    assert responses[0]["result"] == {}


def test_stdout_stays_clean(db):
    """日志必须走 stderr——stdout 混进一行非 JSON 宿主就会断连。"""
    _, proc = run_session(db, [init_msg()])
    for line in proc.stdout.splitlines():
        if line.strip():
            json.loads(line)  # 抛异常即失败
    assert "litmus-mcp" in proc.stderr


def test_empty_result_carries_hint(db):
    """空结果要带排查提示，这是项目护栏的一部分。"""
    responses, _ = run_session(
        db, [call_tool(1, "run_sql", {"sql": "select * from orders where order_status = 'nope'"})]
    )
    text = responses[0]["result"]["content"][0]["text"]
    assert "empty_result" in text
