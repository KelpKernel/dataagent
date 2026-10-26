"""把 DataAgent 的护栏化 SQL 工具暴露成一个 MCP server（stdio）。

为什么不是"把整个 agent 包成 MCP"：
宿主（Claude Desktop / Cursor / Reasonix）本身就是模型，它已经在做规划和推理了。
再套一层 agent loop 只会多一次 LLM 调用、多一份成本和延迟。
所以这里暴露的是**能力**（只读沙箱 + schema 探索），规划交给宿主。

`ask` 工具仍然提供，但只在配了 API key 时才出现在 tools/list 里——
它的用途是"宿主不想自己写 SQL，直接让 DataAgent 跑完整闭环"。

实现取舍：
- 零第三方依赖（手写 JSON-RPC 2.0 over stdio），启动快、不用 pip install
- 所有日志写 stderr。**stdout 只能有协议帧**，混一行日志宿主就会断连
- stdio 传输是"一行一个 JSON"，不是 LSP 的 Content-Length 头

用法（宿主里配）：
  {"command": "python", "args": ["-m", "dataagent.mcp_server"], "env": {"DATAAGENT_DB": "..."}}
"""

from __future__ import annotations

import json
import sys
from typing import Any

from dataagent import PROMPT_VERSION, __version__
from dataagent.config import Config, is_demo_db
from dataagent.sandbox import Sandbox
from dataagent.tools import TOOLS, ToolBox, default_schema_text

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "dataagent"


def log(message: str) -> None:
    """日志一律走 stderr——stdout 被协议占用。"""
    print(f"[dataagent-mcp] {message}", file=sys.stderr, flush=True)


def _mcp_tools(include_ask: bool) -> list[dict[str, Any]]:
    """把内部工具定义转成 MCP 的 tools/list 格式。"""
    tools: list[dict[str, Any]] = [
        {
            # 宿主模型第一件事通常是"这库里有什么表"，一次给全比逐表问省好几轮
            "name": "schema_overview",
            "description": (
                "一次拿到全部表的结构：表名、行数、列名与类型、列注释、文本列的真实取值样例。"
                "表不多时先调这个，比逐个 describe_table 省调用。"
            ),
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        }
    ]
    tools += [
        {
            "name": item["function"]["name"],
            "description": item["function"]["description"],
            "inputSchema": item["function"]["parameters"],
        }
        for item in TOOLS
    ]
    if include_ask:
        tools.append(
            {
                "name": "ask",
                "description": (
                    "用自然语言提问，由 DataAgent 自己规划并执行 SQL（需要配置 API key）。"
                    "如果你（宿主模型）愿意自己写 SQL，用 run_sql 更省一次调用。"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "要回答的问题"}
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            }
        )
    return tools


class DataAgentServer:
    def __init__(self, db_path: str | None = None) -> None:
        self.config = Config.from_env(db_path)
        self.sandbox = Sandbox(
            self.config.db_path,
            row_limit=self.config.row_limit,
            timeout_s=self.config.query_timeout_s,
        )
        self.schema_text = default_schema_text(self.sandbox)
        self.toolbox = ToolBox(
            self.sandbox, schema_text=self.schema_text, row_limit=self.config.row_limit
        )
        self.demo = is_demo_db(self.config.db_path)
        log(
            f"db={self.config.db_path} demo={self.demo} "
            f"tables={len(self.sandbox.tables())} model={self.config.model}"
        )

    # ---------- 工具 ----------
    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "ask":
            return self._ask(str(arguments.get("question", "")))
        if name == "schema_overview":
            text = f"【数据库结构】\n{self.schema_text}"
            return self._text(text)
        outcome = self.toolbox.call(name, json.dumps(arguments, ensure_ascii=False))
        result: dict[str, Any] = self._text(outcome.text)
        if not outcome.ok:
            result["isError"] = True
        return result

    def _ask(self, question: str) -> dict[str, Any]:
        if not self.config.api_key:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "ask 需要 DATAAGENT_API_KEY。要么配上它，要么改用 run_sql 自己写 SQL。",
                    }
                ],
                "isError": True,
            }
        from dataagent.agent import DataAgent

        agent = DataAgent(self.config, sandbox=self.sandbox)
        try:
            result = agent.run(question)
        finally:
            agent.close()
        text = result.answer
        if result.unsourced:
            text += f"\n\n（未溯源数字：{'、'.join(result.unsourced)}）"
        return self._text(text)

    def _text(self, text: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": text}]}

    # ---------- JSON-RPC ----------
    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        request_id = message.get("id")

        # 通知（没有 id）不需要回包
        if request_id is None:
            if method == "notifications/initialized":
                log("initialized")
            return None

        if method == "initialize":
            return self._ok(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": __version__},
                    "instructions": (
                        "只读 SQL 查询工具。先用 list_tables / describe_table 确认列名和取值，"
                        "再用 run_sql 执行单条 SELECT。结果为空或为 0 时先怀疑口径，不要原样重试。"
                    ),
                },
            )

        if method == "ping":
            return self._ok(request_id, {})

        if method == "tools/list":
            return self._ok(
                request_id, {"tools": _mcp_tools(include_ask=bool(self.config.api_key))}
            )

        if method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                return self._error(request_id, -32602, "arguments 必须是对象")
            try:
                return self._ok(request_id, self.call_tool(str(name), arguments))
            except Exception as exc:  # 单次工具失败不应让整条连接断掉
                log(f"tool error: {type(exc).__name__}: {exc}")
                return self._ok(
                    request_id,
                    {
                        "content": [
                            {"type": "text", "text": f"[server_error] {type(exc).__name__}: {exc}"}
                        ],
                        "isError": True,
                    },
                )

        return self._error(request_id, -32601, f"未知方法：{method}")

    @staticmethod
    def _ok(request_id: Any, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    # ---------- 主循环 ----------
    def serve(self) -> int:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                payload = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"JSON 解析失败: {exc}"},
                }
                sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
                sys.stdout.flush()
                continue

            response = self.handle(message)
            if response is not None:
                sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                sys.stdout.flush()
        log("stdin closed, exiting")
        return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    db_path = None
    if argv:
        if argv[0] in ("-h", "--help"):
            print("用法：python -m dataagent.mcp_server [数据库路径]", file=sys.stderr)
            return 0
        db_path = argv[0]
    try:
        server = DataAgentServer(db_path)
    except Exception as exc:
        log(f"启动失败：{exc}")
        return 1
    return server.serve()


if __name__ == "__main__":
    raise SystemExit(main())
