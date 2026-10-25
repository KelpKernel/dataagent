"""四个工具 + 结果自检。

工具粒度本身就是设计决策：
- 不给 "execute anything"：只读、有行数上限、有超时
- 给 sample_rows：模型很容易把 order_status 写成 'Delivered'（真实值是 delivered），
  空结果往往就来源于这种“想当然”，所以必须让模型能自己核对枚举值
- run_sql 要带 purpose：模型得说明这条 SQL 想验证什么，
  失败分析时才能区分「SQL 写错」和「问题理解错」——这是两种完全不同的修法

关于 error_kind 的一个关键取舍：
工具执行成功但结果为空时，ok=True 而 error_kind='empty_result'。
「SQL 能跑通」和「答案对」是两件事，混在一起就失去了诊断价值。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import exp

from dataagent.sandbox import QueryResult, Sandbox
from dataagent.schema import build_schema_text, quote_ident, table_columns

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "列出数据库里所有表名与行数。不确定有哪些表时先调用它。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": "查看某张表的列名、类型、注释和文本列的真实取值样例。不确定列名或枚举值时调用。",
            "parameters": {
                "type": "object",
                "properties": {"table": {"type": "string", "description": "表名"}},
                "required": ["table"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sample_rows",
            "description": "抽样查看某张表的前几行真实数据，用于确认枚举值拼写、日期格式、id 形态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "表名"},
                    "limit": {
                        "type": "integer",
                        "description": "抽样行数，默认 5，最大 20",
                    },
                },
                "required": ["table"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "执行一条只读 SELECT 查询（SQLite 方言）。只允许单条查询、只读、有行数上限和超时。"
                "如果结果为空或返回 0，工具会给出 [empty_result] / [suspicious_zero] 标记，"
                "请先核对枚举值与过滤条件，不要原样重提同一条 SQL。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQLite SELECT 语句"},
                    "purpose": {
                        "type": "string",
                        "description": "这条 SQL 想验证什么（一句话），用于排查问题时复盘",
                    },
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass(slots=True)
class ToolOutcome:
    ok: bool
    text: str
    error_kind: str | None = None
    sql: str | None = None
    result: QueryResult | None = None


_EMPTY_HINT = (
    "结果为空。先别重写同一条 SQL，按顺序排查：\n"
    "1) 枚举值大小写/拼写是否与真实数据一致（用 sample_rows 或 describe_table 核对）\n"
    "2) 时间范围是否写错（本库日期是 TEXT，格式 'YYYY-MM-DD HH:MM:SS'）\n"
    "3) join 键是否正确（orders.customer_id 与 customers.customer_id 才是同一粒度，\n"
    "   customers.customer_unique_id 才是跨订单的真实客户标识）"
)

_ZERO_HINT = (
    "结果是 0 或 NULL。若与预期不符，优先检查 join 是否丢行、过滤条件是否过严。"
)

_NO_FROM_HINT = (
    "外层聚合没有 FROM 子句，SQLite 里此时 count(*) 恒等于 1，分母会悄悄变成 1。"
    "把分母写成 (select count(*) from orders) 这样的子查询，或用 case when 聚合。"
)

_RATIO_HINT = (
    "这个比例不在 0–100 之间，几乎肯定是分母算错了（常见：聚合没有 FROM、"
    "join 后行数被放大、拿订单数除订单项数）。请重算分母并确认口径。"
)

_RESULT_HINTS = {
    "empty_result": _EMPTY_HINT,
    "suspicious_zero": _ZERO_HINT,
    "aggregate_without_from": _NO_FROM_HINT,
    "suspicious_ratio": _RATIO_HINT,
}


def self_check(result: QueryResult, sql: str = "") -> str | None:
    """结果自检：返回值得提示模型的 error_kind，None 表示结果看起来正常。"""
    if not result.ok:
        return result.error_kind
    if not result.rows:
        return "empty_result"
    # 外层聚合没有 FROM：SQLite 里 count(*) 此时恒为 1，分母悄悄变成 1
    if _aggregate_without_from(sql):
        return "aggregate_without_from"
    if len(result.rows) == 1 and len(result.columns) == 1:
        value = result.rows[0][0]
        if value is None or value in (0, "0", 0.0):
            return "suspicious_zero"
        if isinstance(value, (int, float)) and _looks_like_ratio(sql):
            if value < 0 or value > 100:
                return "suspicious_ratio"
    return None


def _aggregate_without_from(sql: str) -> bool:
    if not sql:
        return False
    try:
        stmt = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.SqlglotError:
        return False
    if not isinstance(stmt, exp.Select):
        return False
    if stmt.find(exp.AggFunc) is None:
        return False
    # 注意：sqlglot 30 的键名是 from_（带下划线），不是 from。
    # 用 find(exp.From) 会命中子查询里的 FROM，所以只能看最外层。
    has_source = (
        stmt.args.get("from_") is not None
        or stmt.args.get("from") is not None
        or bool(stmt.args.get("joins"))
    )
    return not has_source


def _looks_like_ratio(sql: str) -> bool:
    lowered = sql.lower()
    return "100.0" in lowered or "percent" in lowered or "pct" in lowered


class ToolBox:
    def __init__(
        self, sandbox: Sandbox, *, schema_text: str | None = None, row_limit: int = 200
    ) -> None:
        self.sandbox = sandbox
        self.row_limit = row_limit
        self._schema_text = schema_text
        self._cache: dict[str, str] = {}

    # ---------- 分发 ----------
    def call(self, name: str, raw_args: str) -> ToolOutcome:
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError as exc:
            return ToolOutcome(
                ok=False,
                text=f"参数不是合法 JSON：{exc}。请重新生成 function arguments。",
                error_kind="invalid_args",
            )
        if not isinstance(args, dict):
            return ToolOutcome(
                ok=False, text="参数必须是 JSON 对象", error_kind="invalid_args"
            )

        handler = {
            "list_tables": self._list_tables,
            "describe_table": self._describe_table,
            "sample_rows": self._sample_rows,
            "run_sql": self._run_sql,
        }.get(name)
        if handler is None:
            return ToolOutcome(
                ok=False,
                text=f"未知工具：{name}。可用工具：list_tables / describe_table / sample_rows / run_sql",
                error_kind="unknown_tool",
            )
        return handler(args)

    # ---------- 工具实现 ----------
    def _list_tables(self, _args: dict) -> ToolOutcome:
        conn = self.sandbox.connect()
        lines = ["表名 | 行数"]
        for table in sorted(self.sandbox.tables()):
            count = conn.execute(
                f"select count(*) from {quote_ident(table)}"
            ).fetchone()[0]
            lines.append(f"{table} | {count:,}")
        return ToolOutcome(ok=True, text="\n".join(lines))

    def _describe_table(self, args: dict) -> ToolOutcome:
        table = str(args.get("table", "")).strip()
        key = f"describe:{table.lower()}"
        if key in self._cache:
            return ToolOutcome(ok=True, text=self._cache[key])
        if table.lower() not in self.sandbox.tables():
            return ToolOutcome(
                ok=False,
                text=f"表不存在：{table}。可用表：{', '.join(sorted(self.sandbox.tables()))}",
                error_kind="unknown_table",
            )
        conn = self.sandbox.connect()
        ddl = conn.execute(
            "select sql from sqlite_master where type='table' and name=?", (table,)
        ).fetchone()
        lines = [f"### {table}"]
        if ddl and ddl[0]:
            lines.append(ddl[0])
        lines.append("")
        for column, ctype in table_columns(conn, table):
            samples = self._column_samples(conn, table, column, ctype)
            suffix = f"  样例: {', '.join(samples)}" if samples else ""
            lines.append(f"- {column} {ctype}{suffix}")
        text = "\n".join(lines)
        self._cache[key] = text
        return ToolOutcome(ok=True, text=text)

    def _column_samples(
        self, conn: sqlite3.Connection, table: str, column: str, ctype: str
    ) -> list[str]:
        if ctype not in ("TEXT", ""):
            return []
        try:
            rows = conn.execute(
                f"select {quote_ident(column)} from {quote_ident(table)} "
                f"where {quote_ident(column)} is not null limit 200"
            ).fetchall()
        except sqlite3.Error:
            return []
        seen: list[str] = []
        for (value,) in rows:
            text = str(value)
            if text not in seen:
                seen.append(text)
            if len(seen) >= 3:
                break
        return seen

    def _sample_rows(self, args: dict) -> ToolOutcome:
        table = str(args.get("table", "")).strip()
        if table.lower() not in self.sandbox.tables():
            return ToolOutcome(
                ok=False,
                text=f"表不存在：{table}。可用表：{', '.join(sorted(self.sandbox.tables()))}",
                error_kind="unknown_table",
            )
        try:
            limit = int(args.get("limit") or 5)
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(limit, 20))
        result = self.sandbox.run(
            f"select * from {quote_ident(table)} limit {limit}"
        )
        return ToolOutcome(
            ok=result.ok,
            text=result.to_table(max_rows=limit),
            error_kind=result.error_kind,
            result=result,
        )

    def _run_sql(self, args: dict) -> ToolOutcome:
        sql = str(args.get("sql", "")).strip()
        result = self.sandbox.run(sql)
        kind = self_check(result, sql)
        lines: list[str] = []
        if not result.ok:
            lines.append(f"[{kind}] {result.error}")
            lines.append(self._fix_hint(kind))
        else:
            lines.append(result.to_table())
            lines.append(f"（{len(result.rows)} 行 / {result.elapsed_ms:.0f}ms）")
            hint = _RESULT_HINTS.get(kind or "")
            if hint:
                lines.append(f"[{kind}] {hint}")
        return ToolOutcome(
            ok=result.ok,
            text="\n".join(lines),
            error_kind=kind,
            sql=sql,
            result=result,
        )

    def _fix_hint(self, kind: str | None) -> str:
        hints = {
            "unknown_column": "列名不存在。用 describe_table 核对真实列名（注意 products 表里是 product_name_lenght，原始数据拼写如此）。",
            "unknown_table": "表名不存在。用 list_tables 核对。",
            "ambiguous_column": "列名有歧义。给列加表名限定，例如 orders.customer_id。",
            "syntax": "语法错误。本库是 SQLite：日期可用 substr() 或 strftime() 处理 TEXT 字段。",
            "timeout": "查询超时。加 where 条件缩小范围，或先聚合再 join。",
            "blocked": "被护栏拦截。只允许单条只读 SELECT。",
            "unknown_function": "函数不存在。SQLite 不支持部分 PostgreSQL/MySQL 函数。",
        }
        return hints.get(kind or "", "")


def default_schema_text(sandbox: Sandbox) -> str:
    return build_schema_text(sandbox.connect())
