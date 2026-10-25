"""只读 SQL 沙箱。

三道闸门，缺一不可：
1. AST 校验（sqlglot）：只放行单条只读查询，表名必须在白名单里，函数黑名单挡住文件/扩展类函数
2. 只读连接：mode=ro + PRAGMA query_only，即使校验被绕过也写不进去
3. 执行期中断：set_progress_handler 做超时，防止一条无 where 的笛卡尔积把进程挂死

第 3 条在 Windows 上没有更好的选择（signal.alarm 不可用），
用 progress handler 是跨平台且能真正打断 sqlite 内部循环的方式。
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import sqlglot
from sqlglot import exp

# 能读写文件或加载扩展的函数，一律拒绝
FORBIDDEN_FUNCS = {
    "load_extension",
    "readfile",
    "writefile",
    "fts3_tokenizer",
    "edit",
    "open",
}

_ERROR_KIND_RULES: list[tuple[str, str]] = [
    ("no such table", "unknown_table"),
    ("no such column", "unknown_column"),
    ("ambiguous column", "ambiguous_column"),
    ("syntax error", "syntax"),
    ("near ", "syntax"),
    ("no such function", "unknown_function"),
    ("interrupted", "timeout"),
    ("datatype mismatch", "type_mismatch"),
    ("misuse of aggregate", "misuse_aggregate"),
]


class SQLBlocked(Exception):
    """闸门 1 拦下的请求。kind 用于 trace 里的失败归类。"""

    def __init__(self, message: str, kind: str = "blocked") -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(slots=True)
class QueryResult:
    ok: bool
    sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    truncated: bool = False
    elapsed_ms: float = 0.0
    error: str | None = None
    error_kind: str | None = None

    @property
    def is_empty(self) -> bool:
        return self.ok and not self.rows

    def to_table(self, max_rows: int = 20) -> str:
        """渲染成 LLM 友好的文本表格。"""
        if not self.ok:
            return f"[{self.error_kind}] {self.error}"
        if not self.rows:
            return "（0 行）"
        shown = self.rows[:max_rows]
        lines = [" | ".join(self.columns)]
        for row in shown:
            lines.append(" | ".join("NULL" if v is None else str(v) for v in row))
        if len(self.rows) > len(shown):
            lines.append(f"... 还有 {len(self.rows) - len(shown)} 行未展示")
        if self.truncated:
            lines.append(f"... 结果被截断在 {len(self.rows)} 行上限")
        return "\n".join(lines)


def classify_error(message: str) -> str:
    lowered = message.lower()
    for needle, kind in _ERROR_KIND_RULES:
        if needle in lowered:
            return kind
    return "sqlite_error"


def validate_sql(sql: str, allowed_tables: set[str] | None = None) -> str:
    """闸门 1：AST 校验。返回归一化后的 SQL，失败抛 SQLBlocked。"""
    text = sql.strip().rstrip(";").strip()
    if not text:
        raise SQLBlocked("SQL 为空", "blocked")

    try:
        statements = sqlglot.parse(text, read="sqlite")
    except sqlglot.errors.ParseError as exc:
        raise SQLBlocked(f"SQL 解析失败：{exc}", "syntax") from exc

    parsed = [s for s in statements if s is not None]
    if len(parsed) != 1:
        raise SQLBlocked(f"只允许单条查询，收到 {len(parsed)} 条", "blocked")

    stmt = parsed[0]
    if not isinstance(stmt, exp.Query):
        raise SQLBlocked(
            f"只允许只读查询，拒绝 {type(stmt).__name__}", "blocked"
        )

    for func in stmt.find_all(exp.Func):
        name = func.sql_name().lower() if hasattr(func, "sql_name") else ""
        if name in FORBIDDEN_FUNCS:
            raise SQLBlocked(f"函数被禁用：{name}", "blocked")

    if allowed_tables:
        # CTE 定义的名字不是真实表，必须排除，否则 with t as (...) 会被误判
        cte_names = {cte.alias_or_name.lower() for cte in stmt.find_all(exp.CTE)}
        for table in stmt.find_all(exp.Table):
            name = table.name.lower()
            if not name or name in cte_names:
                continue
            if name not in allowed_tables:
                raise SQLBlocked(
                    f"表不存在：{name}（可用表：{', '.join(sorted(allowed_tables))}）",
                    "unknown_table",
                )

    return stmt.sql(dialect="sqlite")


class Sandbox:
    def __init__(
        self,
        db_path: str | Path,
        *,
        row_limit: int = 200,
        timeout_s: float = 15.0,
    ) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库不存在：{self.db_path}")
        self.row_limit = row_limit
        self.timeout_s = timeout_s
        self._conn: sqlite3.Connection | None = None
        self._tables: set[str] | None = None

    # 闸门 2：只读连接
    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            uri = f"file:{self.db_path.as_posix()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            conn.execute("PRAGMA query_only = ON")
            self._conn = conn
        return self._conn

    def tables(self) -> set[str]:
        if self._tables is None:
            rows = self.connect().execute(
                "select name from sqlite_master where type='table'"
            ).fetchall()
            self._tables = {
                r[0].lower() for r in rows if not r[0].startswith("sqlite_")
            }
        return self._tables

    # 闸门 3：执行期中断
    def run(self, sql: str) -> QueryResult:
        started = time.perf_counter()

        def elapsed() -> float:
            return (time.perf_counter() - started) * 1000

        try:
            validate_sql(sql, self.tables())
        except SQLBlocked as exc:
            return QueryResult(
                ok=False,
                sql=sql,
                error=str(exc),
                error_kind=exc.kind,
                elapsed_ms=elapsed(),
            )

        conn = self.connect()
        deadline = time.monotonic() + self.timeout_s
        conn.set_progress_handler(
            lambda: 1 if time.monotonic() > deadline else 0, 5000
        )
        try:
            cursor = conn.execute(sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(self.row_limit + 1)
            truncated = len(rows) > self.row_limit
            return QueryResult(
                ok=True,
                sql=sql,
                columns=columns,
                rows=rows[: self.row_limit],
                truncated=truncated,
                elapsed_ms=elapsed(),
            )
        except sqlite3.Error as exc:
            message = str(exc)
            kind = classify_error(message)
            if kind == "timeout":
                message = f"查询超过 {self.timeout_s:.0f}s 被中断，请缩小范围或加 where 条件"
            return QueryResult(
                ok=False,
                sql=sql,
                error=message,
                error_kind=kind,
                elapsed_ms=elapsed(),
            )
        finally:
            conn.set_progress_handler(None, 0)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
