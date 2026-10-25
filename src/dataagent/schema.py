"""从 SQLite 元数据生成"给 LLM 看的 schema 文档"。

为什么不用向量检索：9 张表的 DDL 只有几千 token，全塞进上下文比检索更准也更快。
等表数上到几十张、schema 上百 KB 时才值得上检索——这个取舍会写进 README 的 Decision Log。
"""

from __future__ import annotations

import re
import sqlite3

_COL_COMMENT = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+[A-Za-z].*?--\s*(.+?)\s*$"
)

# 样例值只从每张表的前 N 行里取，避免在 100 万行的表上做全表 DISTINCT
_SAMPLE_SCAN_ROWS = 500
_SAMPLE_VALUES = 3


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "select name from sqlite_master where type='table' order by name"
    ).fetchall()
    return [r[0] for r in rows if not r[0].startswith("sqlite_")]


def table_columns(conn: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    rows = conn.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
    return [(r[1], (r[2] or "").upper()) for r in rows]


def _comments_from_ddl(ddl: str | None) -> dict[str, str]:
    """把 DDL 行尾的 `-- 注释` 归属到对应列上（这套 DDL 里自带中文列说明）。"""
    if not ddl:
        return {}
    found: dict[str, str] = {}
    for line in ddl.splitlines():
        match = _COL_COMMENT.match(line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def _sample_values(conn: sqlite3.Connection, table: str, column: str) -> list[str]:
    try:
        rows = conn.execute(
            f"select {quote_ident(column)} from {quote_ident(table)} "
            f"where {quote_ident(column)} is not null limit {_SAMPLE_SCAN_ROWS}"
        ).fetchall()
    except sqlite3.Error:
        return []
    seen: list[str] = []
    for (value,) in rows:
        text = str(value)
        if text not in seen:
            seen.append(text)
        if len(seen) >= _SAMPLE_VALUES:
            break
    return seen


def build_schema_text(
    conn: sqlite3.Connection, *, include_samples: bool = True
) -> str:
    """生成紧凑的 schema 文档：表名 + 行数 + 列（类型 + 注释）+ 文本列样例值。"""
    blocks: list[str] = []
    for table in list_tables(conn):
        row_count = conn.execute(
            f"select count(*) from {quote_ident(table)}"
        ).fetchone()[0]
        ddl = conn.execute(
            "select sql from sqlite_master where type='table' and name=?", (table,)
        ).fetchone()
        comments = _comments_from_ddl(ddl[0] if ddl else None)

        lines = [f"### {table}  ({row_count:,} 行)"]
        for column, ctype in table_columns(conn, table):
            note = comments.get(column, "")
            suffix = f"  -- {note}" if note else ""
            lines.append(f"- {column} {ctype}{suffix}".rstrip())

        if include_samples:
            for column, ctype in table_columns(conn, table):
                if ctype not in ("TEXT", ""):
                    continue
                values = _sample_values(conn, table, column)
                if values:
                    shown = ", ".join(values)
                    lines.append(f"  · {column} 样例: {shown}")

        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
