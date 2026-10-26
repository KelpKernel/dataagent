"""trace 落盘（SQLite）。

从第一行代码就写 trace，不是可选项：
- 调试时能看到模型在哪一步走错
- 第 2 阶段的 AgentEval 直接吃这张表做采样噪声与显著性分析
- "失败模式分类"这份报告的数据源就是 error_kind 字段
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
create table if not exists runs(
    run_id          text primary key,
    question        text not null,
    model           text,
    prompt_version  text,
    started_at      text,
    finished_at     text,
    status          text,
    answer          text,
    steps           integer default 0,
    tokens_in       integer default 0,
    tokens_out      integer default 0,
    cost            real    default 0,
    elapsed_ms      real    default 0
);

create table if not exists steps(
    run_id          text not null,
    idx             integer not null,
    kind            text,            -- assistant / tool
    name            text,            -- 工具名
    args            text,
    sql             text,
    result_preview  text,
    error_kind      text,            -- syntax / unknown_column / empty_result / ...
    ok              integer,
    latency_ms      real,
    tokens_in       integer,
    tokens_out      integer,
    cost            real,
    ts              text,
    primary key (run_id, idx)
);

create index if not exists idx_runs_question on runs(question);
create index if not exists idx_steps_error on steps(error_kind);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class TraceStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self.run_id = ""
        self._step = 0
        self._started = 0.0

    # ---------- 一次运行 ----------
    def start(self, question: str, model: str, prompt_version: str) -> str:
        self.run_id = uuid.uuid4().hex[:12]
        self._step = 0
        self._started = time.perf_counter()
        self._conn.execute(
            "insert into runs(run_id, question, model, prompt_version, started_at, status)"
            " values (?,?,?,?,?,?)",
            (self.run_id, question, model, prompt_version, _now(), "running"),
        )
        self._conn.commit()
        return self.run_id

    def add_step(
        self,
        *,
        kind: str,
        name: str = "",
        args: Any = None,
        sql: str | None = None,
        result_preview: str = "",
        error_kind: str | None = None,
        ok: bool = True,
        latency_ms: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost: float = 0.0,
    ) -> None:
        self._step += 1
        self._conn.execute(
            "insert into steps(run_id, idx, kind, name, args, sql, result_preview,"
            " error_kind, ok, latency_ms, tokens_in, tokens_out, cost, ts)"
            " values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                self.run_id,
                self._step,
                kind,
                name,
                json.dumps(args, ensure_ascii=False) if args is not None else None,
                sql,
                result_preview[:2000],
                error_kind,
                int(ok),
                latency_ms,
                tokens_in,
                tokens_out,
                cost,
                _now(),
            ),
        )

    def finish(
        self,
        *,
        status: str,
        answer: str,
        tokens_in: int,
        tokens_out: int,
        cost: float,
    ) -> None:
        elapsed = (time.perf_counter() - self._started) * 1000
        self._conn.execute(
            "update runs set finished_at=?, status=?, answer=?, steps=?,"
            " tokens_in=?, tokens_out=?, cost=?, elapsed_ms=? where run_id=?",
            (
                _now(),
                status,
                answer,
                self._step,
                tokens_in,
                tokens_out,
                cost,
                elapsed,
                self.run_id,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
