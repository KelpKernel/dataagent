"""从 trace 计算护栏效果：命中率、修复率、失败模式分布。

为什么单独做这个脚本：
准确率只告诉你"对了几个"，修复率告诉你"护栏有没有起作用"。
这是两个完全不同的数——一个模型可能靠猜拿到不低的准确率，
但修复率才是 agent 能力的直接证据，也是后面做模型对照实验的核心指标。

指标定义（写清楚，否则同一份 trace 能算出三种修复率）：
- 护栏命中（hit）：一次工具调用返回了非空 error_kind
- 修复（repair）：某次出错之后，同一次运行里的**下一次 run_sql** 成功
- 重试仍失败（failed_retry）：出错后还有 SQL 尝试，但那次也失败
- 修复率（repair_rate）= repair / (repair + failed_retry)
  只在"还有机会改"时算分母，否则打满步数的运行会被误判成修复能力差
- 放弃（give_up）：出错过但之后再没有新的 run_sql（步数耗尽或模型直接给答案）

用法：
  python evals/analyze_trace.py
  python evals/analyze_trace.py --trace .dataagent/trace.sqlite --detail
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class Step:
    idx: int
    kind: str
    name: str
    error_kind: str | None
    ok: bool
    sql: str | None


@dataclass(slots=True)
class RunStat:
    run_id: str
    question: str
    status: str
    steps: int
    tokens: int
    errors: int
    repairs: int
    failed_retry: int
    gives_up: int
    error_kinds: list[str]


def load(trace_path: Path) -> tuple[list[RunStat], Counter, int]:
    conn = sqlite3.connect(trace_path)
    conn.row_factory = sqlite3.Row
    runs = conn.execute(
        "select run_id, question, status, steps, tokens_in, tokens_out"
        " from runs order by started_at"
    ).fetchall()

    stats: list[RunStat] = []
    kind_counter: Counter = Counter()
    tool_calls = 0

    for run in runs:
        steps = [
            Step(s["idx"], s["kind"], s["name"] or "", s["error_kind"], bool(s["ok"]), s["sql"])
            for s in conn.execute(
                "select idx, kind, name, error_kind, ok, sql from steps"
                " where run_id = ? order by idx",
                (run["run_id"],),
            )
        ]
        tool_calls += sum(1 for s in steps if s.kind == "tool")

        errors = repairs = failed_retry = gives_up = 0
        kinds: list[str] = []
        for position, step in enumerate(steps):
            if step.kind != "tool" or not step.error_kind:
                continue
            errors += 1
            kind_counter[step.error_kind] += 1
            kinds.append(step.error_kind)
            following = next(
                (
                    later
                    for later in steps[position + 1 :]
                    if later.kind == "tool" and later.name == "run_sql"
                ),
                None,
            )
            if following is None:
                gives_up += 1
            elif following.ok and not following.error_kind:
                repairs += 1
            else:
                failed_retry += 1

        stats.append(
            RunStat(
                run_id=run["run_id"],
                question=run["question"],
                status=run["status"],
                steps=run["steps"],
                tokens=(run["tokens_in"] or 0) + (run["tokens_out"] or 0),
                errors=errors,
                repairs=repairs,
                failed_retry=failed_retry,
                gives_up=gives_up,
                error_kinds=kinds,
            )
        )

    conn.close()
    return stats, kind_counter, tool_calls


def report(
    stats: list[RunStat], kind_counter: Counter, tool_calls: int, *, detail: bool
) -> None:
    if not stats:
        print("trace 里还没有运行记录")
        return

    total_errors = sum(s.errors for s in stats)
    total_repairs = sum(s.repairs for s in stats)
    total_failed = sum(s.failed_retry for s in stats)
    total_giveups = sum(s.gives_up for s in stats)
    opportunities = total_repairs + total_failed

    print("=== 总览 ===")
    print(f"运行数            {len(stats)}")
    print(f"工具调用          {tool_calls}")
    if tool_calls:
        print(
            f"护栏命中          {total_errors}"
            f"  （占工具调用 {total_errors / tool_calls * 100:.1f}%）"
        )
    else:
        print(f"护栏命中          {total_errors}")
    print(f"重试后成功        {total_repairs}")
    print(f"重试后仍失败      {total_failed}")
    print(f"未再尝试(放弃)    {total_giveups}")
    rate = total_repairs / opportunities * 100 if opportunities else 0.0
    print(f"修复率            {rate:.1f}%  （{total_repairs}/{opportunities} 次重试机会）")
    print(f"平均命中/次运行   {total_errors / len(stats):.2f}")
    print()
    print("=== 失败模式分布 ===")
    for kind, count in kind_counter.most_common():
        share = count / total_errors * 100 if total_errors else 0
        print(f"  {kind:<26} {count:>3}  {share:5.1f}%")
    print()
    print("=== 按题目聚合（护栏在哪些题上真起作用）===")
    by_question: dict[str, list[RunStat]] = defaultdict(list)
    for stat in stats:
        by_question[stat.question].append(stat)
    rows = []
    for question, group in by_question.items():
        repairs = sum(g.repairs for g in group)
        failed = sum(g.failed_retry for g in group)
        if repairs + failed:
            rows.append((repairs / (repairs + failed), repairs + failed, repairs, question))
    for ratio, opportunities_count, repairs, question in sorted(rows, reverse=True)[:10]:
        print(
            f"  {ratio * 100:5.1f}%  ({repairs}/{opportunities_count})  {question[:42]}"
        )

    if detail:
        print()
        print("=== 逐条运行 ===")
        for stat in sorted(stats, key=lambda s: (-s.errors, s.steps)):
            flag = "✗" if stat.status != "ok" else "✓"
            kinds = ",".join(stat.error_kinds) or "-"
            print(
                f"{flag} {stat.run_id} steps={stat.steps:>2} hit={stat.errors:>2} "
                f"repair={stat.repairs} fail={stat.failed_retry} giveup={stat.gives_up} "
                f"tok={stat.tokens:>6} | {kinds} | {stat.question[:30]}"
            )

    print()
    print("（口径：修复只在“出错后还有下一次 SQL 尝试”时计入分母；同一题多次运行会重复计入）")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default=str(ROOT / ".dataagent" / "trace.sqlite"))
    parser.add_argument("--detail", action="store_true", help="打印逐条运行明细")
    args = parser.parse_args()

    path = Path(args.trace)
    if not path.exists():
        print(f"找不到 trace：{path}")
        return 1
    stats, kind_counter, tool_calls = load(path)
    report(stats, kind_counter, tool_calls, detail=args.detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
