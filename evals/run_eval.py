"""评测 runner（骨架版）。

评测口径（诚实版，先写清楚再跑）：
- 金标准由 ref_sql 现场执行得到（execution-based），不手写答案，避免人手抄错
- 单值题：最终答案里出现该数值（允许 %／小数一次换算）即判对
- 多行题：答案覆盖到的金标准数值比例 >= 0.5 判对（宽松，因为答案应当只给摘要）
- 同时记录步数、token、失败标记、未溯源数字——这些是 AgentEval 阶段要统计的原料

用法：
  python evals/run_eval.py --gold-only       # 只验证 12 条 ref_sql 能否跑通
  python evals/run_eval.py --limit 3         # 先跑 3 题试水
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataagent.config import Config  # noqa: E402
from dataagent.sandbox import Sandbox  # noqa: E402
from dataagent.verify import collect_from_values, _parse_token  # noqa: E402

_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def load_questions(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def gold_from_sql(sandbox: Sandbox, ref_sql: str):
    result = sandbox.run(ref_sql)
    if not result.ok:
        raise RuntimeError(f"参考 SQL 执行失败：{result.error_kind} {result.error}")
    values: list[float] = []
    for row in result.rows:
        values.extend(collect_from_values(list(row)))
    kind = "scalar" if len(result.rows) == 1 and len(result.columns) == 1 else "table"
    return kind, values, result


def _numbers_in(text: str) -> list[float]:
    out = []
    for token in _NUMBER.findall(text):
        value = _parse_token(token)
        if value is not None:
            out.append(value)
    return out


def _hit(value: float, pool: list[float], tol: float = 1e-2) -> bool:
    for gold in pool:
        for candidate in (value, value * 100, value / 100):
            if math.isclose(candidate, gold, rel_tol=tol, abs_tol=1e-9):
                return True
    return False


def judge(kind: str, answer: str, gold_values: list[float]) -> tuple[bool, float]:
    numbers = _numbers_in(answer)
    if not numbers:
        return False, 0.0
    hits = sum(1 for value in numbers if _hit(value, gold_values))
    coverage = hits / len(numbers)
    if kind == "scalar":
        return _hit(numbers[0], gold_values) or coverage >= 0.5, coverage
    return coverage >= 0.5, coverage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db")
    parser.add_argument("--questions", default=str(ROOT / "evals" / "questions.yaml"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gold-only", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--out", default=str(ROOT / "evals" / "results"))
    args = parser.parse_args()

    config = Config.from_env(db_path=args.db, model=args.model)
    questions = load_questions(Path(args.questions))
    if args.limit:
        questions = questions[: args.limit]

    sandbox = Sandbox(config.db_path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 第一步：先把金标准跑出来，坏的 ref_sql 不能进评测
    golds: dict[str, tuple[str, list[float]]] = {}
    for item in questions:
        kind, values, result = gold_from_sql(sandbox, item["ref_sql"])
        golds[item["id"]] = (kind, values)
        preview = result.to_table(max_rows=3).replace("\n", " / ")[:90]
        print(f"[gold] {item['id']:<4} {kind:<6} {preview}")

    if not args.gold_only:
        from dataagent.agent import DataAgent

        agent = DataAgent(config)
        records = []
        try:
            for item in questions:
                kind, gold_values = golds[item["id"]]
                started = time.perf_counter()
                result = agent.run(item["question"])
                elapsed = (time.perf_counter() - started) * 1000
                correct, coverage = judge(kind, result.answer, gold_values)
                record = {
                    "id": item["id"],
                    "difficulty": item["difficulty"],
                    "question": item["question"],
                    "answer": result.answer,
                    "correct": correct,
                    "coverage": round(coverage, 3),
                    "steps": result.steps,
                    "tokens_in": result.usage.prompt_tokens,
                    "tokens_out": result.usage.completion_tokens,
                    "cost": round(result.usage.cost, 6),
                    "elapsed_ms": round(elapsed, 1),
                    "error_kinds": result.error_kinds,
                    "unsourced": result.unsourced,
                }
                records.append(record)
                flag = "✓" if correct else "✗"
                print(
                    f"{flag} {item['id']:<4} {item['difficulty']:<9} "
                    f"cov={coverage:.2f} steps={result.steps} "
                    f"tok={result.tokens} {result.answer[:60]}"
                )
        finally:
            agent.close()

        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_file = out_dir / f"run-{stamp}.json"
        summary = {
            "model": config.model,
            "prompt_version": config.prompt_version,
            "n": len(records),
            "accuracy": round(sum(r["correct"] for r in records) / len(records), 3),
            "avg_steps": round(sum(r["steps"] for r in records) / len(records), 2),
            "total_tokens": sum(r["tokens_in"] + r["tokens_out"] for r in records),
            "total_cost": round(sum(r["cost"] for r in records), 6),
        }
        out_file.write_text(
            json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("\n== 汇总 ==")
        for key, value in summary.items():
            print(f"{key}: {value}")
        print(f"明细: {out_file}")

    sandbox.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
