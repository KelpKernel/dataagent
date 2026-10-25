"""离线重算评测结果：改判对口径不需要重跑模型。

来由：60 题首轮跑完发现 judge 口径偏严——模型答对 9803 时顺带报了 112650、98666
（都是真实查询结果），按数字覆盖率只有 0.18 就被判成错。
重跑一次要 40 分钟、1.2M token；而答案已经存在 results json 里，
金标准也能用 ref_sql 现场重算，所以离线重算就够了。

这件事本身就是 AgentEval 要解决的问题：**判对口径是评测的一部分，
口径改了却没能力重算，等于所有历史数据作废。**

用法：
  python evals/recompute.py                       # 重算最新一次结果
  python evals/recompute.py --file evals/results/run-20261025-190000.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from dataagent.config import Config  # noqa: E402
from dataagent.sandbox import Sandbox  # noqa: E402
from run_eval import gold_from_sql, judge, load_questions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", nargs="+", help="一个或多个 results json，默认取最新一个"
    )
    parser.add_argument("--db")
    parser.add_argument("--questions", default=str(ROOT / "evals" / "questions.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="只打印不覆盖文件")
    args = parser.parse_args()

    results_dir = ROOT / "evals" / "results"
    if args.file:
        targets = [Path(name) for name in args.file]
    else:
        latest = max(
            results_dir.glob("run-*.json"), key=lambda p: p.stat().st_mtime, default=None
        )
        targets = [latest] if latest else []
    targets = [path for path in targets if path and Path(path).exists()]
    if not targets:
        print("找不到 results json，先跑一次 evals/run_eval.py")
        return 1

    payloads = [
        (path, json.loads(Path(path).read_text(encoding="utf-8"))) for path in targets
    ]
    refs = {item["id"]: item for item in load_questions(Path(args.questions))}

    # gold 只建一次：60 条 ref_sql 要跑十几秒，重算多个文件时不能重复付这个代价
    needed = {record["id"] for _, payload in payloads for record in payload["records"]}
    config = Config.from_env(db_path=args.db)
    sandbox = Sandbox(config.db_path)
    golds = {}
    for question_id in sorted(needed):
        item = refs.get(question_id)
        if item is None:
            print(f"警告：{question_id} 不在题目文件里，跳过重算")
            continue
        kind, values, primary, _ = gold_from_sql(sandbox, item["ref_sql"])
        golds[question_id] = (kind, values, primary)
    sandbox.close()

    for target, payload in payloads:
        records = payload["records"]
        changed = []
        for record in records:
            gold = golds.get(record["id"])
            if gold is None:
                continue
            kind, values, primary = gold
            correct, coverage = judge(kind, record["answer"], values, primary)
            if correct != record["correct"]:
                changed.append((record["id"], record["correct"], correct))
            record["correct"] = correct
            record["coverage"] = round(coverage, 3)

        n = len(records)
        accuracy = round(sum(r["correct"] for r in records) / n, 3) if n else 0.0
        by_difficulty: dict[str, list[bool]] = {}
        for record in records:
            by_difficulty.setdefault(record["difficulty"], []).append(record["correct"])

        summary = payload.get("summary", {})
        summary.update(
            {
                "n": n,
                "accuracy": accuracy,
                "by_difficulty": {
                    key: round(sum(values) / len(values), 3)
                    for key, values in sorted(by_difficulty.items())
                },
                "judge_version": "primary_value_v2",
            }
        )
        payload["summary"] = summary

        print(f"文件：{target}")
        print(f"  重算后准确率：{accuracy}（{sum(r['correct'] for r in records)}/{n}）")
        for key, value in summary.get("by_difficulty", {}).items():
            print(f"    {key:<10} {value}")
        if changed:
            print(f"  判对结果变化 {len(changed)} 条：")
            for question_id, before, after in changed:
                print(f"    {question_id}: {'对' if before else '错'} → {'对' if after else '错'}")
        else:
            print("  判对结果没有变化")

        if not args.dry_run:
            Path(target).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
