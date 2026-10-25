"""配对对照实验：比较两个配置（模型或 prompt 版本）在同一套题上的差异。

为什么必须做配对而不是比两个准确率：
- 8/60 vs 12/60 这种比较忽略了"同一题"的信息。配对之后只有"一个对一个错"的
  不一致对才有信息量（都对/都错对判断差异没有贡献）
- 60 题里如果 b=8、c=2（b=A对B错，c=A错B对），结论是 A 更好且可检验
- 用精确 McNemar 检验（不依赖 scipy），避免"看起来涨了 5 个点"这种拍脑袋

这个脚本是 AgentEval 的第一个模块：跑评测只是收集数据，配对检验才是得出结论。

用法：
  python evals/compare.py --a evals/baselines/glm-4-flash-60q-v0.2.0.json \
                          --b evals/results/run-xxx.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_by_id"] = {record["id"]: record for record in payload["records"]}
    return payload


def mcnemar_exact(b: int, c: int) -> float:
    """精确 McNemar（二项检验），返回双侧 p 值。不需要 scipy。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="基线 results json")
    parser.add_argument("--b", required=True, help="对照组 results json")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    args = parser.parse_args()

    a = load(Path(args.a))
    b = load(Path(args.b))
    ids = [question_id for question_id in a["_by_id"] if question_id in b["_by_id"]]
    if not ids:
        print("两份结果没有共同题目，无法配对")
        return 1

    both_right = a_right_only = b_right_only = both_wrong = 0
    for question_id in ids:
        left = a["_by_id"][question_id]["correct"]
        right = b["_by_id"][question_id]["correct"]
        if left and right:
            both_right += 1
        elif left:
            a_right_only += 1
        elif right:
            b_right_only += 1
        else:
            both_wrong += 1

    acc_a = (both_right + a_right_only) / len(ids)
    acc_b = (both_right + b_right_only) / len(ids)

    print(f"配对样本：{len(ids)} 题")
    print(f"  {args.label_a:<18} 准确率 {acc_a:.3f}")
    print(f"  {args.label_b:<18} 准确率 {acc_b:.3f}")
    print(f"  差值 {acc_b - acc_a:+.3f}")
    print()
    print("2x2 配对表（只有对角线外的格子在提供证据）：")
    print(f"  都对                {both_right:>3}")
    print(f"  只有 {args.label_a} 对     {a_right_only:>3}")
    print(f"  只有 {args.label_b} 对     {b_right_only:>3}")
    print(f"  都错                {both_wrong:>3}")
    print()
    p_value = mcnemar_exact(a_right_only, b_right_only)
    discordant = a_right_only + b_right_only
    print(f"不一致对 {discordant} 个 → 精确 McNemar p = {p_value:.4f}")
    if discordant == 0:
        print("  → 两者在每一题上表现相同，没有证据表明有差异")
    elif p_value < 0.05:
        better = args.label_b if b_right_only > a_right_only else args.label_a
        print(f"  → 在 0.05 水平上显著，{better} 更好")
    else:
        print("  → 差异不显著：这个样本量还不足以支持结论")

    # 需要的样本量（粗略提示）：按当前不一致率推算
    if discordant:
        rate = discordant / len(ids)
        print(f"\n参考：当前不一致率 {rate:.0%}；"
              f"要稳定检出 10 个点以上的准确率差异，量级上需要 {int(10 / max(rate, 0.05)) * 10} 题左右的量级")

    print()
    print("按难度：")
    difficulties = sorted({a["_by_id"][i]["difficulty"] for i in ids})
    for difficulty in difficulties:
        subset = [i for i in ids if a["_by_id"][i]["difficulty"] == difficulty]
        ra = sum(a["_by_id"][i]["correct"] for i in subset) / len(subset)
        rb = sum(b["_by_id"][i]["correct"] for i in subset) / len(subset)
        print(f"  {difficulty:<10} {args.label_a} {ra:.2f}   {args.label_b} {rb:.2f}   ({len(subset)} 题)")

    print()
    print("效率与成本：")
    for label, payload in ((args.label_a, a), (args.label_b, b)):
        records = [payload["_by_id"][i] for i in ids]
        print(
            f"  {label:<18} 平均步数 {mean([r['steps'] for r in records]):.1f}"
            f"   token {sum(r['tokens_in'] + r['tokens_out'] for r in records):>8}"
            f"   成本 ¥{sum(r['cost'] for r in records):.4f}"
        )

    # 未收敛（打满步数）比例：能解释一部分准确率差异
    print()
    print("未收敛（打满步数）题数：")
    for label, payload in ((args.label_a, a), (args.label_b, b)):
        stalled = sum(
            1 for i in ids if "已达最大步数" in payload["_by_id"][i]["answer"]
        )
        print(f"  {label:<18} {stalled}/{len(ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
