"""命令行入口。

三个子命令刻意分开：
- schema / sql 不调用 LLM，用来验证护栏和数据本身（离线可跑）
- ask 才是 agent
这样在没配 key 或想复现某个报错时，不必花 token。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _setup_console() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # 老终端不支持时忽略
        pass


def _config(args):
    from dataagent.config import Config

    prompt = getattr(args, "prompt", None)
    return Config.from_env(
        db_path=args.db,
        model=getattr(args, "model", None),
        work_dir=Path(args.work_dir) if getattr(args, "work_dir", None) else None,
        row_limit=getattr(args, "row_limit", None),
        query_timeout_s=getattr(args, "timeout", None),
        use_cache=False if getattr(args, "no_cache", False) else None,
        prompt_version=prompt,
        # 复现 v0.2.0 基线时要关掉 v0.3.0 引入的强制反思，否则不是同一个实验条件
        force_reflect=None if prompt is None else prompt != "v0.2.0",
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", help="SQLite 数据库路径（默认读 DATAAGENT_DB）")
    parser.add_argument("--work-dir", help="trace 与缓存目录，默认 .dataagent")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataagent", description="带护栏的自然语言查数 Agent"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_schema = sub.add_parser("schema", help="打印给 LLM 看的 schema 文档")
    _add_common(p_schema)

    p_sql = sub.add_parser("sql", help="直接执行一条 SQL（不走 LLM，验证护栏用）")
    _add_common(p_sql)
    p_sql.add_argument("sql")
    p_sql.add_argument("--row-limit", type=int, default=200)
    p_sql.add_argument("--timeout", type=float, default=15.0)

    p_ask = sub.add_parser("ask", help="用自然语言提问")
    _add_common(p_ask)
    p_ask.add_argument("question")
    p_ask.add_argument("--model", help="模型名，默认 glm-4-flash")
    p_ask.add_argument("--prompt", help="提示词版本，默认 v0.3.0（可用 v0.2.0 复现基线）")
    p_ask.add_argument("--max-steps", type=int, default=None)
    p_ask.add_argument("--row-limit", type=int, default=None)
    p_ask.add_argument("--timeout", type=float, default=None)
    p_ask.add_argument("--no-cache", action="store_true", help="禁用 LLM 缓存")
    p_ask.add_argument("-v", "--verbose", action="store_true", help="打印每一步")

    return parser


def main(argv: list[str] | None = None) -> int:
    _setup_console()
    args = build_parser().parse_args(argv)
    config = _config(args)

    from dataagent.config import is_demo_db

    if is_demo_db(config.db_path):
        print(
            "提示：正在使用仓库自带的演示库（5,108 订单 / 7 MB），"
            "数字与完整数据集（99,441 订单）不同；跑 60 题评测请用 --db 指向完整库。",
            file=sys.stderr,
        )

    if args.cmd == "schema":
        from dataagent.sandbox import Sandbox
        from dataagent.schema import build_schema_text

        sandbox = Sandbox(config.db_path)
        print(build_schema_text(sandbox.connect()))
        sandbox.close()
        return 0

    if args.cmd == "sql":
        from dataagent.sandbox import Sandbox

        sandbox = Sandbox(
            config.db_path, row_limit=args.row_limit, timeout_s=args.timeout
        )
        result = sandbox.run(args.sql)
        if result.ok:
            print(result.to_table(max_rows=args.row_limit))
            print(f"（{len(result.rows)} 行 / {result.elapsed_ms:.0f}ms）")
        else:
            print(f"[{result.error_kind}] {result.error}")
            print(f"（{result.elapsed_ms:.0f}ms）")
        sandbox.close()
        return 0 if result.ok else 1

    # ask
    from dataagent.agent import DataAgent

    if not config.api_key:
        print(
            "缺少 API key：复制 .env.example 为 .env 并填 DATAAGENT_API_KEY",
            file=sys.stderr,
        )
        return 2

    agent = DataAgent(config, verbose=args.verbose)
    try:
        if args.max_steps:
            agent.config.max_steps = args.max_steps
        result = agent.run(args.question)
    finally:
        agent.close()

    print()
    print(result.answer or "（模型没有给出回答）")
    print()
    print(
        f"--- steps={result.steps}  tokens={result.tokens} "
        f"(in {result.usage.prompt_tokens}/out {result.usage.completion_tokens})  "
        f"cost=¥{result.usage.cost:.4f}  run_id={result.run_id}"
    )
    if result.error_kinds:
        print(f"--- 失败标记: {', '.join(result.error_kinds)}")
    if result.unsourced:
        print(
            f"--- 未溯源数字: {'、'.join(result.unsourced)}"
            "（答案里出现但查询结果中没有出处，可信度自行判断）"
        )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
