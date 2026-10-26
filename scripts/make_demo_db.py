"""从完整的 Olist 库生成一个几 MB 的演示库，让 clone 下来的人零配置就能跑。

为什么需要它：
README 的快速开始第一步就是填 DATAAGENT_DB，但那个 198 MB 的库不在仓库里，
陌生人根本跑不起来。一个"打不开的 demo"比少一个功能致命得多。

抽样策略（按 order_id 的 hex 前缀，确定性、可复现、时间分布均匀）：
  order_id 前两位 < "0d"  →  约 5.1% 的订单
然后按关联关系收窄其他表，保证 join 不会丢行、也不会带进孤立记录。

用法：
  python scripts/make_demo_db.py --src <完整库路径>
  python scripts/make_demo_db.py --src ... --prefix-max 0a    # 更小的子集
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "demo.sqlite"

# 插入顺序 = 依赖顺序：先父表，后子表
INSERT_ORDER = [
    "orders",
    "customers",
    "order_items",
    "order_payments",
    "order_reviews",
    "products",
    "sellers",
    "geolocation",
    "category_translation",
]


def select_sql(table: str, prefix_max: str) -> str:
    """每张表的抽样语句。子表都用父表已筛过的结果收窄，保证引用完整。"""
    order_filter = f"substr(order_id, 1, 2) < '{prefix_max}'"
    statements = {
        "orders": f"select * from full.orders where {order_filter}",
        "customers": (
            "select * from full.customers "
            "where customer_id in (select customer_id from orders)"
        ),
        "order_items": (
            "select * from full.order_items "
            "where order_id in (select order_id from orders)"
        ),
        "order_payments": (
            "select * from full.order_payments "
            "where order_id in (select order_id from orders)"
        ),
        "order_reviews": (
            "select * from full.order_reviews "
            "where order_id in (select order_id from orders)"
        ),
        "products": (
            "select * from full.products "
            "where product_id in (select product_id from order_items)"
        ),
        "sellers": (
            "select * from full.sellers "
            "where seller_id in (select seller_id from order_items)"
        ),
        # 地理表 100 万行：只保留客户与卖家实际用到的邮编，
        # 且每个邮编只留一条坐标（同一邮编有多个采样点，demo 不需要精度）
        "geolocation": (
            "select * from full.geolocation where rowid in ("
            "  select min(rowid) from full.geolocation "
            "  where geolocation_zip_code_prefix in ("
            "    select customer_zip_code_prefix from customers "
            "    where customer_zip_code_prefix is not null "
            "    union "
            "    select seller_zip_code_prefix from sellers "
            "    where seller_zip_code_prefix is not null)"
            "  group by geolocation_zip_code_prefix)"
        ),
        # 类目翻译表只有 71 行，全量保留
        "category_translation": "select * from full.category_translation",
    }
    return statements[table]


def main() -> int:
    parser = argparse.ArgumentParser(description="生成演示用的小型数据库")
    parser.add_argument("--src", required=True, help="完整 Olist 库路径")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出路径")
    parser.add_argument(
        "--prefix-max",
        default="0d",
        help="order_id 前两位 hex 的上限，0d≈5%%，0a≈3.9%%，1f≈12%%",
    )
    args = parser.parse_args()

    src = Path(args.src).resolve()
    if not src.exists():
        print(f"源库不存在：{src}")
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    conn = sqlite3.connect(out, uri=True)
    conn.execute("ATTACH DATABASE ? AS full", (f"file:{src.as_posix()}?mode=ro",))

    # 1) 原样复制建表与建索引语句（保留 DDL 里的中文列注释）
    objects = conn.execute(
        "select type, name, sql from full.sqlite_master "
        "where sql is not null and type in ('table', 'index') "
        "and name not like 'sqlite_%'"
    ).fetchall()
    for _, _, sql in objects:
        conn.execute(sql)

    # 2) 按依赖顺序灌入抽样数据
    counts: dict[str, int] = {}
    for table in INSERT_ORDER:
        conn.execute(f'insert into "{table}" {select_sql(table, args.prefix_max)}')
        counts[table] = conn.execute(f'select count(*) from "{table}"').fetchone()[0]

    # 3) 元信息：让使用方知道这不是完整数据
    conn.execute("create table _meta(key text primary key, value text not null)")
    meta = {
        "source": "Olist Brazilian E-Commerce public dataset",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample_rule": f"orders where substr(order_id,1,2) < '{args.prefix_max}'",
        "orders": str(counts.get("orders", 0)),
        "note": (
            "This is a SAMPLE database for demo purposes. Row counts and any aggregate "
            "differ from the full dataset (99,441 orders). Do not use it for benchmarking."
        ),
    }
    conn.executemany("insert into _meta values (?, ?)", meta.items())
    conn.commit()

    conn.execute("DETACH DATABASE full")
    conn.execute("VACUUM")
    conn.close()

    size_mb = out.stat().st_size / 1e6
    print(f"已生成：{out}")
    print(f"体积：{size_mb:.2f} MB")
    print("各表行数：")
    for table, rows in counts.items():
        print(f"  {table:<24}{rows:>8,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
