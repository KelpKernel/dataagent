"""配置与运行期参数。

原则：所有可能影响 LLM 输出或成本的东西都必须显式可配置，
并且 prompt_version 要参与缓存 key（否则改提示词后只会读回旧缓存）。
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from litmus import PROMPT_VERSION

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL = "glm-4-flash"

# 仓库里自带的演示库（5.1k 订单 / 7 MB）。没配 LITMUS_DB 时用它，
# 这样 clone 下来不填任何东西也能跑通。
DEMO_DB = Path(__file__).resolve().parents[2] / "data" / "demo.sqlite"

# provider 注册表：按模型名自动选 base_url 与 key 所在环境变量。
# 做模型对照实验时只需 --model，不用手改 .env。
PROVIDERS: dict[str, dict] = {
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env": "LITMUS_API_KEY",
        "models": ("glm-4-flash", "glm-4-flash-250414", "glm-4-air", "glm-4-plus"),
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "models": ("deepseek-chat", "deepseek-reasoner", "deepseek-flash"),
    },
}

# 每 1K token 单价（人民币元）。免费模型记 0。
# 只用于横向对照，不追求与厂商账单精确一致。
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "glm-4-flash": (0.0, 0.0),
    "glm-4-flash-250414": (0.0, 0.0),
    "glm-4-air": (0.0005, 0.0005),
    "glm-4-plus": (0.005, 0.005),
    # DeepSeek 按缓存未命中价估算，实际账单可能更低（命中前缀缓存时）
    "deepseek-chat": (0.002, 0.008),
    "deepseek-reasoner": (0.002, 0.008),
    "deepseek-flash": (0.002, 0.008),
}


def is_demo_db(db_path: str | Path) -> bool:
    """判断是否为仓库自带的演示库。

    演示库带一张 `_meta` 表（记录抽样规则），完整库没有。用它提示使用者
    「这里的数字和文章里的 99,441 不一样」，避免拿 demo 结果去对照 60 题评测集。
    """
    try:
        conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return False
    try:
        return conn.execute("select value from _meta where key = 'note'").fetchone() is not None
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def resolve_endpoint(model: str) -> tuple[str, str]:
    """按模型名解析 (base_url, api_key)；未注册的模型走 LITMUS_* 环境变量。

    注意：不能用 LITMUS_BASE_URL 覆盖已注册 provider 的 base_url，
    否则 .env 里为智谱写的 base_url 会把 DeepSeek 的请求也发到智谱去。
    """
    for spec in PROVIDERS.values():
        if model in spec["models"]:
            return spec["base_url"], _env(spec["api_key_env"])
    return (
        _env("LITMUS_BASE_URL", DEFAULT_BASE_URL),
        _env("LITMUS_API_KEY"),
    )


def load_dotenv(path: Path) -> None:
    """极简 .env 读取，已存在的环境变量优先，不覆盖。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# 2026-10-26 项目由 dataagent 改名为 litmus。旧环境变量名继续认，
# 新名优先：这样已经写好的 .env、CI secret、外部脚本不会一次性失效，
# 而新代码只需要知道 LITMUS_*。
_LEGACY_ENV: dict[str, str] = {
    "LITMUS_API_KEY": "DATAAGENT_API_KEY",
    "LITMUS_BASE_URL": "DATAAGENT_BASE_URL",
    "LITMUS_MODEL": "DATAAGENT_MODEL",
    "LITMUS_DB": "DATAAGENT_DB",
}


def _env(name: str, default: str = "") -> str:
    """读环境变量，找不到时回退到改名前的 DATAAGENT_* 旧名。"""
    value = os.environ.get(name)
    if value:
        return value
    legacy = _LEGACY_ENV.get(name)
    if legacy:
        return os.environ.get(legacy, default)
    return default


@dataclass(slots=True)
class Config:
    db_path: Path
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.0

    # Agent 循环预算
    max_steps: int = 8
    row_limit: int = 200
    query_timeout_s: float = 15.0

    # 产出目录：trace 与 LLM 缓存都落在这里，方便事后做统计分析
    work_dir: Path = field(default_factory=lambda: Path(".litmus"))
    prompt_version: str = PROMPT_VERSION
    use_cache: bool = True
    # v0.3.0 起的强制反思干预；复现 v0.2.0 基线时关掉
    force_reflect: bool = True

    @property
    def trace_db(self) -> Path:
        return self.work_dir / "trace.sqlite"

    @property
    def cache_dir(self) -> Path:
        return self.work_dir / "cache"

    def price(self) -> tuple[float, float]:
        return PRICE_TABLE.get(self.model, (0.0, 0.0))

    @classmethod
    def from_env(cls, db_path: str | Path | None = None, **overrides) -> "Config":
        load_dotenv(Path(".env"))
        # 优先级：--db 参数 > 环境变量/.env > 仓库自带演示库
        resolved_db = db_path or _env("LITMUS_DB") or ""
        if not resolved_db:
            if not DEMO_DB.exists():
                raise ValueError(
                    "缺少数据库路径：用 --db 指定，或设置 LITMUS_DB。"
                    f"仓库自带演示库也不存在（{DEMO_DB}）。"
                )
            resolved_db = str(DEMO_DB)
        model = overrides.get("model") or _env(
            "LITMUS_MODEL", DEFAULT_MODEL
        )
        base_url, api_key = resolve_endpoint(model)
        params = {
            "db_path": Path(resolved_db).expanduser().resolve(),
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
        }
        params.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**params)
