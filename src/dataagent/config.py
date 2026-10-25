"""配置与运行期参数。

原则：所有可能影响 LLM 输出或成本的东西都必须显式可配置，
并且 prompt_version 要参与缓存 key（否则改提示词后只会读回旧缓存）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dataagent import PROMPT_VERSION

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL = "glm-4-flash"

# 每 1K token 单价（人民币元）。免费模型记 0。
# 只用于横向对照，不追求与厂商账单精确一致。
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "glm-4-flash": (0.0, 0.0),
    "glm-4-flash-250414": (0.0, 0.0),
    "glm-4-air": (0.0005, 0.0005),
    "glm-4-plus": (0.005, 0.005),
    "deepseek-chat": (0.001, 0.002),
}


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
    work_dir: Path = field(default_factory=lambda: Path(".dataagent"))
    prompt_version: str = PROMPT_VERSION
    use_cache: bool = True

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
        resolved_db = db_path or os.environ.get("DATAAGENT_DB") or ""
        if not resolved_db:
            raise ValueError(
                "缺少数据库路径：用 --db 指定，或设置 DATAAGENT_DB / .env 中的 DATAAGENT_DB"
            )
        params = {
            "db_path": Path(resolved_db).expanduser().resolve(),
            "api_key": os.environ.get("DATAAGENT_API_KEY", ""),
            "base_url": os.environ.get("DATAAGENT_BASE_URL", DEFAULT_BASE_URL),
            "model": os.environ.get("DATAAGENT_MODEL", DEFAULT_MODEL),
        }
        params.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**params)
