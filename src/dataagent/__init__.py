"""DataAgent: 带护栏的自然语言查数 Agent。

设计主张：不是让 LLM 写 SQL，而是让它在写错的时候知道自己是错的。
"""

__version__ = "0.1.0"
# 改提示词必须同步升版本号：缓存 key 与 trace 统计都按它聚合
PROMPT_VERSION = "v0.3.1"
