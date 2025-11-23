"""生成层：RAG生成（兼容层，实际使用Agentic系统）"""

from .rag import generate_answer, normalize_answer

__all__ = [
    "generate_answer",
    "normalize_answer",
]
