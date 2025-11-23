from __future__ import annotations

from typing import List

from ..config import CogMemConfig
from .llm_client import create_llm_client


class EmbeddingEncoder:
    def __init__(self, config: CogMemConfig) -> None:
        self._cfg = config
        self._client = create_llm_client(config)

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._client.embed(texts, model=self._cfg.model.embedding_model)
