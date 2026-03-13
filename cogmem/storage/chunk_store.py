from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import faiss  # type: ignore
import numpy as np

@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    turn_id: str | None = None

class ChunkStore:
    """文档片段的向量索引，用于直接检索包含答案的原文片段"""
    
    def __init__(self, dim: int, metric: str = "cosine") -> None:
        self._dim = dim
        self._metric = metric
        if metric == "cosine":
            self._index = faiss.IndexFlatIP(dim)
        else:
            self._index = faiss.IndexFlatL2(dim)
        self._chunks: List[Chunk] = []

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms

    def add(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        """添加chunks，自动去重（基于chunk.id）"""
        if not chunks or not vectors:
            return
        
        # 去重：只添加新chunk（基于ID）
        existing_chunk_ids = {c.id for c in self._chunks}
        new_chunks: List[Chunk] = []
        new_vectors: List[List[float]] = []
        
        for chunk, vec in zip(chunks, vectors):
            if chunk.id not in existing_chunk_ids:
                new_chunks.append(chunk)
                new_vectors.append(vec)
        
        if not new_chunks:
            return  # 所有chunk都已存在
        
        arr = np.asarray(new_vectors, dtype=np.float32)
        if self._metric == "cosine":
            arr = self._normalize(arr)
        self._index.add(arr)
        self._chunks.extend(new_chunks)

    def search(self, query_vec: List[float], top_k: int = 10) -> List[Tuple[Chunk, float]]:
        q = np.asarray([query_vec], dtype=np.float32)
        if self._metric == "cosine":
            q = self._normalize(q)
        D, I = self._index.search(q, top_k)
        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    def save(self, index_path: str, chunks_path: str) -> None:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self._index, index_path)
        # 保存chunk元数据
        import pickle
        with open(chunks_path, "wb") as f:
            pickle.dump(self._chunks, f)

    def load(self, index_path: str, chunks_path: str) -> None:
        if os.path.exists(index_path):
            self._index = faiss.read_index(index_path)
        if os.path.exists(chunks_path):
            import pickle
            with open(chunks_path, "rb") as f:
                self._chunks = pickle.load(f)

    def size(self) -> int:
        return len(self._chunks)
