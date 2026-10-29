from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import List, Tuple

import faiss  # type: ignore
import numpy as np
from .vector_store import validate_vectors, validate_snapshot

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
        if not chunks and not vectors:
            return
        # Validate the full batch before storing any source or vector.
        arr = validate_vectors([c.id for c in chunks], vectors, self._dim)
        known = {c.id: c for c in self._chunks}
        new_chunks, positions = [], []
        for i, chunk in enumerate(chunks):
            if chunk.id in known:
                if known[chunk.id] != chunk:
                    raise ValueError('A chunk address cannot identify different original source content')
                continue
            known[chunk.id] = chunk
            new_chunks.append(replace(chunk))
            positions.append(i)
        if not new_chunks:
            return
        arr = arr[positions]
        if self._metric == "cosine":
            arr = self._normalize(arr)
        self._index.add(arr)
        self._chunks.extend(new_chunks)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Resolve an exact source address without a corpus search or mutation."""
        for chunk in self._chunks:
            if chunk.id == chunk_id:
                return replace(chunk)
        return None

    def search(self, query_vec: List[float], top_k: int = 10) -> List[Tuple[Chunk, float]]:
        if top_k <= 0:
            return []
        q = validate_vectors(["query"], [query_vec], self._dim)
        if self._metric == "cosine":
            q = self._normalize(q)
        D, I = self._index.search(q, top_k)
        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    def search_in_scope(self, query_vec: List[float], allowed_texts: set[str], top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """Rank the permitted sources themselves, before applying the read limit."""
        if top_k <= 0 or not allowed_texts:
            return []
        positions = [i for i, chunk in enumerate(self._chunks) if chunk.text in allowed_texts]
        if {self._chunks[i].text for i in positions} != allowed_texts:
            raise ValueError('Some permitted sources are missing from the chunk index')
        query = np.asarray([query_vec], dtype=np.float32)
        if query.shape != (1, self._dim) or not np.isfinite(query).all():
            raise ValueError('Invalid query vector')
        if self._metric == 'cosine':
            query = self._normalize(query)
        vectors = np.asarray([self._index.reconstruct(i) for i in positions])
        scores = vectors @ query[0] if self._metric == 'cosine' else ((vectors-query[0])**2).sum(axis=1)
        ranked = sorted(range(len(positions)), key=lambda i: (
            -float(scores[i]) if self._metric == 'cosine' else float(scores[i]), positions[i]))
        return [(self._chunks[positions[i]], float(scores[i])) for i in ranked[:top_k]]

    def save(self, index_path: str, chunks_path: str) -> None:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self._index, index_path)
        # 保存chunk元数据
        import pickle
        with open(chunks_path, "wb") as f:
            pickle.dump(self._chunks, f)

    def load(self, index_path: str, chunks_path: str) -> None:
        present = (os.path.exists(index_path), os.path.exists(chunks_path))
        if any(present) and not all(present):
            raise ValueError('Incomplete chunk snapshot: index and original metadata are both required')
        if not any(present):
            empty = ChunkStore(self._dim, self._metric)
            self._index, self._chunks = empty._index, []
            return
        import pickle
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        if not isinstance(chunks, list) or any(not isinstance(c, Chunk) for c in chunks):
            raise ValueError('Invalid original chunk metadata')
        validate_snapshot(index, [c.id for c in chunks], self._dim, self._metric)
        self._index, self._chunks = index, chunks

    def size(self) -> int:
        return len(self._chunks)
