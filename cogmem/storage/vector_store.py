from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import faiss  # type: ignore
import numpy as np

class VectorStore:
    def __init__(self, dim: int, metric: str = "cosine") -> None:
        self._dim = dim
        self._metric = metric
        if metric == "cosine":
            self._index = faiss.IndexFlatIP(dim)
        else:
            self._index = faiss.IndexFlatL2(dim)
        self._ids: List[str] = []

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms

    def add(self, ids: List[str], vectors: List[List[float]]) -> None:
        if not ids and not vectors:
            return
        arr = np.asarray(vectors, dtype=np.float32)
        if len(ids) != len(vectors) or arr.shape != (len(ids), self._dim):
            raise ValueError("Each ID requires one vector of the configured dimension")
        if not np.isfinite(arr).all():
            raise ValueError("Vectors must be finite")
        if self._metric == "cosine":
            arr = self._normalize(arr)
        # An ID names one current derived projection, not multiple ANN entries.
        if len(set(self._ids + ids)) != len(self._ids) + len(ids):
            current = self._index.reconstruct_n(0, len(self._ids))
            latest = dict(zip(self._ids, current))
            latest.update(zip(ids, arr))
            self._ids = list(latest)
            self._index.reset()
            self._index.add(np.asarray(list(latest.values()), dtype=np.float32))
        else:
            self._index.add(arr)
            self._ids.extend(ids)

    def get_vector(self, node_id: str):
        if node_id not in self._ids:
            return None
        return self._index.reconstruct(self._ids.index(node_id)).tolist()

    def remove(self, node_id: str) -> None:
        positions = [i for i, value in enumerate(self._ids) if value == node_id]
        if positions:
            self._index.remove_ids(np.asarray(positions, dtype=np.int64))
            self._ids = [value for value in self._ids if value != node_id]

    def search(self, query_vec: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar vectors
        
        Args:
            query_vec: Query vector (must match index dimension)
            top_k: Number of results to return
            
        Returns:
            List of (entity_id, similarity_score) tuples
        """
        # 检查维度是否匹配
        query_dim = len(query_vec)
        if query_dim != self._dim:
            raise ValueError(
                f"Query vector dimension mismatch: query has {query_dim} dimensions, "
                f"but index expects {self._dim} dimensions. "
                f"This usually means the embedding model changed. "
                f"Please rebuild the index with the current embedding model."
            )
        
        q = np.asarray([query_vec], dtype=np.float32)
        if self._metric == "cosine":
            q = self._normalize(q)
        D, I = self._index.search(q, top_k)
        results: List[Tuple[str, float]] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self._ids):
                continue
            results.append((self._ids[idx], float(score)))
        return results

    def save(self, index_path: str, id_path: str) -> None:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self._index, index_path)
        with open(id_path, "wb") as f:
            np.save(f, np.array(self._ids, dtype=object))

    def load(self, index_path: str, id_path: str) -> None:
        """Load FAISS index and entity IDs from disk
        
        Args:
            index_path: Path to FAISS index file
            id_path: Path to entity IDs file
            
        Raises:
            ValueError: If loaded index dimension doesn't match VectorStore dimension.
            This is a critical error - the embedding model dimension must match the index dimension.
        """
        if os.path.exists(index_path):
            loaded_index = faiss.read_index(index_path)
            loaded_dim = loaded_index.d
            if loaded_dim != self._dim:
                raise ValueError(
                    f"CRITICAL: Index dimension mismatch!\n"
                    f"  - Loaded index has {loaded_dim} dimensions\n"
                    f"  - VectorStore expects {self._dim} dimensions (from embedding model)\n"
                    f"  - This mismatch will cause search failures!\n"
                    f"  - SOLUTION: Rebuild the index with the current embedding model, "
                    f"or switch to the embedding model that built this index."
                )
            self._index = loaded_index
        if os.path.exists(id_path):
            with open(id_path, "rb") as f:
                self._ids = list(np.load(f, allow_pickle=True))

    def size(self) -> int:
        return len(self._ids)
