from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import faiss  # type: ignore
import numpy as np

def validate_snapshot(index, addresses, dim, metric):
    """Validate a complete derived projection before it can replace live state."""
    expected_metric = faiss.METRIC_INNER_PRODUCT if metric == 'cosine' else faiss.METRIC_L2
    if index.d != dim or index.metric_type != expected_metric:
        raise ValueError('Index dimension or distance metric does not match configuration')
    if index.ntotal != len(addresses) or len(set(addresses)) != len(addresses):
        raise ValueError('Index vectors and unique addresses must align one-to-one')
    if any(not isinstance(key, str) or not key for key in addresses):
        raise ValueError('Index addresses must be nonempty strings')


def validate_vectors(ids, vectors, dim):
    arr = np.asarray(vectors, dtype=np.float32)
    if len(ids) != len(vectors) or arr.shape != (len(ids), dim):
        raise ValueError("Each ID requires one vector of the configured dimension")
    if any(not isinstance(key, str) or not key for key in ids):
        raise ValueError("Vector IDs must be nonempty strings")
    if not np.isfinite(arr).all():
        raise ValueError("Vectors must be finite")
    return arr

class VectorStore:
    def __init__(self, dim: int, metric: str = "cosine") -> None:
        if metric not in {"cosine", "l2"}:
            raise ValueError("Unsupported vector metric")
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

    def _validate_vectors(self, ids, vectors):
        return validate_vectors(ids, vectors, self._dim)

    def add(self, ids: List[str], vectors: List[List[float]]) -> None:
        if not ids and not vectors:
            return
        arr = self._validate_vectors(ids, vectors)
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
        
        if top_k <= 0:
            return []
        q = self._validate_vectors(["query"], [query_vec])
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
        present = (os.path.exists(index_path), os.path.exists(id_path))
        if any(present) and not all(present):
            raise ValueError('Incomplete vector snapshot: index and ID file are both required')
        if not any(present):
            empty = VectorStore(self._dim, self._metric)
            self._index, self._ids = empty._index, []
            return
        loaded_index = faiss.read_index(index_path)
        with open(id_path, "rb") as f:
            loaded_ids = list(np.load(f, allow_pickle=True))
        validate_snapshot(loaded_index, loaded_ids, self._dim, self._metric)
        self._index, self._ids = loaded_index, loaded_ids

    def size(self) -> int:
        return len(self._ids)
