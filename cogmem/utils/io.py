from __future__ import annotations

import os
from typing import Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_paths(base_dir: str, graph_file: str, faiss_index_file: str, entity_store_file: str) -> Tuple[str, str, str]:
    ensure_dir(base_dir)
    graph_path = os.path.join(base_dir, graph_file)
    faiss_path = os.path.join(base_dir, faiss_index_file)
    entities_path = os.path.join(base_dir, entity_store_file)
    return graph_path, faiss_path, entities_path
