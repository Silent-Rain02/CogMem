from __future__ import annotations

import os
import pickle
from typing import Dict, List, DefaultDict
from collections import defaultdict


class EvidenceStore:
    def __init__(self) -> None:
        self._entity_to_chunks: DefaultDict[str, List[str]] = defaultdict(list)
        self._processed_doc_ids: set[str] = set()

    def add_chunk_for_entity(self, entity_id: str, chunk: str) -> None:
        arr = self._entity_to_chunks[entity_id]
        if len(arr) < 5:  # 限制每实体的片段数
            arr.append(chunk[:500])

    def get_chunks(self, entity_id: str, limit: int = 3) -> List[str]:
        return self._entity_to_chunks.get(entity_id, [])[:limit]

    def build_answer_evidence(self, entity_ids: List[str], limit_per_entity: int = 2) -> List[str]:
        ev: List[str] = []
        for eid in entity_ids:
            ev.extend(self.get_chunks(eid, limit=limit_per_entity))
        return ev[:10]
    
    def mark_doc_processed(self, doc_id: str) -> None:
        """标记文档已处理"""
        self._processed_doc_ids.add(doc_id)
    
    def get_processed_doc_ids(self) -> set[str]:
        """获取已处理的文档ID集合"""
        return self._processed_doc_ids.copy()
    
    def save(self, path: str) -> None:
        """保存已处理的文档ID到文件"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, 'wb') as f:
            # 只保存已处理的文档ID（节省空间，实体到chunks的映射通常可以重建）
            pickle.dump({
                'processed_doc_ids': self._processed_doc_ids,
                # 可选：也保存实体到chunks的映射（如果需要）
                # 'entity_to_chunks': dict(self._entity_to_chunks)
            }, f)
    
    def load(self, path: str) -> None:
        """从文件加载已处理的文档ID"""
        if not os.path.exists(path):
            return
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self._processed_doc_ids = data.get('processed_doc_ids', set())
                # 可选：恢复实体到chunks的映射
                # if 'entity_to_chunks' in data:
                #     self._entity_to_chunks = defaultdict(list, data['entity_to_chunks'])
        except Exception as e:
            print(f"警告：加载EvidenceStore失败: {e}，将使用空的已处理文档集合")
            self._processed_doc_ids = set()
