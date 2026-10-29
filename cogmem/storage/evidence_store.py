from __future__ import annotations
from cogmem.models.execution_errors import raise_if_execution_failure

import os
import pickle
import hashlib
import re
from typing import Dict, List, DefaultDict
from collections import defaultdict

class EvidenceStore:
    def __init__(self) -> None:
        self._entity_to_chunks: DefaultDict[str, List[str]] = defaultdict(list)
        self._processed_doc_ids: set[str] = set()
        self._document_hashes: Dict[str, str] = {}

    def add_chunk_for_entity(self, entity_id: str, chunk: str) -> None:
        """Retain complete source chunks; retrieval limits apply only when reading."""
        self._entity_to_chunks[entity_id].append(chunk)

    def get_chunks(self, entity_id: str, limit: int | None = 3) -> List[str]:
        """修复2: 简单的chunk检索（按顺序返回）"""
        return list(self._entity_to_chunks.get(entity_id, []))[:limit]

    def get_chunks_by_query(self, entity_id: str, query: str, limit: int = 5, embedding_encoder=None, chunk_store=None) -> List[str]:
        """Rank only the original sources referenced by this node."""
        return self.rank_chunks(self.get_chunks(entity_id,limit=None),query,limit,embedding_encoder,chunk_store)

    def rank_chunks(self, chunks: List[str], query: str, limit: int = 5, embedding_encoder=None, chunk_store=None) -> List[str]:
        """Read-only ranking of an explicitly resolved provenance scope."""
        chunks = list(dict.fromkeys(chunks))
        if not chunks:
            return []
        limit = max(1, int(limit))

        # Scope must be applied before truncation: a global shortlist can starve
        # a permitted source when unrelated documents rank above it.
        if embedding_encoder and chunk_store and chunk_store.size() > 0:
            try:
                query_embedding = embedding_encoder.encode([query])[0]
                candidate_results = chunk_store.search_in_scope(query_embedding, set(chunks), top_k=limit)
                entity_text_set = set(chunks)
                sorted_chunks = []
                for chunk, score in candidate_results:
                    text = getattr(chunk, 'text', None) or (chunk if isinstance(chunk, str) else '')
                    if text in entity_text_set:
                        sorted_chunks.append((text, score))
                        if len(sorted_chunks) >= limit:
                            break
                if sorted_chunks:
                    return [t for t, _ in sorted_chunks]
            except Exception as execution_error:
                raise_if_execution_failure(execution_error)
                pass

        # Fallback: embed entity chunks and score by similarity
        if embedding_encoder:
            try:
                query_embedding = embedding_encoder.encode([query])[0]
                chunk_embeddings = embedding_encoder.encode(chunks)
                import numpy as np
                query_vec = np.array(query_embedding)
                similarities = []
                for chunk_emb in chunk_embeddings:
                    chunk_vec = np.array(chunk_emb)
                    similarity = np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec) + 1e-12)
                    similarities.append(similarity)
                min_sim = 0.2
                scored = [(chunks[i], similarities[i]) for i in range(len(chunks)) if similarities[i] >= min_sim]
                scored.sort(key=lambda x: x[1], reverse=True)
                if len(scored) < limit:
                    scored = [(chunks[i], similarities[i]) for i in range(len(chunks))]
                    scored.sort(key=lambda x: x[1], reverse=True)
                return [t for t, _ in scored[:limit]]
            except Exception as execution_error:
                raise_if_execution_failure(execution_error)
                pass

        # 优化2: 如果没有向量检索，使用改进的关键词匹配
        query_lower = query.lower()
        query_words = set(query_lower.split())
        # 移除停用词
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'to', 'of', 'in', 'on', 'at', 'for', 'with', 'by', 'from', 'as', 'and', 'or', 'but', 'if', 'that', 'this', 'these', 'those', 'what', 'which', 'who', 'when', 'where', 'why', 'how'}
        query_words = query_words - stopwords

        scored_chunks = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            # 计算关键词匹配分数（改进版）
            chunk_words = set(chunk_lower.split())
            chunk_words = chunk_words - stopwords
            common_words = query_words & chunk_words

            # 优化2: 改进评分算法 - 考虑关键词频率和位置
            if not query_words:
                score = 0.0
            else:
                # 基础分数：共同词汇比例
                base_score = len(common_words) / len(query_words)
                # 奖励：如果chunk包含所有查询关键词
                if len(common_words) == len(query_words):
                    base_score *= 1.5
                score = base_score

            scored_chunks.append((score, chunk))

        # 按分数排序，只返回高相关性的chunks（分数 > 0.2）
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        filtered_chunks = [chunk for score, chunk in scored_chunks if score > 0.2]

        # 如果高相关性的chunks不够，返回所有chunks
        if len(filtered_chunks) < limit:
            filtered_chunks = [chunk for score, chunk in scored_chunks[:limit]]

        return filtered_chunks[:limit]

    def build_answer_evidence(self, entity_ids: List[str], limit_per_entity: int = 2) -> List[str]:
        ev: List[str] = []
        for eid in entity_ids:
            ev.extend(self.get_chunks(eid, limit=limit_per_entity))
        return ev[:10]

    def register_documents(self, originals):
        """Reserve immutable source identities only after validating the full batch.

        Legacy processed IDs without a raw-document fingerprint remain readable,
        but cannot certify that a newly supplied document is the same original.
        """
        hashes = {}
        for key, text in originals.items():
            digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
            previous = self._document_hashes.get(key)
            if previous is not None and previous != digest:
                raise ValueError('Source document ID already identifies different original content')
            if key in self._processed_doc_ids and previous is None:
                raise ValueError('Legacy source identity is unverifiable; rebuild or use a new source namespace')
            hashes[key] = digest
        self._document_hashes.update(hashes)

    def mark_doc_processed(self, doc_id: str) -> None:
        """标记文档已处理"""
        self._processed_doc_ids.add(doc_id)

    def get_processed_doc_ids(self) -> set[str]:
        """获取已处理的文档ID集合"""
        return self._processed_doc_ids.copy()

    def save(self, path: str) -> None:
        """保存已处理的文档ID和实体到chunks的映射到文件"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, 'wb') as f:
            # 保存已处理的文档ID和实体到chunks的映射
            pickle.dump({
                'processed_doc_ids': self._processed_doc_ids,
                'document_hashes': self._document_hashes,
                'entity_to_chunks': dict(self._entity_to_chunks)  # 启用：保存实体到chunks的映射
            }, f)

    def load(self, path: str) -> None:
        """从文件加载已处理的文档ID和实体到chunks的映射"""
        if not os.path.exists(path):
            self._processed_doc_ids = set()
            self._document_hashes = {}
            self._entity_to_chunks = defaultdict(list)
            return

        # Validation and deserialization finish before replacing live evidence.
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise ValueError('Invalid evidence snapshot')
        processed = data.get('processed_doc_ids', set())
        chunks = data.get('entity_to_chunks', {})
        hashes = data.get('document_hashes', {})
        if not isinstance(hashes, dict) or any(not isinstance(key, str) or not isinstance(value, str) or
                re.fullmatch(r'[0-9a-f]{64}', value) is None for key, value in hashes.items()):
            raise ValueError('Invalid original document fingerprints')
        if not isinstance(processed, (set, list, tuple)) or not isinstance(chunks, dict):
            raise ValueError('Invalid evidence snapshot fields')
        if any(not isinstance(key, str) for key in processed):
            raise ValueError('Invalid processed source address')
        if any(not isinstance(key, str) or not isinstance(values, list) or
               any(not isinstance(text, str) for text in values) for key, values in chunks.items()):
            raise ValueError('Invalid source references in evidence snapshot')
        self._document_hashes = dict(hashes)
        self._processed_doc_ids = set(processed)
        self._entity_to_chunks = defaultdict(list, chunks)
