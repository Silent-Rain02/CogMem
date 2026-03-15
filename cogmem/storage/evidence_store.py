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
        """优化2: 增加chunk数量限制，改进chunk格式"""
        arr = self._entity_to_chunks[entity_id]
        if len(arr) < 15:  # 优化2: 从10个增加到15个，确保不会遗漏答案
            # 优化2: 不要截断chunk，或者至少保留更多内容（从1000增加到1500）
            arr.append(chunk[:1500])  # 保留更多内容，避免截断重要信息

    def get_chunks(self, entity_id: str, limit: int = 3) -> List[str]:
        """修复2: 简单的chunk检索（按顺序返回）"""
        return self._entity_to_chunks.get(entity_id, [])[:limit]
    
    def get_chunks_by_query(self, entity_id: str, query: str, limit: int = 5, embedding_encoder=None, chunk_store=None) -> List[str]:
        """Query-aware chunk retrieval: use ChunkStore.search() when available, else re-embed entity chunks."""
        chunks = self._entity_to_chunks.get(entity_id, [])
        if not chunks:
            return []
        limit = max(1, int(limit))

        # Prefer ChunkStore global search then filter by entity (avoids re-embedding all entity chunks)
        if embedding_encoder and chunk_store and chunk_store.size() > 0:
            try:
                query_embedding = embedding_encoder.encode([query])[0]
                candidate_results = chunk_store.search(query_embedding, top_k=min(max(limit * 3, limit), chunk_store.size()))
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
            except Exception:
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
            except Exception:
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
                'entity_to_chunks': dict(self._entity_to_chunks)  # 启用：保存实体到chunks的映射
            }, f)
    
    def load(self, path: str) -> None:
        """从文件加载已处理的文档ID和实体到chunks的映射"""
        if not os.path.exists(path):
            return
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self._processed_doc_ids = data.get('processed_doc_ids', set())
                # 恢复实体到chunks的映射
                if 'entity_to_chunks' in data:
                    self._entity_to_chunks = defaultdict(list, data['entity_to_chunks'])
                    print(f"[EvidenceStore] 加载了 {len(self._entity_to_chunks)} 个实体的chunks映射")
        except Exception as e:
            print(f"警告：加载EvidenceStore失败: {e}，将使用空的已处理文档集合")
            self._processed_doc_ids = set()
            self._entity_to_chunks = defaultdict(list)
