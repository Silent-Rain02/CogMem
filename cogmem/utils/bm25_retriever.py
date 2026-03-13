"""BM25 检索工具

用于实体和 Topic 的精确匹配检索，基于词频的 BM25 算法通常比向量相似度更适合短文本匹配。
"""

from typing import List, Tuple, Dict, Optional
import re
from collections import Counter

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("[BM25Retriever] Warning: rank-bm25 not installed, BM25 retrieval will be disabled")

class BM25Retriever:
    """BM25 检索器：用于实体和 Topic 的精确匹配"""
    
    def __init__(self, corpus: List[str], tokenize_fn=None):
        """初始化 BM25 检索器
        
        Args:
            corpus: 文档列表（每个文档是一个字符串）
            tokenize_fn: 可选的tokenize函数，默认使用简单分词
        """
        if not BM25_AVAILABLE:
            raise ImportError("rank-bm25 is required. Install with: pip install rank-bm25")
        
        self.corpus = corpus
        self.tokenize_fn = tokenize_fn or self._default_tokenize
        
        # 构建 BM25 索引
        tokenized_corpus = [self.tokenize_fn(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def _default_tokenize(self, text: str) -> List[str]:
        """默认分词：简单的小写分词"""
        if not text:
            return []
        # 转换为小写，分割单词
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """搜索最相关的文档
        
        Args:
            query: 查询字符串
            top_k: 返回的文档数量
            
        Returns:
            List of (doc_index, score) tuples, sorted by score descending
        """
        if not query:
            return []
        
        tokenized_query = self.tokenize_fn(query)
        if not tokenized_query:
            return []
        
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取 top-k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]
    
    def get_top_docs(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """获取最相关的文档文本
        
        Args:
            query: 查询字符串
            top_k: 返回的文档数量
            
        Returns:
            List of (doc_text, score) tuples, sorted by score descending
        """
        results = self.search(query, top_k)
        return [(self.corpus[idx], score) for idx, score in results]

class EntityBM25Matcher:
    """基于 BM25 的实体匹配器"""
    
    def __init__(self, entities: List[Dict], entity_text_fn=None):
        """初始化实体 BM25 匹配器
        
        Args:
            entities: 实体列表，每个实体是一个字典，包含 id, name 等字段
            entity_text_fn: 从实体字典中提取文本的函数
        """
        if not BM25_AVAILABLE:
            self.available = False
            return
        
        self.entities = entities
        self.entity_text_fn = entity_text_fn or self._default_entity_text
        
        # 构建实体文本列表
        entity_texts = [self.entity_text_fn(entity) for entity in entities]
        
        # 创建 BM25 检索器
        self.bm25_retriever = BM25Retriever(entity_texts)
        self.available = True
    
    def _default_entity_text(self, entity: Dict) -> str:
        """默认实体文本提取：name + aliases + description"""
        parts = []
        
        if 'name' in entity:
            parts.append(str(entity['name']))
        if 'canonical_name' in entity:
            parts.append(str(entity['canonical_name']))
        if 'aliases' in entity and entity['aliases']:
            if isinstance(entity['aliases'], list):
                parts.extend([str(alias) for alias in entity['aliases']])
            else:
                parts.append(str(entity['aliases']))
        if 'description' in entity:
            parts.append(str(entity['description']))
        
        return ' '.join(parts)
    
    def match(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """匹配实体
        
        Args:
            query: 查询字符串
            top_k: 返回的实体数量
            
        Returns:
            List of (entity_dict, score) tuples, sorted by score descending
        """
        if not self.available:
            return []
        
        results = self.bm25_retriever.search(query, top_k)
        return [(self.entities[idx], score) for idx, score in results]
