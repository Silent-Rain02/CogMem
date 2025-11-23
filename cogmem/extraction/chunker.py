"""文档切分模块：根据场景（对话 vs 通用）选择不同的切分策略"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..storage.schema import Doc


class ChunkerStrategy(ABC):
    """文档切分策略基类"""

    @abstractmethod
    def chunk(self, doc: Doc) -> List[str]:
        """将文档切分为多个片段
        
        Args:
            doc: 输入文档
            
        Returns:
            切分后的文本片段列表
        """
        pass


class DialogChunker(ChunkerStrategy):
    """对话场景切分器：保持轮次完整性
    
    特点：
    - 每个轮次作为一个独立 chunk
    - 保留轮次元数据（speaker, turn_id, timestamp）
    - 支持多轮合并（可配置，如每 2-3 轮合并）
    - 如果单轮过长，按句子切分但保持在同一轮次内
    """

    def __init__(self, turns_per_chunk: int = 1, max_length: int = 1000):
        """
        Args:
            turns_per_chunk: 每个chunk包含的轮次数（默认1，但文档可能已合并多个turns）
            max_length: 单个chunk的最大长度（字符数，默认1000以支持合并后的对话）
        """
        self.turns_per_chunk = turns_per_chunk
        self.max_length = max_length

    def chunk(self, doc: Doc) -> List[str]:
        """切分对话文档"""
        chunks = []
        text = doc.text

        # 简单实现：每个轮次一个 chunk
        if len(text) <= self.max_length:
            chunks.append(text)
        else:
            # 如果单轮过长，按句子切分但保持在同一轮次内
            sentences = self._split_sentences(text)
            current_chunk = []
            current_len = 0
            for sent in sentences:
                if current_len + len(sent) > self.max_length and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sent]
                    current_len = len(sent)
                else:
                    current_chunk.append(sent)
                    current_len += len(sent)
            if current_chunk:
                chunks.append(" ".join(current_chunk))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """简单句子切分：中英文兼容"""
        # 按中英文句子结束符切分
        parts = re.split(r'([。！？.!?])', text)
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            if i + 1 < len(parts):
                sentences.append(parts[i] + parts[i + 1])
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1])
        return [s for s in sentences if s.strip()]


class GeneralChunker(ChunkerStrategy):
    """通用场景切分器：语义完整性 + 重叠
    
    特点：
    - 按句子边界切分
    - 添加 overlap（默认 50 字符）
    - 保持语义完整性
    """

    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        """
        Args:
            chunk_size: 目标chunk大小（字符数）
            overlap: 相邻chunk之间的重叠部分（字符数）
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Doc) -> List[str]:
        """切分通用文档"""
        text = doc.text
        sentences = self._split_sentences(text)

        chunks = []
        current_chunk = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > self.chunk_size and current_chunk:
                # 保存当前 chunk
                chunks.append("".join(current_chunk))

                # 计算 overlap：保留最后几个句子
                overlap_sents = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= self.overlap:
                        overlap_sents.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break

                # 开始新 chunk（带 overlap）
                current_chunk = overlap_sents + [sent]
                current_len = overlap_len + sent_len
            else:
                current_chunk.append(sent)
                current_len += sent_len

        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """中英文句子切分"""
        # 中英文句子切分：匹配句子及其结束符
        pattern = r'([^。！？.!?\n]+[。！？.!?\n]?)'
        sentences = re.findall(pattern, text)
        if not sentences:
            sentences = [text]
        return [s.strip() for s in sentences if s.strip()]


class ChunkerFactory:
    """切分器工厂：根据数据类型自动选择策略"""

    @staticmethod
    def create(doc: Doc, config=None) -> ChunkerStrategy:
        """创建切分器
        
        Args:
            doc: 文档对象
            config: 配置对象（可选）
            
        Returns:
            对应的切分策略实例
        """
        # 根据 doc.id 或 doc.metadata 判断是否为对话场景
        # 对话场景的标识：doc.id以"turn-"或"dialog-"开头，或包含"dialog"关键词
        if doc.id.startswith("turn-") or doc.id.startswith("dialog-") or "dialog" in doc.id.lower():
            # 对话场景
            if config and hasattr(config, 'chunk'):
                return DialogChunker(
                    turns_per_chunk=config.chunk.dialog_turns_per_chunk,
                    max_length=config.chunk.dialog_max_length
                )
            return DialogChunker()
        else:
            # 通用场景
            if config and hasattr(config, 'chunk'):
                return GeneralChunker(
                    chunk_size=config.chunk.general_chunk_size,
                    overlap=config.chunk.general_overlap
                )
            return GeneralChunker()

