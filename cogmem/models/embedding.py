from __future__ import annotations

from typing import List

from ..config import CogMemConfig
from .llm_client import create_embedding_client, LLMClient


class EmbeddingEncoder:
    """嵌入编码器
    
    支持根据配置使用独立的 embedding provider，允许 Chat 和 Embedding
    使用不同的模型服务（例如 Chat 使用 Ollama，Embedding 使用 SiliconFlow）。
    """

    def __init__(self, config: CogMemConfig) -> None:
        self._cfg = config
        # 使用 create_embedding_client 以支持独立的 embedding_provider 配置
        self._client: LLMClient = create_embedding_client(config)

    def encode(self, texts: List[str]) -> List[List[float]]:
        """将文本编码为向量
        
        Args:
            texts: 待编码的文本列表
            
        Returns:
            向量列表，每个向量是一个浮点数列表
        """
        if not texts:
            return []
        
        # 根据 embedding_provider 选择正确的模型
        embedding_provider = self._cfg.model.embedding_provider
        if embedding_provider is None:
            # 使用主 provider 的模型配置
            model = self._cfg.model.embedding_model
        elif embedding_provider.lower() == "ollama":
            model = self._cfg.model.ollama_embedding_model or self._cfg.model.embedding_model
        elif embedding_provider.lower() == "siliconflow":
            model = self._cfg.model.siliconflow_embedding_model or self._cfg.model.embedding_model
        elif embedding_provider.lower() == "modelscope":
            model = self._cfg.model.modelscope_embedding_model or self._cfg.model.embedding_model
        elif embedding_provider.lower() == "bailian":
            model = self._cfg.model.embedding_model
        else:
            model = self._cfg.model.embedding_model
            
        return self._client.embed(texts, model=model)
    
    def encode_single(self, text: str) -> List[float]:
        """将单条文本编码为向量
        
        Args:
            text: 待编码的文本
            
        Returns:
            向量（浮点数列表）
        """
        result = self.encode([text])
        return result[0] if result else []
