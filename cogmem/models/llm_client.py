from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import CogMemConfig
from ..utils.cache import LRUCache
from ..utils.metrics import METRICS

# Import OpenAI at module level to avoid import issues in methods
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


class LLMClient:
    """LLM 客户端基类"""

    def chat(self, system: str, user: str, model: Optional[str] = None) -> str:
        raise NotImplementedError

    def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        raise NotImplementedError

    def _get_chat_model(self, model: Optional[str] = None) -> Optional[str]:
        """获取聊天模型名称（子类可覆盖）"""
        return model

    def _get_embedding_model(self, model: Optional[str] = None) -> Optional[str]:
        """获取嵌入模型名称（子类可覆盖）"""
        return model


class OpenAICompatClient(LLMClient):
    """OpenAI 兼容 API 客户端（基础实现）"""

    def __init__(
        self,
        config: CogMemConfig,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required. "
                "Install it with: pip install openai"
            )
        
        self._cfg = config
        self._base_url = base_url or self._cfg.model.api_base_url
        self._api_key = api_key or self._cfg.model.api_key
        self._init_openai_client()
        self._chat_cache: LRUCache[str] = LRUCache(256)
        self._embed_cache: LRUCache[Tuple[str, Tuple[str, ...]]] = LRUCache(128)

    def _init_openai_client(self) -> None:
        """初始化 OpenAI 客户端（子类可覆盖）"""
        timeout = self._cfg.runtime.request_timeout_s
        http_client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0)
        )
        self._client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            http_client=http_client,
        )

    def _get_chat_model(self, model: Optional[str] = None) -> Optional[str]:
        """获取聊天模型名称"""
        return model or self._cfg.model.chat_model

    def _get_embedding_model(self, model: Optional[str] = None) -> Optional[str]:
        """获取嵌入模型名称"""
        return model or self._cfg.model.embedding_model

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
    def chat(self, system: str, user: str, model: Optional[str] = None) -> str:
        mdl = self._get_chat_model(model)
        key = f"{mdl}||{hash(system)}||{hash(user)}"
        cached = self._chat_cache.get(key)
        if cached is not None:
            METRICS.record_chat(latency_ms=0.0, cache_hit=True)
            return cached
        start = time.time()
        resp = self._client.chat.completions.create(
            model=mdl,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self._cfg.model.temperature,
        )
        content = resp.choices[0].message.content or ""
        self._chat_cache.set(key, content)
        latency_ms = (time.time() - start) * 1000.0
        usage = getattr(resp, "usage", None)
        if usage is not None:
            try:
                METRICS.record_chat(
                    latency_ms=latency_ms,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    total_tokens=getattr(usage, "total_tokens", None),
                )
            except Exception:
                METRICS.record_chat(latency_ms=latency_ms)
        else:
            METRICS.record_chat(latency_ms=latency_ms)
        return content

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
    def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        mdl = self._get_embedding_model(model)
        key = (mdl, tuple(texts))
        cached = self._embed_cache.get(key)
        if cached is not None:
            METRICS.record_embed(latency_ms=0.0, cache_hit=True)
            return cached  # type: ignore[return-value]
        start = time.time()
        # OpenAI 兼容 API 限制每批最多 10 条
        max_batch_size = 10
        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), max_batch_size):
            batch = texts[i : i + max_batch_size]
            resp = self._client.embeddings.create(model=mdl, input=batch)
            all_vectors.extend([d.embedding for d in resp.data])
        vectors = all_vectors
        self._embed_cache.set(key, vectors)  # type: ignore[arg-type]
        latency_ms = (time.time() - start) * 1000.0
        try:
            METRICS.record_embed(latency_ms=latency_ms)
        except Exception:
            pass
        return vectors


class BailianClient(OpenAICompatClient):
    """阿里云百炼客户端"""

    def __init__(self, config: CogMemConfig) -> None:
        api_base = (
            config.model.api_base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("BAILIAN_BASE_URL")
        )
        api_key = (
            config.model.api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("BAILIAN_API_KEY")
        )
        super().__init__(config, base_url=api_base, api_key=api_key)


class SiliconFlowClient(OpenAICompatClient):
    """SiliconFlow 客户端"""

    def __init__(self, config: CogMemConfig) -> None:
        api_base = config.model.siliconflow_base_url
        api_key = config.model.siliconflow_api_key or os.getenv("SILICONFLOW_API_KEY")
        super().__init__(config, base_url=api_base, api_key=api_key)

    def _get_chat_model(self, model: Optional[str] = None) -> Optional[str]:
        return model or self._cfg.model.siliconflow_chat_model or self._cfg.model.chat_model

    def _get_embedding_model(self, model: Optional[str] = None) -> Optional[str]:
        return model or self._cfg.model.siliconflow_embedding_model or self._cfg.model.embedding_model


class ModelScopeClient(OpenAICompatClient):
    """ModelScope 客户端"""

    def __init__(self, config: CogMemConfig) -> None:
        self._api_base = config.model.modelscope_base_url
        api_key = config.model.modelscope_api_key
        super().__init__(config, base_url=self._api_base, api_key=api_key)

    def _get_chat_model(self, model: Optional[str] = None) -> Optional[str]:
        return model or self._cfg.model.modelscope_chat_model or self._cfg.model.chat_model

    def _get_embedding_model(self, model: Optional[str] = None) -> Optional[str]:
        return model or self._cfg.model.modelscope_embedding_model or self._cfg.model.embedding_model


class OllamaClient(LLMClient):
    """Ollama 本地模型客户端"""

    def __init__(self, config: CogMemConfig) -> None:
        self._cfg = config
        self._base = (
            config.model.ollama_base_url
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        )
        self._chat_cache: LRUCache[str] = LRUCache(256)
        self._embed_cache: LRUCache[Tuple[str, Tuple[str, ...]]] = LRUCache(128)

    def _get_chat_model(self, model: Optional[str] = None) -> Optional[str]:
        return model or self._cfg.model.ollama_chat_model or self._cfg.model.chat_model

    def _get_embedding_model(self, model: Optional[str] = None) -> Optional[str]:
        return model or self._cfg.model.ollama_embedding_model or self._cfg.model.embedding_model

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
    def chat(self, system: str, user: str, model: Optional[str] = None) -> str:
        mdl = self._get_chat_model(model)
        key = f"{mdl}||{hash(system)}||{hash(user)}"
        cached = self._chat_cache.get(key)
        if cached is not None:
            METRICS.record_chat(latency_ms=0.0, cache_hit=True)
            return cached
        payload = {
            "model": mdl,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": self._cfg.model.temperature},
        }
        start = time.time()
        r = requests.post(f"{self._base}/v1/chat/completions", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        self._chat_cache.set(key, content)
        latency_ms = (time.time() - start) * 1000.0
        METRICS.record_chat(latency_ms=latency_ms)
        return content

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
    def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        mdl = self._get_embedding_model(model)
        key = (mdl, tuple(texts))
        cached = self._embed_cache.get(key)
        if cached is not None:
            METRICS.record_embed(latency_ms=0.0, cache_hit=True)
            return cached  # type: ignore[return-value]
        payload = {"model": mdl, "input": texts}
        start = time.time()
        r = requests.post(f"{self._base}/v1/embeddings", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        if "data" in data:
            vectors = [item["embedding"] for item in data["data"]]
        else:
            vectors = data.get("embeddings", [])
        self._embed_cache.set(key, vectors)  # type: ignore[arg-type]
        latency_ms = (time.time() - start) * 1000.0
        METRICS.record_embed(latency_ms=latency_ms)
        return vectors


def create_llm_client(config: CogMemConfig) -> LLMClient:
    """根据配置创建对应的 LLM 客户端
    
    Args:
        config: CogMem 配置对象
        
    Returns:
        对应 provider 的 LLMClient 实例
        
    Raises:
        ValueError: 如果 provider 不被支持
    """
    provider = config.model.provider.lower()
    
    if provider == "ollama":
        return OllamaClient(config)
    elif provider == "bailian":
        return BailianClient(config)
    elif provider == "siliconflow":
        return SiliconFlowClient(config)
    elif provider == "modelscope":
        return ModelScopeClient(config)
    else:
        # 默认使用 SiliconFlow
        return SiliconFlowClient(config)


def create_embedding_client(config: CogMemConfig) -> LLMClient:
    """根据配置创建独立的 Embedding 客户端
    
    如果配置了 embedding_provider，则使用独立的 provider，否则使用主 provider。
    这允许 Chat 和 Embedding 使用不同的服务。
    
    Args:
        config: CogMem 配置对象
        
    Returns:
        对应 embedding_provider 的 LLMClient 实例
    """
    embedding_provider = config.model.embedding_provider
    
    if embedding_provider is None:
        # 使用主 provider
        return create_llm_client(config)
    
    provider = embedding_provider.lower()
    
    if provider == "ollama":
        return OllamaClient(config)
    elif provider == "siliconflow":
        return SiliconFlowClient(config)
    elif provider == "modelscope":
        return ModelScopeClient(config)
    elif provider == "bailian":
        return BailianClient(config)
    else:
        # 对于未知的 embedding_provider，默认使用 SiliconFlow
        return SiliconFlowClient(config)
