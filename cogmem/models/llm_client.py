from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import CogMemConfig
from ..utils.cache import LRUCache
from ..utils.metrics import METRICS


class LLMClient:
    def chat(self, system: str, user: str, model: Optional[str] = None) -> str:
        raise NotImplementedError

    def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        raise NotImplementedError


class OpenAICompatClient(LLMClient):
    def __init__(self, config: CogMemConfig) -> None:
        from openai import OpenAI
        import httpx

        self._cfg = config
        api_base = (
            self._cfg.model.api_base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("BAILIAN_BASE_URL")
        )
        api_key = (
            self._cfg.model.api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("BAILIAN_API_KEY")
        )
        # 设置超时时间（从配置读取，默认60秒）
        timeout = self._cfg.runtime.request_timeout_s
        http_client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0)  # 连接超时10秒，总超时使用配置值
        )
        self._client = OpenAI(base_url=api_base, api_key=api_key, http_client=http_client)
        self._chat_cache: LRUCache[str] = LRUCache(256)
        self._embed_cache: LRUCache[Tuple[str, Tuple[str, ...]]] = LRUCache(128)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
    def chat(self, system: str, user: str, model: Optional[str] = None) -> str:
        mdl = model or self._cfg.model.chat_model
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
            temperature=0.2,
        )
        content = resp.choices[0].message.content or ""
        self._chat_cache.set(key, content)
        latency_ms = (time.time() - start) * 1000.0
        # 记录usage
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
        mdl = model or self._cfg.model.embedding_model
        key = (mdl, tuple(texts))
        cached = self._embed_cache.get(key)
        if cached is not None:
            METRICS.record_embed(latency_ms=0.0, cache_hit=True)
            return cached  # type: ignore[return-value]
        start = time.time()
        # 百炼/OpenAI 兼容 API 限制每批最多 10 条
        max_batch_size = 10
        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), max_batch_size):
            batch = texts[i : i + max_batch_size]
            resp = self._client.embeddings.create(model=mdl, input=batch)
            all_vectors.extend([d.embedding for d in resp.data])
        vectors = all_vectors
        self._embed_cache.set(key, vectors)  # type: ignore[arg-type]
        latency_ms = (time.time() - start) * 1000.0
        # 记录最后一次调用的 usage（如果有）
        try:
            METRICS.record_embed(latency_ms=latency_ms)
        except Exception:
            pass
        return vectors


class OllamaClient(LLMClient):
    def __init__(self, config: CogMemConfig) -> None:
        self._cfg = config
        self._base = (
            self._cfg.model.ollama_base_url
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        )
        self._chat_cache: LRUCache[str] = LRUCache(256)
        self._embed_cache: LRUCache[Tuple[str, Tuple[str, ...]]] = LRUCache(128)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
    def chat(self, system: str, user: str, model: Optional[str] = None) -> str:
        mdl = model or self._cfg.model.ollama_chat_model or self._cfg.model.chat_model
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
            "options": {"temperature": 0.2},
        }
        start = time.time()
        r = requests.post(f"{self._base}/v1/chat/completions", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        self._chat_cache.set(key, content)
        latency_ms = (time.time() - start) * 1000.0
        # Ollama 通常无 usage 信息
        METRICS.record_chat(latency_ms=latency_ms)
        return content

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
    def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        mdl = model or self._cfg.model.ollama_embedding_model or self._cfg.model.embedding_model
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
    if config.model.provider == "ollama":
        return OllamaClient(config)
    return OpenAICompatClient(config)
