from __future__ import annotations

import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass
class Counter:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0


class MetricsTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chat = Counter()
        self._embed = Counter()

    def record_chat(self, latency_ms: float, prompt_tokens: Optional[int] = None, completion_tokens: Optional[int] = None, total_tokens: Optional[int] = None, cache_hit: bool = False) -> None:
        with self._lock:
            self._chat.calls += 1
            if cache_hit:
                self._chat.cache_hits += 1
            if prompt_tokens is not None:
                self._chat.prompt_tokens += int(prompt_tokens)
            if completion_tokens is not None:
                self._chat.completion_tokens += int(completion_tokens)
            if total_tokens is not None:
                self._chat.total_tokens += int(total_tokens)
            self._chat.total_latency_ms += float(latency_ms)

    def record_embed(self, latency_ms: float, prompt_tokens: Optional[int] = None, total_tokens: Optional[int] = None, cache_hit: bool = False) -> None:
        with self._lock:
            self._embed.calls += 1
            if cache_hit:
                self._embed.cache_hits += 1
            if prompt_tokens is not None:
                self._embed.prompt_tokens += int(prompt_tokens)
            if total_tokens is not None:
                self._embed.total_tokens += int(total_tokens)
            self._embed.total_latency_ms += float(latency_ms)

    def snapshot(self) -> Dict:
        with self._lock:
            return {
                "chat": asdict(self._chat),
                "embed": asdict(self._embed),
            }

    def reset(self) -> None:
        with self._lock:
            self._chat = Counter()
            self._embed = Counter()


# 全局单实例
METRICS = MetricsTracker()
