from __future__ import annotations

import re
from typing import Iterable, List


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub("\n{3,}", "\n\n", text)
    return text.strip()


def split_by_length(text: str, max_len: int = 800) -> List[str]:
    text = normalize_text(text)
    if len(text) <= max_len:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_len)
        chunks.append(text[start:end])
        start = end
    return chunks
