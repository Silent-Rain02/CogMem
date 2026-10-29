"""Document chunking strategies for different document types."""

from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import List

from ..config import CogMemConfig
from ..storage.schema import Doc


class BaseChunker(ABC):
    """Base class for document chunkers."""
    
    def __init__(self, config: CogMemConfig):
        self._cfg = config
    
    @abstractmethod
    def chunk(self, doc: Doc) -> List[str]:
        """Chunk a document into text segments."""
        pass


class GeneralChunker(BaseChunker):
    """General purpose chunker with overlap."""
    
    def chunk(self, doc: Doc) -> List[str]:
        """Chunk text using size and overlap from config."""
        text = doc.text
        chunk_size = self._cfg.chunk.general_chunk_size
        overlap = self._cfg.chunk.general_overlap
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            
        return chunks


class DialogChunker(BaseChunker):
    """Chunker for dialog/conversation documents."""
    
    def chunk(self, doc: Doc) -> List[str]:
        """Chunk dialog by turns or using max length."""
        text = doc.text
        max_length = self._cfg.chunk.dialog_max_length
        
        # If text fits in one chunk, return as-is
        if len(text) <= max_length:
            return [text]
        
        # Dialogue delimiters are structural line prefixes, not words quoted in
        # an utterance. Preserve original offsets, whitespace and line endings.
        turn_start = re.compile(r"^[ \t]*(?:(?:User|Assistant|Human|AI):|Turn[ \t]+\d+[ \t]*:)")
        chunks = []
        start = offset = 0
        for line in text.splitlines(keepends=True):
            if turn_start.match(line) and offset - start >= max_length and offset > start:
                chunks.append(text[start:offset])
                start = offset
            offset += len(line)
        if start < len(text):
            chunks.append(text[start:])
        return chunks or [text]


class ChunkerFactory:
    """Factory for creating appropriate chunkers based on document type."""
    
    @staticmethod
    def create(doc: Doc, config: CogMemConfig) -> BaseChunker:
        """Create a chunker appropriate for the document type.
        
        Args:
            doc: Document to chunk
            config: CogMem configuration
            
        Returns:
            Appropriate chunker instance
        """
        doc_id = doc.id.lower()
        text = doc.text
        
        # Detect dialog/conversation documents
        is_dialog = (
            doc_id.startswith("dialog-") or
            doc_id.startswith("turn-") or
            doc_id.startswith("session-") or
            "user:" in text[:500].lower() or
            "assistant:" in text[:500].lower() or
            "human:" in text[:500].lower()
        )
        
        if is_dialog:
            return DialogChunker(config)
        
        return GeneralChunker(config)
