"""Document chunking strategies for different document types."""

from __future__ import annotations

from abc import ABC, abstractmethod
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
        
        # Try to split by turn markers (common patterns)
        # Patterns: "User:", "Assistant:", "Human:", "AI:", "Turn", etc.
        turn_markers = ["\nUser:", "\nAssistant:", "\nHuman:", "\nAI:", "\nTurn "]
        
        chunks = []
        current_chunk = ""
        lines = text.split("\n")
        
        for line in lines:
            # Check if this line starts a new turn
            is_new_turn = any(marker.strip() in line for marker in turn_markers)
            
            if is_new_turn and current_chunk and len(current_chunk) >= max_length:
                chunks.append(current_chunk.strip())
                current_chunk = line
            else:
                current_chunk += "\n" + line if current_chunk else line
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If no good split points found, fall back to general chunking
        if not chunks:
            return GeneralChunker(self._cfg).chunk(doc)
        
        return chunks


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
