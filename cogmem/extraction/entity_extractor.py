"""Entity extraction module: identifies and extracts entities from text"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import ExtractionConfig
    from ..storage.schema import Entity
    from ..models.llm_client import LLMClient


class EntityExtractor:
    """Entity extractor: uses StructuredExtractor as backend for unified extraction"""

    def __init__(self, config: ExtractionConfig, llm_client: LLMClient):
        """
        Args:
            config: Extraction configuration
            llm_client: LLM client
        """
        self.config = config
        self.llm = llm_client
        # Use StructuredExtractor as backend
        from ..extraction.structured_extractor import StructuredExtractor
        self._structured_extractor = StructuredExtractor(llm_client)

    def extract(self, text: str, context_type: str = "general") -> List[Entity]:
        """Extract entities
        
        Args:
            text: Text to extract from
            context_type: Context type (general|dialog|observation), kept for backward compatibility
            
        Returns:
            List of extracted entities
        """
        # Use StructuredExtractor for unified extraction
        import hashlib
        doc_id = f"entity_extract_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"
        
        extraction_result = self._structured_extractor.extract(
            input_text=text,
            doc_id=doc_id,
            session_id=None,
            turn_id=None,
            question_date_iso=None,
            context_window_start_iso=None,
            context_window_end_iso=None
        )
        
        # Convert to Entity format
        entities = self._structured_extractor.to_entities(extraction_result)
        
        # Post-processing: temporal entity normalization (if enabled)
        if self.config.enable_temporal_extraction:
            entities = self._normalize_temporal_entities(entities)
        
        return entities

    def _normalize_temporal_entities(self, entities: List[Entity]) -> List[Entity]:
        """Normalize temporal entities
        
        For example:
        - "May 7, 2023" → "7 May 2023"
        - "last Tuesday" → keep as is (requires context reasoning)
        """
        import re
        for entity in entities:
            if entity.subtype == "temporal":
                entity.name = self._normalize_date(entity.name)
        return entities

    def _normalize_date(self, date_str: str) -> str:
        """Date format normalization"""
        # Simple date format conversion
        date_str = date_str.strip()
        
        # "May 7, 2023" -> "7 May 2023"
        pattern1 = r'(\w+)\s+(\d+),\s+(\d{4})'
        match1 = re.match(pattern1, date_str)
        if match1:
            month, day, year = match1.groups()
            return f"{day} {month} {year}"
        
        # "7/5/2023" -> "7 May 2023" (requires month mapping)
        pattern2 = r'(\d+)/(\d+)/(\d{4})'
        match2 = re.match(pattern2, date_str)
        if match2:
            day, month_num, year = match2.groups()
            months = ["", "January", "February", "March", "April", "May", "June",
                     "July", "August", "September", "October", "November", "December"]
            try:
                month_idx = int(month_num)
                if 1 <= month_idx <= 12:
                    return f"{day} {months[month_idx]} {year}"
            except ValueError:
                pass
        
        # Other formats keep as is
        return date_str
