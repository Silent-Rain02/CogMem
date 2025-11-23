"""Structured extractor for person-centered graph: extracts Person, Event, Fact, Preference, EvidenceSpan, TimeNormalization"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional
from datetime import datetime

from ..models.llm_client import LLMClient
from ..storage.schema import (
    Person, Event, Fact, Preference, EvidenceSpan, TimeNormalization, Metadata, Entity
)
from ..utils.temporal_parser import normalize_time
from ..utils.json_utils import safe_json_loads


class StructuredExtractor:
    """Extracts structured information using the provided prompt template"""
    
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client
    
    def extract(
        self,
        input_text: str,
        doc_id: str,
        session_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        question_date_iso: Optional[str] = None,
        context_window_start_iso: Optional[str] = None,
        context_window_end_iso: Optional[str] = None
    ) -> Dict[str, List]:
        """Extract structured information from input text
        
        Returns:
            Dict with keys: nodes (persons, entities, events, facts, preferences, evidence_spans, time_normalizations)
            and edges (belongs_to, spoken_by, involved_in, asserts, mentions, supports_answer, supersedes, same_as)
        """
        from ..prompt import STRUCTURED_EXTRACTION_SYSTEM, STRUCTURED_EXTRACTION_USER
        
        # Format the prompt with template variables
        user_prompt = STRUCTURED_EXTRACTION_USER.format(
            INPUT_TEXT=input_text,
            DOC_ID=doc_id,
            SESSION_ID=session_id or "",
            TURN_ID=turn_id or "",
            QUESTION_DATE_ISO=question_date_iso or "",
            CONTEXT_WINDOW_START_ISO=context_window_start_iso or "",
            CONTEXT_WINDOW_END_ISO=context_window_end_iso or ""
        )
        
        # Call LLM
        response = self.llm.chat(
            system=STRUCTURED_EXTRACTION_SYSTEM,
            user=user_prompt
        )
        
        # Parse JSON response using safe_json_loads (with demjson3 support)
        # safe_json_loads 已经内置了 demjson3 支持和回退机制
        data = safe_json_loads(response)
        if not isinstance(data, dict):
            # 如果 safe_json_loads 返回的不是字典，尝试额外的清理和解析
            try:
                # 尝试清理响应文本（移除可能的 markdown 代码块标记）
                cleaned = response.strip()
                if cleaned.startswith("```"):
                    # 移除 markdown 代码块标记
                    lines = cleaned.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].strip() == "```":
                        lines = lines[:-1]
                    cleaned = "\n".join(lines)
                    # 再次尝试解析清理后的文本
                    data = safe_json_loads(cleaned)
                    if not isinstance(data, dict):
                        return self._empty_result()
                else:
                    return self._empty_result()
            except Exception as e:
                # 调试：打印解析失败的信息
                import traceback
                print(f"⚠️ JSON解析失败: {str(e)}")
                print(f"   响应前300字符: {str(response)[:300]}")
                # 只在调试模式下打印完整堆栈
                import os
                if os.getenv("DEBUG", "").lower() in ("1", "true", "yes"):
                    traceback.print_exc()
                return self._empty_result()
        
        # Process nodes and edges
        nodes = data.get("nodes", {})
        edges = data.get("edges", {})
        
        # 先解析 evidence_spans，用于后续解析其他节点时查找
        evidence_spans_list = self._parse_evidence_spans(nodes.get("evidence_spans", []), turn_id)
        # 创建 evidence_span ID 到详细信息的映射
        evidence_span_map = {span.id: span for span in evidence_spans_list}
        
        result = {
            "nodes": {
                "persons": self._parse_persons(nodes.get("persons", []), session_id, turn_id, evidence_span_map),
                "entities": self._parse_entities(nodes.get("entities", []), session_id, turn_id, evidence_span_map),
                "events": self._parse_events(nodes.get("events", []), session_id, turn_id, question_date_iso, evidence_span_map),
                "facts": self._parse_facts(nodes.get("facts", []), session_id, turn_id, question_date_iso, evidence_span_map),
                "preferences": self._parse_preferences(nodes.get("preferences", []), session_id, turn_id, question_date_iso, evidence_span_map),
                "evidence_spans": evidence_spans_list,
                "time_normalizations": self._parse_time_normalizations(
                    nodes.get("time_normalizations", []),
                    question_date_iso,
                    context_window_start_iso,
                    context_window_end_iso
                )
            },
            "edges": {
                "belongs_to": edges.get("belongs_to", []),
                "spoken_by": edges.get("spoken_by", []),
                "involved_in": edges.get("involved_in", []),
                "asserts": edges.get("asserts", []),
                "mentions": edges.get("mentions", []),
                "supports_answer": edges.get("supports_answer", []),
                "supersedes": edges.get("supersedes", []),
                "same_as": edges.get("same_as", [])
            },
            "meta": data.get("meta", {})
        }
        
        return result
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            "nodes": {
                "persons": [],
                "entities": [],
                "events": [],
                "facts": [],
                "preferences": [],
                "evidence_spans": [],
                "time_normalizations": []
            },
            "edges": {
                "belongs_to": [],
                "spoken_by": [],
                "involved_in": [],
                "asserts": [],
                "mentions": [],
                "supports_answer": [],
                "supersedes": [],
                "same_as": []
            },
            "meta": {}
        }
    
    def _parse_persons(self, items: List[Dict], session_id: Optional[str], turn_id: Optional[str], evidence_span_map: Optional[Dict] = None) -> List[Person]:
        """Parse Person nodes"""
        if evidence_span_map is None:
            evidence_span_map = {}
        
        persons = []
        for item in items:
            try:
                # 处理 evidence_span：可能是字符串ID或字典
                evidence_span = item.get("evidence_span")
                evidence_span_dict = None
                if isinstance(evidence_span, str):
                    # 如果是字符串ID，从 evidence_span_map 中查找
                    if evidence_span in evidence_span_map:
                        span_obj = evidence_span_map[evidence_span]
                        evidence_span_dict = {
                            "text": span_obj.text,
                            "char_start": span_obj.char_start,
                            "char_end": span_obj.char_end
                        }
                elif isinstance(evidence_span, dict):
                    evidence_span_dict = evidence_span
                
                person = Person(
                    id=item.get("id", f"person_{uuid.uuid4().hex[:12]}"),
                    canonical_name=item.get("canonical_name", ""),
                    aliases=item.get("aliases", []),
                    persona_traits=item.get("persona_traits", []),
                    profession=item.get("profession", ""),
                    background=item.get("background", ""),
                    identity=item.get("identity", ""),
                    confidence=float(item.get("confidence", 1.0)),
                    evidence_span=evidence_span_dict,
                    source_session_id=item.get("source_session_id", session_id),
                    source_turn_id=item.get("source_turn_id", turn_id),
                    community_ids=item.get("community_hint", {}).get("topics", []),
                    topic_ids=item.get("community_hint", {}).get("topics", []),
                    metadata=Metadata(
                        source=evidence_span_dict.get("text", "") if evidence_span_dict else "",
                        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        source_session_id=item.get("source_session_id", session_id),
                        source_turn_id=item.get("source_turn_id", turn_id)
                    )
                )
                persons.append(person)
            except Exception as e:
                # 调试：打印解析失败的信息
                import traceback
                print(f"⚠️ Person解析失败: {str(e)}")
                print(f"   Item: {item}")
                traceback.print_exc()
                continue
        return persons
    
    def _parse_entities(self, items: List[Dict], session_id: Optional[str], turn_id: Optional[str], evidence_span_map: Optional[Dict] = None) -> List[Dict]:
        """Parse Entity nodes (return as dict for compatibility)"""
        entities = []
        for item in items:
            try:
                entity = {
                    "id": item.get("id", f"entity_{uuid.uuid4().hex[:12]}"),
                    "canonical_name": item.get("canonical_name", ""),
                    "category": item.get("category", "other"),
                    "aliases": item.get("aliases", []),
                    "confidence": float(item.get("confidence", 1.0)),
                    "evidence_span": item.get("evidence_span"),
                    "source_session_id": item.get("source_session_id", session_id),
                    "source_turn_id": item.get("source_turn_id", turn_id)
                }
                entities.append(entity)
            except Exception:
                continue
        return entities
    
    def _parse_events(self, items: List[Dict], session_id: Optional[str], turn_id: Optional[str], question_date_iso: Optional[str], evidence_span_map: Optional[Dict] = None) -> List[Event]:
        """Parse Event nodes"""
        if evidence_span_map is None:
            evidence_span_map = {}
        
        events = []
        for item in items:
            try:
                # 处理 evidence_span：可能是字符串ID或字典
                evidence_span = item.get("evidence_span")
                evidence_span_dict = None
                if isinstance(evidence_span, str):
                    if evidence_span in evidence_span_map:
                        span_obj = evidence_span_map[evidence_span]
                        evidence_span_dict = {
                            "text": span_obj.text,
                            "char_start": span_obj.char_start,
                            "char_end": span_obj.char_end
                        }
                elif isinstance(evidence_span, dict):
                    evidence_span_dict = evidence_span
                
                event = Event(
                    id=item.get("id", f"event_{uuid.uuid4().hex[:12]}"),
                    title=item.get("title", ""),
                    participants=item.get("participants", []),
                    location=item.get("location"),
                    time_start=item.get("time_start"),
                    time_end=item.get("time_end"),
                    granularity=item.get("granularity"),
                    time_confidence=float(item.get("time_confidence", 1.0)),
                    temporal_conflict=bool(item.get("temporal_conflict", False)),
                    evidence_span=evidence_span_dict,
                    source_session_id=item.get("source_session_id", session_id),
                    source_turn_id=item.get("source_turn_id", turn_id),
                    metadata=Metadata(
                        source=evidence_span_dict.get("text", "") if evidence_span_dict else "",
                        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        source_session_id=item.get("source_session_id", session_id),
                        source_turn_id=item.get("source_turn_id", turn_id)
                    )
                )
                events.append(event)
            except Exception:
                continue
        return events
    
    def _parse_facts(self, items: List[Dict], session_id: Optional[str], turn_id: Optional[str], question_date_iso: Optional[str], evidence_span_map: Optional[Dict] = None) -> List[Fact]:
        """Parse Fact nodes"""
        if evidence_span_map is None:
            evidence_span_map = {}
        
        facts = []
        for item in items:
            try:
                # 处理 evidence_span：可能是字符串ID或字典
                evidence_span = item.get("evidence_span")
                evidence_span_dict = None
                if isinstance(evidence_span, str):
                    if evidence_span in evidence_span_map:
                        span_obj = evidence_span_map[evidence_span]
                        evidence_span_dict = {
                            "text": span_obj.text,
                            "char_start": span_obj.char_start,
                            "char_end": span_obj.char_end
                        }
                elif isinstance(evidence_span, dict):
                    evidence_span_dict = evidence_span
                
                fact = Fact(
                    id=item.get("id", f"fact_{uuid.uuid4().hex[:12]}"),
                    subject=item.get("subject", ""),
                    predicate=item.get("predicate", ""),
                    object=item.get("object"),
                    value=item.get("value"),
                    value_type=item.get("value_type"),
                    valid_from=item.get("valid_from"),
                    valid_to=item.get("valid_to"),
                    evidence_span=evidence_span_dict,
                    source_session_id=item.get("source_session_id", session_id),
                    source_turn_id=item.get("source_turn_id", turn_id),
                    confidence=float(item.get("confidence", 1.0)),
                    supersedes=item.get("supersedes", []),
                    metadata=Metadata(
                        source=evidence_span_dict.get("text", "") if evidence_span_dict else "",
                        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        source_session_id=item.get("source_session_id", session_id),
                        source_turn_id=item.get("source_turn_id", turn_id)
                    )
                )
                facts.append(fact)
            except Exception:
                continue
        return facts
    
    def _parse_preferences(self, items: List[Dict], session_id: Optional[str], turn_id: Optional[str], question_date_iso: Optional[str], evidence_span_map: Optional[Dict] = None) -> List[Preference]:
        """Parse Preference nodes"""
        if evidence_span_map is None:
            evidence_span_map = {}
        
        preferences = []
        for item in items:
            try:
                # 处理 evidence_span：可能是字符串ID或字典
                evidence_span = item.get("evidence_span")
                evidence_span_dict = None
                if isinstance(evidence_span, str):
                    if evidence_span in evidence_span_map:
                        span_obj = evidence_span_map[evidence_span]
                        evidence_span_dict = {
                            "text": span_obj.text,
                            "char_start": span_obj.char_start,
                            "char_end": span_obj.char_end
                        }
                elif isinstance(evidence_span, dict):
                    evidence_span_dict = evidence_span
                
                preference = Preference(
                    id=item.get("id", f"pref_{uuid.uuid4().hex[:12]}"),
                    holder=item.get("holder", ""),
                    aspect=item.get("aspect", ""),
                    polarity=item.get("polarity", "neutral"),
                    strength=float(item.get("strength", 0.5)),
                    valid_from=item.get("valid_from"),
                    valid_to=item.get("valid_to"),
                    evidence_span=evidence_span_dict,
                    source_session_id=item.get("source_session_id", session_id),
                    source_turn_id=item.get("source_turn_id", turn_id),
                    confidence=float(item.get("confidence", 1.0)),
                    supersedes=item.get("supersedes", []),
                    metadata=Metadata(
                        source=evidence_span_dict.get("text", "") if evidence_span_dict else "",
                        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        source_session_id=item.get("source_session_id", session_id),
                        source_turn_id=item.get("source_turn_id", turn_id)
                    )
                )
                preferences.append(preference)
            except Exception:
                continue
        return preferences
    
    def _parse_evidence_spans(self, items: List[Dict], turn_id: Optional[str]) -> List[EvidenceSpan]:
        """Parse EvidenceSpan nodes"""
        spans = []
        for item in items:
            try:
                # 支持 "source_text" 或 "text" 字段（prompt 示例使用 source_text）
                text = item.get("source_text") or item.get("text", "")
                span = EvidenceSpan(
                    id=item.get("id", f"ev_{uuid.uuid4().hex[:12]}"),
                    turn_id=item.get("turn_id", turn_id or ""),
                    char_start=int(item.get("char_start", 0)),
                    char_end=int(item.get("char_end", 0)),
                    text=text,
                    has_answer=bool(item.get("has_answer", False)),
                    metadata=Metadata(
                        source=text,
                        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                )
                spans.append(span)
            except Exception:
                continue
        return spans
    
    def _parse_time_normalizations(
        self,
        items: List[Dict],
        question_date_iso: Optional[str],
        context_window_start_iso: Optional[str],
        context_window_end_iso: Optional[str]
    ) -> List[TimeNormalization]:
        """Parse TimeNormalization nodes"""
        normalizations = []
        for item in items:
            try:
                # Use temporal parser if time fields are missing
                original_text = item.get("original_text", "")
                if not item.get("time_start"):
                    # Try to normalize using temporal parser
                    normalized = normalize_time(
                        original_text,
                        question_date_iso,
                        context_window_start_iso,
                        context_window_end_iso
                    )
                    time_start = normalized.get("time_start")
                    time_end = normalized.get("time_end")
                    granularity = normalized.get("granularity")
                    confidence = normalized.get("confidence", 0.0)
                    anchor = normalized.get("anchor", "unknown")
                else:
                    time_start = item.get("time_start")
                    time_end = item.get("time_end")
                    granularity = item.get("granularity")
                    confidence = float(item.get("confidence", 1.0))
                    anchor = item.get("anchor", "unknown")
                
                norm = TimeNormalization(
                    id=item.get("id", f"time_{uuid.uuid4().hex[:12]}"),
                    original_text=original_text,
                    time_start=time_start,
                    time_end=time_end,
                    granularity=granularity,
                    confidence=confidence,
                    anchor=anchor,
                    evidence_span=item.get("evidence_span"),
                    metadata=Metadata(
                        source=original_text,
                        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                )
                normalizations.append(norm)
            except Exception:
                continue
        return normalizations
    
    def to_entities(self, extraction_result: Dict[str, List]) -> List[Entity]:
        """Convert structured extraction result to Entity list for backward compatibility
        
        Args:
            extraction_result: Result from extract() method
            
        Returns:
            List of Entity objects converted from Person/Event/Fact/Preference nodes
        """
        entities = []
        nodes = extraction_result.get("nodes", {})
        
        # Convert Person nodes to Entity
        for person in nodes.get("persons", []):
            if isinstance(person, Person):
                entity = Entity(
                    id=person.id,
                    name=person.canonical_name,
                    description=f"Person: {', '.join(person.aliases) if person.aliases else person.canonical_name}",
                    type="world",
                    subtype="person",
                    aliases=person.aliases,
                    confidence=person.confidence,
                    # Person 模型没有 valid_from, valid_to, time_interval 属性
                    valid_from=None,
                    valid_to=None,
                    time_interval=None,
                    metadata=person.metadata
                )
                entities.append(entity)
        
        # Convert Event nodes to Entity
        for event in nodes.get("events", []):
            if isinstance(event, Event):
                description = f"Event: {event.title}"
                if event.location:
                    description += f" at {event.location}"
                # Event 模型没有 confidence 和 time_interval，使用 time_confidence 和构建 time_interval
                time_interval = None
                if event.time_start or event.time_end:
                    time_interval = {
                        "start": event.time_start,
                        "end": event.time_end,
                        "granularity": event.granularity
                    }
                entity = Entity(
                    id=event.id,
                    name=event.title,
                    description=description,
                    type="world",
                    subtype="event",
                    confidence=event.time_confidence,  # 使用 time_confidence 作为 confidence
                    time_interval=time_interval,
                    metadata=event.metadata
                )
                entities.append(entity)
        
        # Convert Fact nodes to Entity (representing the fact as an entity)
        for fact in nodes.get("facts", []):
            if isinstance(fact, Fact):
                fact_str = f"{fact.subject} {fact.predicate}"
                if fact.object:
                    fact_str += f" {fact.object}"
                elif fact.value:
                    fact_str += f" {fact.value}"
                entity = Entity(
                    id=fact.id,
                    name=fact_str,
                    description=f"Fact: {fact_str}",
                    type="world",
                    subtype="concept",
                    confidence=fact.confidence,
                    valid_from=fact.valid_from,
                    valid_to=fact.valid_to,
                    metadata=fact.metadata
                )
                entities.append(entity)
        
        # Convert Preference nodes to Entity
        for pref in nodes.get("preferences", []):
            if isinstance(pref, Preference):
                entity = Entity(
                    id=pref.id,
                    name=f"{pref.holder} {pref.aspect}",
                    description=f"Preference: {pref.holder} {pref.polarity} {pref.aspect} (strength: {pref.strength})",
                    type="preference",
                    subtype="preference",
                    confidence=pref.confidence,
                    valid_from=pref.valid_from,
                    valid_to=pref.valid_to,
                    metadata=pref.metadata
                )
                entities.append(entity)
        
        # Convert Entity dicts to Entity objects (from structured extraction)
        for entity_dict in nodes.get("entities", []):
            if isinstance(entity_dict, dict):
                try:
                    entity = Entity(
                        id=entity_dict.get("id", f"entity_{uuid.uuid4().hex[:12]}"),
                        name=entity_dict.get("canonical_name", ""),
                        description=entity_dict.get("description", ""),
                        type="world",
                        subtype=entity_dict.get("category", "concept"),
                        aliases=entity_dict.get("aliases", []),
                        confidence=float(entity_dict.get("confidence", 1.0)),
                        metadata=Metadata(
                            source=entity_dict.get("evidence_span", {}).get("text", ""),
                            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            source_session_id=entity_dict.get("source_session_id"),
                            source_turn_id=entity_dict.get("source_turn_id")
                        )
                    )
                    entities.append(entity)
                except Exception:
                    continue
        
        return entities

