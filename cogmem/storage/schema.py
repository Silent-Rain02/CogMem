from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional, Union, Tuple, Any
from pydantic import BaseModel, Field, field_validator, ValidationInfo, model_validator

# ============================================================================
# Memory Lifecycle (unified state for Fact/Claim and retrieval priority)
# ============================================================================

class MemoryLifecycle(str, Enum):
    """Unified lifecycle state for memory nodes (Fact, Claim).
    
    Migration rules:
    - candidate: From indexing or weak extraction; not yet consolidated. Low retrieval priority.
    - stable: Consolidated / deduplicated; canonical for QA. Full retrieval priority.
    - archived: Superseded or deprecated; retained for history. Excluded from default retrieval.
    """
    CANDIDATE = "candidate"   # Ephemeral / pre-consolidation
    STABLE = "stable"         # Canonical after consolidation
    ARCHIVED = "archived"     # Superseded; soft-deleted from active use

# ============================================================================
# Node Type Enumeration (P-E-C-F + Claim Core Architecture)
# ============================================================================

class NodeTypeEnum(str, Enum):
    """Strongly-typed node types for the knowledge graph

    Core 5 types for v3.2:
    - PERSON: People/characters
    - EVENT: Time-bounded happenings with participants
    - CONCEPT: Abstract concepts/topics/objects (replaces Entity)
    - FACT: Objective facts and self-reported preferences (Is/Self)
    - CLAIM: Subjective claims/opinions (Believes-Others)
    """
    PERSON = "person"
    EVENT = "event"
    CONCEPT = "concept"
    FACT = "fact"
    CLAIM = "claim"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def get_prefix(cls, node_type: 'NodeTypeEnum') -> str:
        """Get the ID prefix for a given node type"""
        return f"{node_type.value}_"

# ============================================================================
# Edge Type Definitions (v3.2 Semantic Edges)
# ============================================================================

# Edge types with semantic meaning (v3.3 Enhanced)
EdgeType = Literal[
    # Person -> Event (with role distinction)
    "PARTICIPATED_IN",      # Person was PRIMARY participant in Event (main actor)
    "INVOLVED_IN",          # Person was involved in Event (general involvement)
    "MENTIONED_IN",         # Person was MENTIONED in Event but not primary actor
    "CO_PARTICIPATED_IN",   # Person co-participated with others in Event
    
    # Event -> Concept
    "INVOLVES",             # Event involves Concept
    
    # Person -> Concept (preferences/habits)
    "PREFERS",              # Person prefers Concept
    "LIKES",                # Person likes Concept
    "USES",                 # Person uses Concept
    "IS_A",                 # Person is a Concept (role/type)
    "HAS_ATTRIBUTE",        # Person has attribute Concept
    
    # Person -> Claim -> Target (Theory of Mind)
    "CLAIMS",               # Person -> Claim (who made the claim)
    "ABOUT",                # Claim -> Target (what the claim is about)
    
    # Fact relationships
    "ASSERTS",              # Event/Fact asserts Fact
    "SUPERSEDES",           # Fact/Claim supersedes Fact/Claim (temporal chain)
    "CONTRADICTS",          # Fact contradicts Fact
    "SUPPORTED_BY",         # Fact -> Event (Fact is supported by specific events)
    
    # Temporal (v3.3 Enhanced)
    "BEFORE",               # Event before Event
    "AFTER",                # Event after Event
    "AT_TIME",              # Event at specific time
    "HAPPENED_BEFORE",      # Event -> Event (explicit temporal ordering)
    "HAPPENED_DURING",      # Event -> TimePeriod (e.g., "2023-05")
    "CONCURRENT_WITH",      # Event -> Event (happened at same time)
    
    # Concept Hierarchy (v3.3 New)
    "IS_PARENT_OF",         # Concept -> Concept (hierarchy: c_food -> c_pizza)
    "IS_CHILD_OF",          # Concept -> Concept (reverse hierarchy)
    "RELATED_TO",           # Concept -> Concept (semantic similarity)
    
    # Identity
    "SAME_AS",              # Node same as Node
    "ALIAS_OF",             # Node alias of Node
]

# ============================================================================
# Metadata Model
# ============================================================================

class Metadata(BaseModel):
    source: Optional[str] = None
    created_at: Optional[str] = None  # YYYY-MM-DD HH:MM:SS 格式，可读时间戳
    updated_at: Optional[str] = None  # YYYY-MM-DD HH:MM:SS 格式，可读时间戳
    doc_type: Optional[str] = None  # 文档类型（如 "dialog", "general"），用于chunker选择策略

# ============================================================================
# Base Node Model
# ============================================================================

class BaseNode(BaseModel):
    """Base model for all knowledge graph nodes

    Provides common fields and validation for all node types:
    - id: Unique identifier with type prefix (e.g., "person_", "event_")
    - content: Core textual content/description
    - timestamp: Optional temporal marker
    - is_archived: Archival status flag (authoritative for "excluded from default retrieval")
    - metadata: Creation and update metadata
    
    Lifecycle (Fact/Claim): Prefer memory_lifecycle when set; otherwise is_archived=True => ARCHIVED.
    Version chain: For Fact/Claim, SUPERSEDES + valid_to + is_archived form the single source of truth;
    is_latest is derived and should be True for the head of the chain only.
    """
    id: str = Field(description="Unique node ID with type prefix (e.g., 'person_abc123')")
    content: str = Field(default="", description="Core textual content or description")
    timestamp: Optional[str] = Field(None, description="ISO-8601 timestamp for temporal nodes", exclude=True)
    is_archived: bool = Field(default=False, description="Whether this node is archived (excluded from default retrieval)")
    metadata: Metadata = Field(default_factory=Metadata, description="Creation and update metadata")
    # Optional lifecycle and priority (used by Fact/Claim and retrieval)
    memory_lifecycle: Optional[MemoryLifecycle] = Field(None, description="Unified state: candidate | stable | archived")
    importance_score: Optional[float] = Field(None, description="Optional 0-1 score for retrieval ranking")
    last_accessed_at: Optional[str] = Field(None, description="ISO-8601 last time this node was retrieved (for decay/priority)")

    @field_validator('id')
    @classmethod
    def validate_id_format(cls, v: str, info: ValidationInfo) -> str:
        """Validate that ID is a non-empty string"""
        if not v or not isinstance(v, str):
            raise ValueError('ID must be a non-empty string')
        return v.strip()

    @field_validator('timestamp', mode='before')
    @classmethod
    def validate_timestamp(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Validate ISO-8601 timestamp format"""
        if v is None:
            return v
        
        # Skip validation if it's a property object (from Event class override)
        if hasattr(v, '__get__') or str(type(v)) == "<class 'property'>":
            return None

        from datetime import datetime
        try:
            if 'T' in v:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            else:
                datetime.strptime(v, "%Y-%m-%d")
            return v
        except (ValueError, TypeError):
            if not isinstance(v, str):
                return None
            raise ValueError(
                f'Invalid ISO-8601 timestamp format: {v}. '
                f'Expected: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS'
            )

    model_config = {
        "validate_assignment": True,
        "validate_default": True,
        "use_enum_values": True,
    }

# ============================================================================
# Core Node Types (P-E-C-F + Claim)
# ============================================================================

class Person(BaseNode):
    """Person node with minimal core fields (v3.3 Streamlined)

    Represents a person/character in the knowledge graph.
    All non-Person entities should be Concept nodes.
    
    Design Principle (v3.3):
    - Person nodes should be LEAN - only contain identity information
    - All descriptive info (profession, traits, etc.) should be Fact nodes
    - This prevents data duplication and update inconsistency
    
    Deprecated fields (moved to Fact nodes):
    - persona_traits -> Fact(subject=person, predicate="has_trait", object=concept)
    - profession -> Fact(subject=person, predicate="works_as", object=concept)
    - background -> Fact(subject=person, predicate="has_background", object=concept)
    - identity -> Fact(subject=person, predicate="identifies_as", object=concept)
    """
    id: str = Field(description="Person ID with 'person_' or 'p_' prefix")
    canonical_name: str = Field(description="Primary name of the person")
    aliases: List[str] = Field(default_factory=list, description="Alternative names/nicknames")
    # Deprecated but kept for backward compatibility - prefer using Fact nodes
    persona_traits: List[str] = Field(default_factory=list, description="[DEPRECATED] Use Fact nodes instead")
    profession: str = Field(default="", description="[DEPRECATED] Use Fact nodes instead")
    background: str = Field(default="", description="[DEPRECATED] Use Fact nodes instead")
    identity: str = Field(default="", description="[DEPRECATED] Use Fact nodes instead")
    confidence: float = Field(default=1.0, description="Confidence score (0-1)")
    evidence_span: Optional[Dict[str, Union[str, int]]] = Field(None, description="{text, char_start, char_end}", exclude=True)
    session_time: Optional[str] = Field(None, description="Session time in ISO format (YYYY-MM-DD) for temporal reasoning")

    @property
    def searchable_text(self) -> str:
        """Generate text for vector embedding (v3.3 Enhancement)
        
        Only includes canonical_name and aliases - NOT deprecated fields
        to avoid retrieval noise from outdated information.
        """
        parts = [self.canonical_name]
        if self.aliases:
            parts.extend(self.aliases)
        return " ".join(parts)

    @field_validator('canonical_name')
    @classmethod
    def validate_canonical_name(cls, v: str, info: ValidationInfo) -> str:
        """Validate that canonical_name is not empty"""
        if not v or not isinstance(v, str):
            raise ValueError('canonical_name must be a non-empty string')
        return v.strip()

    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str, info: ValidationInfo) -> str:
        """Validate ID format"""
        if not v or not isinstance(v, str):
            raise ValueError('ID must be a non-empty string')
        return v.strip()

    model_config = {
        "validate_assignment": True,
        "validate_default": True,
    }

class Concept(BaseNode):
    """Concept node representing abstract concepts, topics, objects, and entities (v3.3 Enhanced)
    
    Replaces Entity class. All non-Person entities (organizations, locations, 
    objects, abstract concepts) should be Concept nodes.
    
    Concepts are normalized via ConceptManager for deduplication.
    
    v3.3 Changes:
    - Removed `embedding` field (stored in TypedVectorStore instead)
    - Added `parent_id` for hierarchy support (e.g., c_pizza IS_CHILD_OF c_food)
    - Added `category` for broad classification
    - Added `searchable_text` property for consistent vectorization
    """
    id: str = Field(description="Concept ID with 'concept_' or 'c_' prefix")
    name: str = Field(description="Normalized concept name (lowercase, underscores)")
    aliases: List[str] = Field(default_factory=list, description="Alternative names/aliases")
    description: Optional[str] = Field(None, description="Concept description")
    # v3.3: Hierarchy support
    parent_id: Optional[str] = Field(None, description="Parent concept ID for hierarchy (e.g., c_food for c_pizza)")
    category: Optional[str] = Field(None, description="Broad category (e.g., 'activity', 'food', 'location', 'organization')")
    # REMOVED: embedding field - now stored in TypedVectorStore
    # This reduces serialization overhead and prevents stale embeddings

    @property
    def searchable_text(self) -> str:
        """Generate text for vector embedding (v3.3 Enhancement)
        
        Includes name (converted to natural language), aliases, and description.
        """
        # Convert normalized name to natural language
        natural_name = self.name.replace('_', ' ').replace('c_', '')
        parts = [natural_name]
        if self.aliases:
            parts.extend(self.aliases)
        if self.description:
            parts.append(self.description)
        return " ".join(parts)

    @property
    def display_name(self) -> str:
        """Human-readable name (v3.3 Enhancement)"""
        return self.name.replace('_', ' ').replace('c_', '').title()

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str, info: ValidationInfo) -> str:
        """Validate that name is not empty"""
        if not v or not isinstance(v, str):
            raise ValueError('name must be a non-empty string')
        return v.strip()

    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str, info: ValidationInfo) -> str:
        """Validate ID format"""
        if not v or not isinstance(v, str):
            raise ValueError('ID must be a non-empty string')
        return v.strip()

    model_config = {
        "validate_assignment": True,
        "validate_default": True,
    }

class Event(BaseNode):
    """Event node with participants and time interval (v3.3 Enhanced)

    Temporal Field Clarification (v3.3):
    - time_start_raw: Original text from dialogue (e.g., "yesterday", "last week")
    - session_time: When the dialogue was recorded (reference for relative time)
    - time_resolved: Absolute time computed from time_start_raw + session_time
    - time_start: Validated ISO-8601 format (may equal time_resolved or be manually set)
    
    Time Resolution Flow:
    1. LLM extracts "yesterday" -> time_start_raw = "yesterday"
    2. session_time = "2023-05-08" (from dialogue metadata)
    3. Indexer computes: time_resolved = "2023-05-07"
    4. time_start = time_resolved (after validation)
    
    Searchable Text:
    - title + description + location are used for vectorization
    - Time fields are NOT vectorized (use temporal queries instead)
    """
    id: str = Field(description="Event ID with 'event_' prefix")
    title: str = Field(description="Event title/summary - PRIMARY searchable field")
    participants: List[str] = Field(default_factory=list, description="Person IDs involved")
    concepts: List[str] = Field(default_factory=list, description="Concept IDs associated with the event (e.g., ['c_dancing', 'c_destress'])")
    location: Optional[str] = Field(None, description="Event location")
    description: Optional[str] = Field(None, description="Enhanced event description with context - SECONDARY searchable field")
    # Time fields with clear semantics (v3.3)
    time_start_raw: Optional[str] = Field(None, description="Raw time text from LLM (e.g., 'yesterday', 'last week') - PRESERVE ORIGINAL")
    time_end_raw: Optional[str] = Field(None, description="Raw end time text from LLM")
    session_time: Optional[str] = Field(None, description="When dialogue was recorded (YYYY-MM-DD) - REFERENCE for relative time")
    time_resolved: Optional[str] = Field(None, description="Computed absolute time from time_start_raw + session_time (v3.3 NEW)")
    time_start: Optional[str] = Field(None, description="Validated ISO-8601 start time - USE FOR QUERIES")
    time_end: Optional[str] = Field(None, description="Validated ISO-8601 end time")
    granularity: Optional[Literal["year", "month", "day", "time"]] = Field(None, description="Time granularity")
    time_confidence: float = Field(default=1.0, description="Time confidence score (0-1)")
    temporal_conflict: bool = Field(default=False, description="Whether temporal data has conflicts")
    evidence_span: Optional[Dict[str, Union[str, int]]] = Field(None, description="{text, char_start, char_end}", exclude=True)
    
    model_config = {
        "validate_assignment": True,
        "validate_default": True,
    }

    @model_validator(mode='before')
    @classmethod
    def exclude_timestamp_from_input(cls, data: Any) -> Any:
        """Remove timestamp from input data since it's a computed property"""
        if isinstance(data, dict) and 'timestamp' in data:
            data = {k: v for k, v in data.items() if k != 'timestamp'}
        return data

    @property
    def timestamp(self) -> Optional[str]:
        """Primary timestamp for the event (time_start if available, otherwise time_end)"""
        return self.time_start or self.time_end
    
    @property
    def searchable_text(self) -> str:
        """Generate text for vector embedding (v3.3 Enhancement)
        
        Includes title, description, and location.
        EXCLUDES time fields - use temporal queries for time-based search.
        """
        parts = [self.title]
        if self.description:
            parts.append(self.description)
        if self.location:
            parts.append(f"at {self.location}")
        return " ".join(parts)
    
    @property
    def effective_time(self) -> Optional[str]:
        """Best available time for this event (v3.3 Enhancement)
        
        Priority: time_resolved > time_start > session_time
        """
        return self.time_resolved or self.time_start or self.session_time
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include computed timestamp"""
        data = super().model_dump(**kwargs)
        data['timestamp'] = self.timestamp
        return data

    @field_validator('time_start', 'time_end')
    @classmethod
    def validate_iso_datetime(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Validate ISO-8601 datetime format for temporal fields

        Supports flexible time formats:
        - Full datetime: YYYY-MM-DDTHH:MM:SS
        - Date only: YYYY-MM-DD
        - Partial dates with 'XX' placeholders
        - Year-month only: YYYY-MM
        - Year only: YYYY (normalized to YYYY-01-01)
        - Special values: 'recently', 'lately' (treated as unknown)
        """
        if v is None:
            return v

        from datetime import datetime
        import re

        v = str(v).strip()

        # Pre-process ISO-8601 format
        if 't' in v and ('z' in v or '+' in v or v.count(':') >= 2):
            v = v.replace('t', 'T').replace('z', 'Z')

        # Handle empty or "unknown"-like strings
        if not v or v.lower() in ["unknown", "none", "null", "n/a", "na", "unspecified"]:
            return "XXXX-XX-XX"

        v_lower = v.lower()
        if v_lower in ["recently", "recent", "lately", "sometime", "sometimes", "earlier", "before"]:
            # Return None instead of "XXXX-XX-XX" - this allows time_start to be None
            # The raw value should be preserved in time_start_raw by the caller
            return None

        # Handle "yesterday", "today", "tomorrow"
        if v_lower in ["yesterday", "today", "tomorrow"]:
            raise ValueError(f"Relative date '{v}' must be resolved to absolute date before validation. Use indexer._normalize_relative_time()")

        # CRITICAL FIX: Try to parse natural language dates like "January 2023", "19 January, 2023"
        # Use dateutil.parser if available, otherwise use enhanced regex
        try:
            from dateutil import parser as date_parser
            try:
                parsed_date = date_parser.parse(v, fuzzy=True, default=None)
                if parsed_date:
                    # Normalize to ISO format
                    # For partial dates like "January 2023", use first day of month
                    if len(v.split()) == 2 and any(month in v_lower for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']):
                        # Year-month only: return YYYY-MM format
                        return parsed_date.strftime("%Y-%m")
                    else:
                        # Full date: return YYYY-MM-DD format
                        return parsed_date.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass
        except ImportError:
            # Fallback: Enhanced regex for common patterns
            # Pattern: "January 2023" or "19 January, 2023" or "January 19, 2023"
            month_names = {
                'january': '01', 'february': '02', 'march': '03', 'april': '04',
                'may': '05', 'june': '06', 'july': '07', 'august': '08',
                'september': '09', 'october': '10', 'november': '11', 'december': '12'
            }
            
            import re
            # Pattern: "Month Year" (e.g., "January 2023")
            month_year_pattern = r'(\w+)\s+(\d{4})'
            match = re.search(month_year_pattern, v_lower, re.IGNORECASE)
            if match:
                month_str, year_str = match.groups()
                month_num = month_names.get(month_str.lower())
                if month_num and 1900 <= int(year_str) <= 2100:
                    # Return YYYY-MM format for partial dates
                    return f"{year_str}-{month_num}"
            
            # Pattern: "DD Month, YYYY" or "Month DD, YYYY"
            date_patterns = [
                r'(\d{1,2})\s+(\w+),?\s+(\d{4})',  # "19 January, 2023"
                r'(\w+)\s+(\d{1,2}),?\s+(\d{4})',   # "January 19, 2023"
            ]
            for pattern in date_patterns:
                match = re.search(pattern, v_lower, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if len(groups) == 3:
                        # Try to identify which is day and which is month
                        if groups[0].isdigit():
                            day, month_str, year = groups[0], groups[1], groups[2]
                        else:
                            month_str, day, year = groups[0], groups[1], groups[2]
                        
                        month_num = month_names.get(month_str.lower())
                        if month_num and 1900 <= int(year) <= 2100 and 1 <= int(day) <= 31:
                            return f"{year}-{month_num}-{day:02d}"

        # Handle partial dates with XX placeholders
        if 'XX' in v.upper():
            parts = v.split('-')

            # XXXX-XX-XX (completely unknown)
            if len(parts) == 3 and parts[0].upper() == 'XXXX' and parts[1].upper() == 'XX' and parts[2].upper() == 'XX':
                return v.upper()

            # YYYY-XX-XX (month and day unknown)
            if len(parts) == 3 and parts[0].isdigit() and parts[1].upper() == 'XX' and parts[2].upper() == 'XX':
                year = int(parts[0])
                if 1900 <= year <= 2100:
                    return v.upper()

            # YYYY-MM-XX (day unknown)
            elif len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].upper() == 'XX':
                year, month = int(parts[0]), int(parts[1])
                if 1900 <= year <= 2100 and 1 <= month <= 12:
                    return v.upper()

            # YYYY-XX (month unknown) - normalize to YYYY-01-XX
            elif len(parts) == 2 and parts[0].isdigit() and parts[1].upper() == 'XX':
                    year = int(parts[0])
                    if 1900 <= year <= 2100:
                        return f"{year}-01-XX"

        # Year only: YYYY → normalize to YYYY-01-01
        if re.match(r'^\d{4}$', v):
            year = int(v)
            if 1900 <= year <= 2100:
                return f"{year}-01-01"

        # Year-month only: YYYY-MM → keep as is
        if re.match(r'^\d{4}-\d{2}$', v):
            try:
                datetime.strptime(v, "%Y-%m")
                return v
            except ValueError:
                pass

        # Full ISO format dates
        if 'T' in v:
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
                return v
            except ValueError:
                pass
        else:
            # Date only: YYYY-MM-DD
            try:
                datetime.strptime(v, "%Y-%m-%d")
                return v
            except ValueError:
                pass

        # If we got here, validation failed
        raise ValueError(
            f'Invalid ISO-8601 datetime format: {v}. '
            f'Supported formats: YYYY-MM-DD, YYYY-MM-XX, YYYY-XX-XX, YYYY-XX, YYYY-MM, YYYY, or YYYY-MM-DDTHH:MM:SS'
        )

    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str, info: ValidationInfo) -> str:
        """Validate ID format"""
        if not v or not isinstance(v, str):
            raise ValueError('ID must be a non-empty string')
        return v.strip()

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation (ensure at least one timestamp is provided)
        
        CRITICAL FIX: Allow Event without time_start/time_end if time_start_raw exists.
        This supports fuzzy time expressions like "recently" that cannot be normalized
        to ISO format but should still be preserved.
        """
        super().model_post_init(__context)
        # Allow Event without time_start/time_end if time_start_raw exists (fuzzy time)
        if not self.time_start and not self.time_end and not self.time_start_raw:
            raise ValueError(
                'Event must have at least one timestamp (time_start, time_end, or time_start_raw)'
            )

    def overlaps_with(self, start: str, end: str) -> bool:
        """Check if this event overlaps with a time range"""
        event_start = self.time_start
        event_end = self.time_end or self.time_start

        if not event_start:
            return False

        return event_start <= end and event_end >= start

    model_config = {
        "validate_assignment": True,
        "validate_default": True,
    }

class Claim(BaseNode):
    """Claim node for subjective opinions, evaluations, and beliefs (v3.3 Enhanced)

    Claim represents subjective statements made by one person about another.
    Architecture: Source --[CLAIMS]--> Claim --[ABOUT]--> Target

    Version and lifecycle (see docs/MEMORY_LIFECYCLE.md):
    - Canonical state: SUPERSEDES + is_archived. is_latest=True only for chain head.
    - memory_lifecycle: stable by default; archived when superseded by ClaimReconciler.
    """
    id: str = Field(description="Claim ID with 'claim_' prefix")
    source_id: str = Field(description="Person ID who made the claim (e.g., 'p_jon')")
    target_id: str = Field(description="Person or Concept ID the claim is about (e.g., 'p_gina' or 'c_dancing')")
    content: str = Field(description="Specific evaluation content (e.g., 'is lazy', 'should work harder') - PRIMARY searchable field")
    sentiment: Optional[Literal["positive", "negative", "neutral"]] = Field(None, description="Sentiment classification")
    evidence_span: Optional[Dict[str, Union[str, int]]] = Field(None, description="{text, char_start, char_end}", exclude=True)
    confidence: float = Field(default=1.0, description="Confidence score (0-1)")
    session_time: Optional[str] = Field(None, description="Session time in ISO format (YYYY-MM-DD) for temporal reasoning")
    # v3.3: Temporal chain management
    is_latest: bool = Field(default=True, description="Whether this is the latest version of this claim")
    supersedes: List[str] = Field(default_factory=list, description="IDs of claims this one supersedes")

    @field_validator('source_id')
    @classmethod
    def validate_source_id(cls, v: str, info: ValidationInfo) -> str:
        """Validate that source_id is a valid person ID"""
        if not v or not isinstance(v, str):
            raise ValueError('source_id must be a non-empty string')
        if not (v.startswith('p_') or v.startswith('person_')):
            raise ValueError(
                f'source_id "{v}" must refer to a Person node (must start with "p_" or "person_"). '
                f'Claims are made by people, not concepts or entities.'
            )
        return v.strip()

    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str, info: ValidationInfo) -> str:
        """Validate that content is not empty and is descriptive"""
        if not v or not isinstance(v, str):
            raise ValueError('content must be a non-empty string')
        v = v.strip()
        if len(v) < 3:
            raise ValueError(f'content "{v}" is too short (minimum 3 characters)')
        return v

    @property
    def searchable_text(self) -> str:
        """Generate text for vector embedding (v3.3 Enhancement)
        
        Format: "[source] claims that [target] [content]"
        """
        source_name = self.source_id.replace('p_', '').replace('_', ' ')
        target_name = self.target_id.replace('p_', '').replace('c_', '').replace('_', ' ')
        return f"{source_name} claims that {target_name} {self.content}"

    def model_post_init(self, __context: Any) -> None:
        """Auto-calculate sentiment from content if not provided"""
        super().model_post_init(__context)
        if self.sentiment is None and self.content:
            negative_words = ['lazy', 'bad', 'worse', 'worst', 'hate', 'stupid', 'dumb', 'slow', 'incompetent', 'useless']
            positive_words = ['great', 'good', 'excellent', 'amazing', 'smart', 'talented', 'skilled', 'helpful', 'best', 'awesome']

            content_lower = self.content.lower()
            neg_count = sum(1 for word in negative_words if word in content_lower)
            pos_count = sum(1 for word in positive_words if word in content_lower)

            if neg_count > pos_count:
                self.sentiment = "negative"
            elif pos_count > neg_count:
                self.sentiment = "positive"
            else:
                self.sentiment = "neutral"

    model_config = {
        "validate_assignment": True,
        "validate_default": True,
    }

class Fact(BaseNode):
    """Fact node with subject-predicate-object structure (v3.3 Enhanced)

    Requirements:
    - subject_id must refer to a Person (person_ prefix)
    - predicate should be atomic and descriptive
    - object_id or purpose_id must refer to a Concept
    - **CRITICAL**: At least one of 'object_id' or 'purpose_id' must be provided
    
    Version and lifecycle (see docs/MEMORY_LIFECYCLE.md):
    - Canonical state: SUPERSEDES + is_archived + valid_to. is_latest=True only for chain head.
    - memory_lifecycle: candidate (from indexing) -> stable (after consolidation) -> archived (when superseded).
    """
    id: str = Field(description="Fact ID with 'fact_' prefix")
    subject_id: str = Field(description="Person ID (e.g., 'person_jon' or 'p_jon')")
    predicate: str = Field(description="Atomic predicate (e.g., 'is_a', 'likes', 'uses')")
    object_id: Optional[str] = Field(None, description="Concept ID (e.g., 'c_dancing')")
    purpose_id: Optional[str] = Field(None, description="Purpose concept ID for strategy facts (e.g., 'c_destress')")
    confidence: float = Field(default=1.0, description="Confidence score (0-1) for objective facts")
    valid_from: Optional[str] = Field(None, description="ISO-8601 start time for validity")
    valid_to: Optional[str] = Field(None, description="ISO-8601 end time for validity")
    evidence_span: Optional[Dict[str, Union[str, int]]] = Field(None, description="{text, char_start, char_end}", exclude=True)
    session_time: Optional[str] = Field(None, description="Session time in ISO format (YYYY-MM-DD) for temporal reasoning")
    # v3.3: Temporal chain management
    is_latest: bool = Field(default=True, description="Whether this is the latest version of this fact")
    supersedes: List[str] = Field(default_factory=list, description="IDs of facts this one supersedes")

    @property
    def is_temporal(self) -> bool:
        """Check if this fact has temporal validity"""
        return bool(self.valid_from or self.valid_to)

    @property
    def is_current(self) -> bool:
        """Check if this fact is currently valid"""
        from datetime import datetime
        if not self.valid_to:
            return True
        try:
            valid_to_dt = datetime.fromisoformat(self.valid_to.replace('Z', '+00:00'))
            return valid_to_dt >= datetime.now(valid_to_dt.tzinfo)
        except:
            return True

    @property
    def searchable_text(self) -> str:
        """Generate text for vector embedding (v3.3 Enhancement)
        
        Priority: content > constructed SPO sentence
        The `content` field should contain a natural language description.
        """
        if self.content:
            return self.content
        # Fallback: construct from SPO
        predicate_natural = self.predicate.replace('_', ' ')
        object_natural = (self.object_id or self.purpose_id or "").replace('c_', '').replace('_', ' ')
        return f"{self.subject_id} {predicate_natural} {object_natural}"

    @field_validator('subject_id')
    @classmethod
    def validate_subject(cls, v: str, info: ValidationInfo) -> str:
        """Validate that subject is a valid person ID"""
        if not v or not isinstance(v, str):
            raise ValueError('subject_id must be a non-empty string')
        return v.strip()

    @field_validator('object_id', 'purpose_id', mode='before')
    @classmethod
    def validate_concept_reference(cls, v: Any, info: ValidationInfo) -> Any:
        """Validate that object_id and purpose_id are valid concept references"""
        if v is None:
            return v
        
        # Handle list type (LLM may incorrectly return a list)
        if isinstance(v, list):
            if len(v) == 1:
                v = v[0]
            elif len(v) > 1:
                v = ", ".join(str(item) for item in v)
            else:
                return None

        if isinstance(v, str):
            return v.strip()

        return v

    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str, info: ValidationInfo) -> str:
        """Validate ID format"""
        if not v or not isinstance(v, str):
            raise ValueError('ID must be a non-empty string')
        return v.strip()

    @field_validator('valid_from', 'valid_to')
    @classmethod
    def validate_temporal_range(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Validate ISO-8601 datetime format for temporal fields
        
        Supports:
        - Full datetime: YYYY-MM-DDTHH:MM:SS
        - Date only: YYYY-MM-DD
        - Year-month: YYYY-MM (normalized to YYYY-MM-01)
        - None: allowed for optional fields
        """
        if v is None:
            return v

        from datetime import datetime
        import re
        
        v = str(v).strip()
        
        # Handle "YYYY-MM" format (year-month only)
        year_month_match = re.match(r'^(\d{4})-(\d{2})$', v)
        if year_month_match:
            year, month = year_month_match.groups()
            # Validate month and return normalized format
            if 1 <= int(month) <= 12:
                return f"{year}-{month}-01"  # Normalize to first day of month
        
        # Handle full datetime format
        if 'T' in v:
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
                return v
            except ValueError:
                raise ValueError(f'Invalid ISO-8601 format: {v}')
        
        # Handle date-only format (YYYY-MM-DD)
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError(f'Invalid ISO-8601 format: {v}')

    @model_validator(mode='after')
    def validate_object_or_purpose(self) -> 'Fact':
        """Validate that at least one of 'object_id' or 'purpose_id' is provided"""
        if self.object_id is None and self.purpose_id is None:
            raise ValueError(
                f"Fact '{self.id}' must have either 'object_id' or 'purpose_id' (both cannot be empty). "
                f"Subject: '{self.subject_id}', Predicate: '{self.predicate}'"
            )
        return self

    model_config = {
        "validate_assignment": True,
        "validate_default": True,
    }

# ============================================================================
# Supporting Models
# ============================================================================

class Relation(BaseModel):
    """Graph edge/relation"""
    head_id: str
    tail_id: str
    rel_type: EdgeType = "related"
    weight: float = 1.0
    confidence: float = 1.0
    metadata: Metadata = Field(default_factory=Metadata)
    version: int = 1
    timestamp: Optional[str] = None  # ISO-8601 format for time-aware edges
    edge_title: Optional[str] = None  # Event title or relation description
    edge_time: Optional[str] = None  # Event time for edge-based filtering

class Doc(BaseModel):
    """Document node"""
    id: str
    text: str
    metadata: Metadata = Field(default_factory=Metadata)

class RetrievalResult(BaseModel):
    """Retrieval result for backward compatibility"""
    query: str
    entities: List[Any] = Field(default_factory=list)  # Changed from Entity to Any
    evidence: List[str] = Field(default_factory=list)
    extracted_entity_names: List[str] = Field(default_factory=list)
    seed_node_ids: List[str] = Field(default_factory=list)
    retrieval_path: List[str] = Field(default_factory=list)
    candidates: List[Any] = Field(default_factory=list)
    reranked_candidates: List[Any] = Field(default_factory=list)
    confidence: float = Field(default=0.0, description="Confidence score for the retrieval result")
