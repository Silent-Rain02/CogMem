from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional, Union, Tuple, Any
from pydantic import BaseModel, Field, field_validator, ValidationInfo


# ============================================================================
# Node Type Enumeration (Refactored for Strong Typing)
# ============================================================================

class NodeTypeEnum(str, Enum):
    """Strongly-typed node types for the knowledge graph

    All node types in the graph must use one of these enum values.
    This ensures type safety and prevents typos in node type strings.
    """
    EVENT = "event"           # Time-bounded happenings with participants
    FACT = "fact"             # Subject-predicate-object knowledge statements
    EPISODE = "episode"       # Narrative episodes or story segments
    TOPIC = "topic"           # Thematic topics or subject areas
    PERSON = "person"         # People/characters
    ENTITY = "entity"         # General entities (organizations, locations, objects)
    PREFERENCE = "preference" # Likes/dislikes/preferences
    RELATION = "relation"     # Edges/relationships (not typically used as node_type)
    SESSION = "session"       # Conversation sessions
    TURN = "turn"            # Dialogue turns

    def __str__(self) -> str:
        return self.value

    @classmethod
    def get_prefix(cls, node_type: 'NodeTypeEnum') -> str:
        """Get the ID prefix for a given node type"""
        return f"{node_type.value}_"


# ============================================================================
# Node Type Definitions: Major Entities vs Minor Concepts (Phase 3)
# ============================================================================

# MAJOR ENTITIES: Primary retrieval entry points (these are searchable entities)
MajorEntityType = Literal[
    "Person",           # People (e.g., "Caroline", "John")
    "Organization",     # Organizations (e.g., "Google", "Red Cross")
    "Location",         # Named locations (e.g., "San Francisco", "Central Park")
    "NamedObject"       # Named objects (e.g., "Tesla Model S", "MacBook Pro")
]

# MINOR CONCEPTS: Connection points and metadata (not standalone searchable entities)
MinorConceptType = Literal[
    "Event",            # Events (e.g., "birthday party", "marathon")
    "Fact",             # Fact statements (subject-predicate-object)
    "Preference",       # Preferences/likes/dislikes
    "TimeNormalization", # Time expressions
    "EvidenceSpan"      # Evidence text spans
]

# All node types combined
NodeType = Union[MajorEntityType, MinorConceptType, Literal[
    "Session", "Turn", "Summary", "Observation", "Question", "PersonFacet", "Entity"
]]

# Entity types (for abstract categorization)
EntityType = Literal["preference", "world"]

# Entity subtypes (for more granular categorization)
EntitySubtype = Literal["person", "location", "temporal", "event", "organization", "concept", "preference"]

# Community types
CommunityKind = Literal["structure", "semantic"]

# Edge types with semantic meaning
EdgeType = Literal[
    "INVOLVED_IN", "ASSERTS", "SUPERSEDES", "CONTRADICTS", "SAME_AS", "ALIAS_OF",
    "BELONGS_TO", "SPOKEN_BY", "MENTIONS", "SUPPORTS_ANSWER", "EVIDENCE_IN",
    "AT_TIME", "BEFORE", "AFTER", "related"  # Keep "related" for backward compatibility
]

# Forbidden entity names - these should NOT be created as standalone nodes (Phase 3)
# These are generic concepts that should be edge properties or fact predicates
FORBIDDEN_ENTITY_NAMES = {
    # Common generic nouns that should not be entities
    "family", "job", "work", "career", "feeling", "emotion",
    "thought", "idea", "plan", "goal", "dream", "hope",
    "problem", "issue", "concern", "worry", "fear", "desire",
    "relationship", "friendship", "love", "hate", "like", "dislike",
    "need", "want", "wish", "preference", "choice", "decision",
}

FORBIDDEN_ENTITY_PREFIXES = {
    "my ", "his ", "her ", "their ", "our ", "your ", "the ", "a ", "an "
}


# ============================================================================
# Metadata Model (Moved before BaseNode to avoid forward reference)
# ============================================================================

class Metadata(BaseModel):
    source: Optional[str] = None
    created_at: Optional[str] = None  # YYYY-MM-DD HH:MM:SS 格式，可读时间戳
    updated_at: Optional[str] = None  # YYYY-MM-DD HH:MM:SS 格式，可读时间戳
    source_session_id: Optional[str] = None
    source_turn_id: Optional[str] = None


# ============================================================================
# Base Node Model (Refactored Core Structure)
# ============================================================================

class BaseNode(BaseModel):
    """Base model for all knowledge graph nodes

    Provides common fields and validation for all node types:
    - id: Unique identifier with type prefix (e.g., "person_", "event_")
    - content: Core textual content/description
    - timestamp: Optional temporal marker
    - topic_id: Optional topic classification
    - is_archived: Archival status flag
    - metadata: Creation and update metadata
    """
    id: str = Field(description="Unique node ID with type prefix (e.g., 'person_abc123')")
    content: str = Field(default="", description="Core textual content or description")
    timestamp: Optional[str] = Field(None, description="ISO-8601 timestamp for temporal nodes")
    topic_id: Optional[str] = Field(None, description="Topic classification ID")
    is_archived: bool = Field(default=False, description="Whether this node is archived")
    metadata: Metadata = Field(default_factory=Metadata, description="Creation and update metadata")

    @field_validator('id')
    @classmethod
    def validate_id_format(cls, v: str, info: ValidationInfo) -> str:
        """Validate that ID has correct format with node type prefix"""
        if not v or not isinstance(v, str):
            raise ValueError('ID must be a non-empty string')

        if '_' not in v:
            raise ValueError(
                f'ID "{v}" must include a node type prefix (e.g., "person_", "event_"). '
                f'Use NodeType.get_prefix(NodeType.PERSON) to generate proper IDs.'
            )

        # Extract prefix and validate against NodeTypeEnum
        prefix = v.split('_')[0]
        valid_prefixes = {nt.value for nt in NodeTypeEnum}

        if prefix not in valid_prefixes:
            raise ValueError(
                f'ID prefix "{prefix}" is not a valid NodeType. '
                f'Valid prefixes: {", ".join(valid_prefixes)}'
            )

        return v

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Validate ISO-8601 timestamp format"""
        if v is None:
            return v

        from datetime import datetime
        try:
            if 'T' in v:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            else:
                datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError(
                f'Invalid ISO-8601 timestamp format: {v}. '
                f'Expected: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS'
            )

    model_config = {
        "validate_assignment": True,
        "validate_default": True,
        "use_enum_values": True,  # Use enum values in serialization
    }


class Entity(BaseNode):
    """Entity node representing a named entity in the knowledge graph

    Inherits from BaseNode and adds entity-specific fields like name, type, subtype,
    and relationship tracking.
    """
    name: str = Field(description="Entity name (e.g., 'Google', 'New York')")
    type: EntityType = Field(default="world", description="Abstract entity type: preference or world")
    subtype: Optional[EntitySubtype] = Field(default=None, description="Concrete entity subtype")
    related_entity_ids: List[str] = Field(default_factory=list, description="IDs of related entities")
    community_ids: List[str] = Field(default_factory=list, description="Community memberships")
    version: int = Field(default=1, description="Version number for tracking updates")
    # Extended fields for person-centered graph
    aliases: List[str] = Field(default_factory=list, description="Alternative names/aliases")
    persona_traits: List[str] = Field(default_factory=list, description="Personality traits (for person subtype)")
    valid_from: Optional[str] = Field(None, description="ISO-8601 start time for temporal validity")
    valid_to: Optional[str] = Field(None, description="ISO-8601 end time for temporal validity")
    time_interval: Optional[Dict[str, Optional[str]]] = Field(None, description="{start, end, granularity}")
    confidence: float = Field(default=1.0, description="Confidence score (0-1)")
    salience: float = Field(default=0.0, description="Salience/importance score")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")


class Relation(BaseModel):
    head_id: str
    tail_id: str
    rel_type: str = "related"  # Can be EdgeType
    weight: float = 1.0
    confidence: float = 1.0
    metadata: Metadata = Field(default_factory=Metadata)
    version: int = 1
    timestamp: Optional[str] = None  # ISO-8601 format for time-aware edges


class Community(BaseModel):
    id: str
    kind: CommunityKind
    label: Optional[str] = None
    score: Optional[float] = None
    member_ids: List[str] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=Metadata)


class Doc(BaseModel):
    id: str
    text: str
    metadata: Metadata = Field(default_factory=Metadata)


class RetrievalResult(BaseModel):
    query: str
    entities: List[Entity] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    extracted_entity_names: List[str] = Field(default_factory=list)  # Entity names extracted from query (before graph lookup)
    seed_node_ids: List[str] = Field(default_factory=list)  # Seed node IDs identified by hybrid retrieval (for debugging)
    retrieval_path: List[str] = Field(default_factory=list)  # Retrieval strategy steps used (e.g., "entity-extraction", "exact-match", "community-expansion", "graph-traversal", "query-rerank", "chunk-search")
    candidates: List[Any] = Field(default_factory=list)  # (node_id, node_type, node_obj, graph_distance)
    reranked_candidates: List[Any] = Field(default_factory=list)  # (node_id, node_type, node_obj, score, breakdown)


# Rebuild the model to handle forward references
RetrievalResult.model_rebuild()


# New node types for person-centered graph

class Person(BaseNode):
    """Person node with rich descriptive information

    Represents a person/character in the knowledge graph with attributes like
    name, profession, background, and personality traits.
    """
    id: str = Field(description="Person ID with 'person_' prefix")
    canonical_name: str = Field(description="Primary name of the person")
    aliases: List[str] = Field(default_factory=list, description="Alternative names/nicknames")
    persona_traits: List[str] = Field(default_factory=list, description="Personality traits and characteristics")
    profession: str = Field(default="", description="Current job or career field")
    background: str = Field(default="", description="Key background information")
    identity: str = Field(default="", description="Important identity aspects")
    confidence: float = Field(default=1.0, description="Confidence score (0-1)")
    evidence_span: Optional[Dict[str, Union[str, int]]] = Field(None, description="{text, char_start, char_end}")
    source_session_id: Optional[str] = Field(None, description="Source session ID")
    source_turn_id: Optional[str] = Field(None, description="Source turn ID")
    community_ids: List[str] = Field(default_factory=list, description="Community memberships")

    @field_validator('canonical_name')
    @classmethod
    def validate_canonical_name(cls, v: str, info: ValidationInfo) -> str:
        """Validate that canonical_name is not a generic concept (Phase 3)"""
        if not v or not isinstance(v, str):
            raise ValueError('canonical_name must be a non-empty string')

        name_lower = v.lower().strip()

        # Check if it's a forbidden generic concept
        if name_lower in FORBIDDEN_ENTITY_NAMES:
            raise ValueError(
                f'Entity name "{v}" is a generic concept and should not be created as a standalone node. '
                f'This should be a fact predicate or edge attribute instead.'
            )

        # Check for forbidden prefixes (possessive/generic)
        for prefix in FORBIDDEN_ENTITY_PREFIXES:
            if name_lower.startswith(prefix):
                raise ValueError(
                    f'Entity name "{v}" starts with generic prefix "{prefix}". '
                    f'Use specific names instead of "{prefix}something".'
                )

        # Ensure it's a proper name (capitalized, not just common words)
        if len(name_lower.split()) <= 2 and not any(c.isupper() for c in v):
            # Allow if it's a single name or short name
            pass

        return v.strip()

    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str, info: ValidationInfo) -> str:
        """Validate ID format"""
        if not v.startswith('person_'):
            raise ValueError(f'Person ID must start with "person_", got: {v}')
        return v

    model_config = {
        "validate_assignment": True,  # Enable validation on attribute assignment
        "validate_default": True,      # Enable validation of default values
    }


class Event(BaseNode):
    """Event node with participants and time interval

    Temporal requirements (Phase 3):
    - time_start or time_end must be provided and be valid ISO-8601 format
    - timestamp property provides easier access to primary time point
    """
    id: str = Field(description="Event ID with 'event_' prefix")
    title: str = Field(description="Event title/summary")
    participants: List[str] = Field(default_factory=list, description="Person/Entity IDs involved")
    location: Optional[str] = Field(None, description="Event location")
    time_start: Optional[str] = Field(None, description="ISO-8601 start time (at least one of time_start/time_end required)")
    time_end: Optional[str] = Field(None, description="ISO-8601 end time")
    granularity: Optional[Literal["year", "month", "day", "time"]] = Field(None, description="Time granularity")
    time_confidence: float = Field(default=1.0, description="Time confidence score (0-1)")
    temporal_conflict: bool = Field(default=False, description="Whether temporal data has conflicts")
    evidence_span: Optional[Dict[str, Union[str, int]]] = Field(None, description="{text, char_start, char_end}")
    source_session_id: Optional[str] = Field(None, description="Source session ID")
    source_turn_id: Optional[str] = Field(None, description="Source turn ID")
    community_ids: List[str] = Field(default_factory=list, description="Community memberships")

    @property
    def timestamp(self) -> Optional[str]:
        """Primary timestamp for the event (time_start if available, otherwise time_end)"""
        return self.time_start or self.time_end

    @field_validator('time_start', 'time_end')
    @classmethod
    def validate_iso_datetime(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Validate ISO-8601 datetime format for temporal fields"""
        if v is None:
            return v

        # Try to parse as ISO datetime
        from datetime import datetime
        try:
            # Support various ISO formats
            if 'T' in v:
                # Full datetime: YYYY-MM-DDTHH:MM:SS
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            else:
                # Date only: YYYY-MM-DD
                datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError(
                f'Invalid ISO-8601 datetime format: {v}. '
                f'Expected format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS'
            )

    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str, info: ValidationInfo) -> str:
        """Validate ID format"""
        if not v.startswith('event_'):
            raise ValueError(f'Event ID must start with "event_", got: {v}')
        return v

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation (ensure at least one timestamp is provided)"""
        super().model_post_init(__context)
        # Validate that at least one timestamp is provided
        if not self.time_start and not self.time_end:
            raise ValueError(
                'Event must have at least one timestamp (time_start or time_end)'
            )

    def overlaps_with(self, start: str, end: str) -> bool:
        """Check if this event overlaps with a time range

        Args:
            start: Start time of the range (ISO-8601)
            end: End time of the range (ISO-8601)

        Returns:
            True if the event overlaps with the time range
        """
        event_start = self.time_start
        event_end = self.time_end or self.time_start

        if not event_start:
            return False

        # Event starts before range ends AND event ends after range starts
        return event_start <= end and event_end >= start

    model_config = {
        "validate_assignment": True,
        "validate_default": True,
    }


class Fact(BaseNode):
    """Fact node with subject-predicate-object structure

    Phase 3 requirements:
    - subject must refer to an existing Person entity (person_ prefix)
    - predicate should be atomic and descriptive
    - object can refer to another Entity or be a literal value
    - value should not be generic concepts (family, job, feeling, etc.)
    """
    id: str = Field(description="Fact ID with 'fact_' prefix")
    subject: str = Field(description="Person/Entity ID (must start with 'person_' or 'entity_')")
    predicate: str = Field(description="Atomic predicate (e.g., 'is', 'has', 'works_at', 'lives_in')")
    object: Optional[str] = Field(None, description="Named entity ID or literal value")
    value: Optional[Union[str, int, float, bool]] = Field(None, description="Literal value (not generic concepts)")
    value_type: Optional[str] = Field(None, description="Type hint: string|int|bool|date")
    subtype: Optional[Literal["trait", "action", "feeling", "state", "relationship", "possession", "location", "other"]] = Field(None, description="Fact subtype")
    valid_from: Optional[str] = Field(None, description="ISO-8601 start time for validity")
    valid_to: Optional[str] = Field(None, description="ISO-8601 end time for validity")
    evidence_span: Optional[Dict[str, Union[str, int]]] = Field(None, description="{text, char_start, char_end}")
    source_session_id: Optional[str] = Field(None, description="Source session ID")
    source_turn_id: Optional[str] = Field(None, description="Source turn ID")
    confidence: float = Field(default=1.0, description="Confidence score (0-1)")
    supersedes: List[str] = Field(default_factory=list, description="IDs of superseded facts")

    @property
    def is_temporal(self) -> bool:
        """Check if this fact has temporal validity"""
        return bool(self.valid_from or self.valid_to)

    @property
    def is_current(self) -> bool:
        """Check if this fact is currently valid (no valid_to or valid_to in future)"""
        from datetime import datetime
        if not self.valid_to:
            return True
        try:
            valid_to_dt = datetime.fromisoformat(self.valid_to.replace('Z', '+00:00'))
            return valid_to_dt >= datetime.now(valid_to_dt.tzinfo)
        except:
            return True  # If can't parse, assume valid

    @field_validator('subject')
    @classmethod
    def validate_subject(cls, v: str, info: ValidationInfo) -> str:
        """Validate that subject is a valid entity ID (not a generic concept)"""
        if not v or not isinstance(v, str):
            raise ValueError('subject must be a non-empty string')

        subject_lower = v.lower()

        # Must be a properly prefixed ID
        if not (subject_lower.startswith('person_') or subject_lower.startswith('entity_')):
            raise ValueError(
                f'Fact subject "{v}" must refer to a Person or Entity (must start with "person_" or "entity_"). '
                f'Generic concepts should not be used as subjects.'
            )

        return v

    @field_validator('object', 'value', mode='before')
    @classmethod
    def validate_no_generic_concepts(cls, v: Any, info: ValidationInfo) -> Any:
        """Validate that object and value are not generic concepts"""
        if v is None:
            return v

        if isinstance(v, str):
            v_lower = v.lower().strip()

            # Check if it's a forbidden generic concept
            if v_lower in FORBIDDEN_ENTITY_NAMES:
                raise ValueError(
                    f'Fact object/value "{v}" is a generic concept and should not be used. '
                    f'Generic concepts like "{v}" should be predicates, not values.'
                )

            # Check for generic prefixes
            for prefix in FORBIDDEN_ENTITY_PREFIXES:
                if v_lower.startswith(prefix):
                    raise ValueError(
                        f'Fact object/value "{v}" starts with generic prefix "{prefix}". '
                        f'Use specific named entities instead.'
                    )

        return v

    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str, info: ValidationInfo) -> str:
        """Validate ID format"""
        if not v.startswith('fact_'):
            raise ValueError(f'Fact ID must start with "fact_", got: {v}')
        return v

    @field_validator('valid_from', 'valid_to')
    @classmethod
    def validate_temporal_range(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Validate ISO-8601 datetime format for temporal fields"""
        if v is None:
            return v

        from datetime import datetime
        try:
            if 'T' in v:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            else:
                datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError(f'Invalid ISO-8601 format: {v}')

    model_config = {
        "validate_assignment": True,
        "validate_default": True,
    }


class Preference(BaseNode):
    """Preference node for likes/dislikes"""
    id: str = Field(description="Preference ID with 'preference_' prefix")
    holder: str = Field(description="Person ID who holds this preference")
    aspect: str = Field(description="Preference aspect (e.g., food/music/brand/activity)")
    polarity: Literal["like", "dislike", "neutral"] = Field(description="Like/dislike/neutral polarity")
    strength: float = Field(default=0.5, description="Preference strength 0-1")
    valid_from: Optional[str] = Field(None, description="ISO-8601 start time for validity")
    valid_to: Optional[str] = Field(None, description="ISO-8601 end time for validity")
    evidence_span: Optional[Dict[str, Union[str, int]]] = Field(None, description="{text, char_start, char_end}")
    source_session_id: Optional[str] = Field(None, description="Source session ID")
    source_turn_id: Optional[str] = Field(None, description="Source turn ID")
    confidence: float = Field(default=1.0, description="Confidence score (0-1)")
    supersedes: List[str] = Field(default_factory=list, description="IDs of superseded preferences")


class Session(BaseModel):
    """Session node representing a conversation session"""
    id: str
    session_id: str
    date_time: Optional[str] = None  # ISO-8601
    text: Optional[str] = None  # Aggregated text
    embedding: Optional[List[float]] = None
    community_ids: List[str] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=Metadata)


class Turn(BaseModel):
    """Turn node representing a single dialogue turn"""
    id: str
    turn_id: str
    speaker: Optional[str] = None  # Person ID or role
    dia_id: Optional[str] = None
    text: str
    img_url: Optional[str] = None
    blip_caption: Optional[str] = None
    source_session_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Metadata = Field(default_factory=Metadata)


class Summary(BaseModel):
    """Summary node for session or community summaries"""
    id: str
    scope: Literal["session", "community"]
    text: str
    session_id: Optional[str] = None
    community_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Metadata = Field(default_factory=Metadata)


class Observation(BaseModel):
    """Observation node from LoCoMo"""
    id: str
    text: str
    turn_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Metadata = Field(default_factory=Metadata)


class Question(BaseModel):
    """Question node"""
    id: str
    question_id: str
    question: str
    type: Optional[str] = None
    date: Optional[str] = None  # ISO-8601
    embedding: Optional[List[float]] = None
    metadata: Metadata = Field(default_factory=Metadata)


class EvidenceSpan(BaseModel):
    """Evidence span for alignment with LoCoMo evaluation"""
    id: str
    turn_id: str
    char_start: int = 0
    char_end: int = 0
    text: str
    has_answer: bool = False  # For LongMemEval alignment
    metadata: Metadata = Field(default_factory=Metadata)


class TimeNormalization(BaseModel):
    """Time normalization result"""
    id: str
    original_text: str
    time_start: Optional[str] = None  # ISO-8601
    time_end: Optional[str] = None  # ISO-8601
    granularity: Optional[Literal["year", "month", "day", "time"]] = None
    confidence: float = 1.0
    anchor: Literal["question_date", "context_window", "unknown"] = "unknown"
    evidence_span: Optional[Dict[str, Union[str, int]]] = None
    metadata: Metadata = Field(default_factory=Metadata)


class PersonFacet(BaseModel):
    """PersonFacet node for partitioning person information"""
    id: str
    person_id: str  # Reference to Person node
    facet_type: Literal["work", "health", "finance", "relationships", "preferences", "tasks", "other"]
    description: str = ""
    community_ids: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    metadata: Metadata = Field(default_factory=Metadata)


# ============================================================================
# EpisodeNode for Time-based Event Grouping
# ============================================================================

class Episode(BaseNode):
    """Episode node representing a time-bounded narrative episode

    Episodes group Events that occur within the same time period (e.g., a week).
    This creates a hierarchical structure: Person -> Episode -> Event -> Fact/Preference

    Temporal grouping:
    - week_start: ISO-8601 date for the Monday of this week
    - week_end: ISO-8601 date for the Sunday of this week
    - year: Calendar year
    - week_number: ISO week number (1-52/53)

    Relationships:
    - Person PARTICIPATES_IN Episode (multiple people can participate)
    - Episode CONTAINS Event (multiple events per episode)
    - Episode CONTINUED_FROM previous_episode_id (for sequential episodes)
    """
    id: str = Field(description="Episode ID with 'episode_' prefix, format: episode_{year}_week{week_number}")
    week_start: str = Field(description="ISO-8601 date for Monday of this week (YYYY-MM-DD)")
    week_end: str = Field(description="ISO-8601 date for Sunday of this week (YYYY-MM-DD)")
    year: int = Field(description="Calendar year")
    week_number: int = Field(description="ISO week number (1-52 or 1-53)")
    participant_ids: List[str] = Field(default_factory=list, description="Person IDs participating in this episode")
    event_ids: List[str] = Field(default_factory=list, description="Event IDs contained in this episode")
    episode_type: Optional[Literal["normal", "important", "milestone"]] = Field(default="normal", description="Episode significance type")
    summary: Optional[str] = Field(None, description="Auto-generated summary of key events in this episode")
    previous_episode_id: Optional[str] = Field(None, description="Previous episode ID for sequential linking")

    @field_validator('id')
    @classmethod
    def validate_episode_id(cls, v: str, info: ValidationInfo) -> str:
        """Validate episode ID format: episode_{year}_week{week_number}"""
        if not v or not isinstance(v, str):
            raise ValueError('Episode ID must be a non-empty string')

        if not v.startswith('episode_'):
            raise ValueError(f'Episode ID must start with "episode_", got: {v}')

        # Check format: episode_{year}_week{week_number}
        parts = v.split('_')
        if len(parts) != 3 or parts[0] != 'episode' or parts[2][:4] != 'week':
            raise ValueError(
                f'Episode ID "{v}" must follow format: episode_{{year}}_week{{week_number}}. '
                f'Example: episode_2024_week15'
            )

        return v

    @field_validator('week_start', 'week_end')
    @classmethod
    def validate_iso_date(cls, v: str, info: ValidationInfo) -> str:
        """Validate ISO-8601 date format (YYYY-MM-DD)"""
        if not v:
            return v

        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError(
                f'Invalid ISO date format: {v}. '
                f'Expected: YYYY-MM-DD'
            )

    def add_participant(self, person_id: str) -> None:
        """Add a participant to this episode if not already present"""
        if person_id not in self.participant_ids:
            self.participant_ids.append(person_id)

    def add_event(self, event_id: str) -> None:
        """Add an event to this episode if not already present"""
        if event_id not in self.event_ids:
            self.event_ids.append(event_id)

    def generate_summary(self, events: List['Event']) -> str:
        """Generate summary from event titles"""
        if not events:
            return "No events in this episode"

        # Sort events by time
        sorted_events = sorted(events, key=lambda e: e.time_start or "")

        # Take first 3 event titles
        titles = [e.title for e in sorted_events[:3]]
        summary = ", ".join(titles)

        if len(sorted_events) > 3:
            summary += f" and {len(sorted_events) - 3} more events"

        return summary

    model_config = {
        "validate_assignment": True,
        "validate_default": True,
        "use_enum_values": True,
    }


# ============================================================================
# TopicNode for Semantic Clustering
# ============================================================================

class Topic(BaseNode):
    """Topic node representing a semantic cluster or thematic grouping

    Topics are generated from semantic clustering (K-means) of nodes based on embeddings.
    They provide high-level thematic organization of the knowledge graph.

    Relationships:
    - Topic COVERS Episode/Fact/Event/Entity (high-level thematic coverage)
    - Topic RELATED_TO Topic (for topic hierarchies)
    - Topic DERIVED_FROM Community (links back to the community that generated it)

    Attributes:
    - id: Topic ID with 'topic_' prefix, includes cluster ID
    - label: Human-readable topic label (e.g., "Work & Career", "Family Life")
    - description: Generated summary of what this topic covers
    - keywords: Key terms extracted from cluster members
    - member_ids: IDs of nodes in this topic cluster
    - centroid: Vector centroid of the cluster (for similarity calculations)
    - source_community_id: ID of the Community that generated this topic
    - confidence: Confidence score for the topic (based on cluster cohesion)
    """
    id: str = Field(description="Topic ID with 'topic_' prefix, format: topic_{semantic|structure}_{cluster_id}")
    label: str = Field(description="Human-readable topic label (e.g., 'Work & Career')")
    description: str = Field(description="Generated summary of what this topic covers")
    keywords: List[str] = Field(default_factory=list, description="Key terms extracted from cluster members")
    member_ids: List[str] = Field(default_factory=list, description="IDs of nodes in this topic cluster")
    centroid: Optional[List[float]] = Field(None, description="Vector centroid of the cluster")
    source_community_id: Optional[str] = Field(None, description="ID of the Community that generated this topic")
    confidence: float = Field(default=1.0, description="Confidence score (0-1) based on cluster cohesion")
    member_types: List[str] = Field(default_factory=list, description="Types of member nodes (e.g., ['Episode', 'Fact'])")
    size: int = Field(default=0, description="Number of members in this topic")

    @field_validator('id')
    @classmethod
    def validate_topic_id(cls, v: str, info: ValidationInfo) -> str:
        """Validate topic ID format: topic_{semantic|structure}_{cluster_id}"""
        if not v or not isinstance(v, str):
            raise ValueError('Topic ID must be a non-empty string')

        if not v.startswith('topic_'):
            raise ValueError(f'Topic ID must start with "topic_", got: {v}')

        parts = v.split('_')
        if len(parts) < 3:
            raise ValueError(
                f'Topic ID "{v}" must follow format: topic_{{semantic|structure}}_{{cluster_id}}. '
                f'Example: topic_semantic_work_career_0'
            )

        return v

    @field_validator('label')
    @classmethod
    def validate_label(cls, v: str, info: ValidationInfo) -> str:
        """Validate that label is meaningful and not empty"""
        if not v or not v.strip():
            raise ValueError('Topic label cannot be empty')

        if len(v) < 3:
            raise ValueError(f'Topic label "{v}" is too short (minimum 3 characters)')

        if len(v) > 100:
            raise ValueError(f'Topic label "{v}" is too long (maximum 100 characters)')

        return v.strip()

    @field_validator('member_ids')
    @classmethod
    def validate_member_ids(cls, v: List[str], info: ValidationInfo) -> List[str]:
        """Validate that member IDs have proper prefixes"""
        if not v:
            return v

        valid_prefixes = {nt.value for nt in NodeTypeEnum}
        invalid_ids = []

        for member_id in v:
            if '_' not in member_id:
                invalid_ids.append(member_id)
                continue

            prefix = member_id.split('_')[0]
            if prefix not in valid_prefixes:
                invalid_ids.append(member_id)

        if invalid_ids:
            raise ValueError(
                f'Topic member_ids contain invalid IDs: {invalid_ids[:5]}. '
                f'All member IDs must have valid NodeType prefixes.'
            )

        return v

    def add_member(self, member_id: str, member_type: Optional[str] = None) -> None:
        """Add a member to this topic"""
        if member_id not in self.member_ids:
            self.member_ids.append(member_id)
            if member_type and member_type not in self.member_types:
                self.member_types.append(member_type)
            self.size = len(self.member_ids)

    def remove_member(self, member_id: str) -> None:
        """Remove a member from this topic"""
        if member_id in self.member_ids:
            self.member_ids.remove(member_id)
            self.size = len(self.member_ids)

    model_config = {
        "validate_assignment": True,
        "validate_default": True,
        "use_enum_values": True,
    }


# ============================================================================
# EventNode for Structured Extraction (Refactored from free-text extraction)
# ============================================================================

class EventNode(BaseModel):
    """EventNode for structured event extraction from text.

    This model replaces free-text event extraction with structured fields:
    - summary: Brief event title/summary
    - description: Detailed event description
    - timestamp: When the event occurred (ISO-8601 format)
    - primary_participants: List of main participants (Person/Entity IDs)

    Unlike the Event model which is a graph node, EventNode is used during
    extraction to capture structured event information before creating graph nodes.
    """
    id: Optional[str] = Field(None, description="Optional ID for this extraction node (auto-generated if not provided)")
    summary: str = Field(description="Brief event title or summary (e.g., 'Attended charity marathon')")
    description: str = Field(description="Detailed description of the event including context and details")
    timestamp: Optional[str] = Field(None, description="When the event occurred (ISO-8601 format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
    primary_participants: List[str] = Field(default_factory=list, description="List of main participant IDs (Person or Entity)")
    location: Optional[str] = Field(None, description="Where the event took place")
    duration: Optional[str] = Field(None, description="Duration of the event (e.g., '2 hours', '1 day')")
    event_type: Optional[Literal["social", "work", "personal", "health", "education", "other"]] = Field(None, description="Type/category of event")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence score for this extraction")
    source_text: Optional[str] = Field(None, description="Original text span this event was extracted from")
    evidence_span: Optional[Dict[str, Union[str, int]]] = Field(None, description="{text, char_start, char_end}")
    metadata: Metadata = Field(default_factory=Metadata, description="Extraction metadata")

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Validate ISO-8601 datetime format for timestamp"""
        if v is None:
            return v

        from datetime import datetime
        try:
            # Support various ISO formats
            if 'T' in v:
                # Full datetime: YYYY-MM-DDTHH:MM:SS
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            else:
                # Date only: YYYY-MM-DD
                datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError(
                f'Invalid ISO-8601 timestamp format: {v}. '
                f'Expected format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS'
            )

    @field_validator('summary')
    @classmethod
    def validate_summary(cls, v: str, info: ValidationInfo) -> str:
        """Validate that summary is not too long and is descriptive"""
        if not v or not isinstance(v, str):
            raise ValueError('summary must be a non-empty string')

        if len(v) > 200:
            raise ValueError(f'summary too long ({len(v)} chars), must be under 200 characters')

        # Check if it's just a verb phrase (should be more descriptive)
        v_lower = v.lower().strip()
        if v_lower in ['volunteered', 'attended', 'participated', 'ran', 'went']:
            raise ValueError(
                f'summary "{v}" is too generic. '
                f'Include what was volunteered/attended (e.g., "volunteered at shelter" not just "volunteered")'
            )

        return v.strip()

    @field_validator('id')
    @classmethod
    def validate_id_if_provided(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """If ID is provided, validate its format with 'event_' prefix"""
        if v is None:
            return v

        if not isinstance(v, str):
            raise ValueError('ID must be a string if provided')

        if not v.startswith('event_'):
            raise ValueError(f'Event extraction node ID must start with "event_", got: {v}')

        # Check ID format follows NodeType standards (no unstructured text)
        parts = v.split('_', 1)
        if len(parts) != 2:
            raise ValueError(f'ID "{v}" must follow format: event_[identifier]')

        identifier = parts[1]
        if not identifier or not identifier.strip():
            raise ValueError(f'ID "{v}" has empty identifier part after prefix')

        # Ensure identifier is properly formatted (not arbitrary text)
        if ' ' in identifier or len(identifier) > 100:
            raise ValueError(
                f'ID "{v}" has invalid identifier format. '
                f'Identifier should be concise (max 100 chars, no spaces). '
                f'Use structured identifiers, not free-text sentences.'
            )

        return v

    def to_event_node(self) -> "Event":
        """Convert EventNode to graph Event node for storage"""
        from datetime import datetime
        import uuid

        # Generate ID with proper validation
        if self.id:
            # If ID was provided, validate it again
            if not self.id.startswith('event_'):
                raise ValueError(f'Invalid event ID format: {self.id}')
            event_id = self.id
        else:
            # Auto-generate ID
            event_id = f"event_{uuid.uuid4().hex[:12]}"

        # Parse timestamp to get start time
        time_start = self.timestamp
        time_end = None

        # If duration is provided, try to calculate end time
        if self.duration and self.timestamp:
            try:
                # Simple duration parsing (could be enhanced)
                if 'hour' in self.duration:
                    hours = int(self.duration.split()[0])
                    # For simplicity, just keep start time
                    pass
            except:
                pass

        return Event(
            id=event_id,
            title=self.summary,
            participants=self.primary_participants,
            location=self.location,
            time_start=time_start,
            time_end=time_end,
            metadata=self.metadata if hasattr(self, 'metadata') else Metadata()
        )

    model_config = {
        "validate_assignment": True,
        "validate_default": True,
    }


# ============================================================================
# Temporal Index and Query Support (Phase 3)
# ============================================================================

class TemporalRange(BaseModel):
    """Represents a time range for temporal queries"""
    start: str  # ISO-8601 format
    end: str    # ISO-8601 format
    granularity: Literal["year", "month", "day", "time"] = "day"

    @field_validator('start', 'end')
    @classmethod
    def validate_iso_format(cls, v: str, info: ValidationInfo) -> str:
        """Validate ISO-8601 datetime format"""
        from datetime import datetime
        try:
            if 'T' in v:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            else:
                datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError(f'Invalid ISO-8601 format: {v}')


class TemporalQuery(BaseModel):
    """Temporal query parameters for filtering evidence by time"""
    query_type: Literal["exact", "range", "before", "after", "between"] = "range"
    target_date: Optional[str] = None  # ISO-8601 for exact/before/after queries
    time_range: Optional[TemporalRange] = None  # For range/between queries
    relative_time: Optional[str] = None  # e.g., "last week", "3 days ago"
    anchor_date: Optional[str] = None  # ISO-8601 for resolving relative times

    def resolve(self) -> Optional[Tuple[str, str]]:
        """Resolve relative times to absolute ISO-8601 ranges

        Returns:
            Tuple of (start_time, end_time) in ISO-8601 format
            or None if resolution fails
        """
        if self.time_range:
            return (self.time_range.start, self.time_range.end)

        if self.query_type == "exact" and self.target_date:
            return (self.target_date, self.target_date)

        if self.relative_time and self.anchor_date:
            # Use temporal parser to resolve relative time
            from ..utils.temporal_parser import normalize_time
            try:
                normalized = normalize_time(
                    self.relative_time,
                    self.anchor_date,
                    self.anchor_date,
                    self.anchor_date
                )
                if normalized.get("time_start") and normalized.get("time_end"):
                    return (normalized["time_start"], normalized["time_end"])
            except:
                pass

        return None
