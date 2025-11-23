from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    provider: str = "bailian"  # bailian | ollama
    chat_model: str = "qwen-flash"
    embedding_model: str = "text-embedding-v3"
    embedding_dim: int = 1536
    # 可选：直接在配置中提供 OpenAI 兼容接口的 Base URL 与 API Key（优先级高于环境变量）
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    # Ollama 可选：本地服务地址与模型名称兼容字段
    ollama_base_url: Optional[str] = None
    ollama_chat_model: Optional[str] = None
    ollama_embedding_model: Optional[str] = None


@dataclass
class FAISSConfig:
    metric: str = "cosine"  # cosine | l2
    index_type: str = "IndexFlatIP"  # IndexFlatIP for cosine
    rebuild_on_load: bool = False


@dataclass
class CommunityConfig:
    enable_structure: bool = True
    leiden_resolution: float = 1.0
    semantic_k: int = 8  # Reduced from 16 to prevent over-segmentation


@dataclass
class RuntimeConfig:
    batch_size: int = 8  # 注意：百炼API限制batch_size <= 10，实际使用时会自动限制
    max_concurrency: int = 8
    request_timeout_s: int = 60
    retry_times: int = 3


@dataclass
class Thresholds:
    exact_match_norm: bool = True
    min_similarity: float = 0.25


@dataclass
class ChunkConfig:
    """切分配置"""
    # 对话场景（注意：文档可能已经合并了多个turns，所以这里的chunk会更大）
    dialog_turns_per_chunk: int = 1
    dialog_max_length: int = 1000  # 增大以支持合并后的对话文档
    # 通用场景
    general_chunk_size: int = 300
    general_overlap: int = 50


@dataclass
class PersonCenteredConfig:
    """Person-centered graph configuration"""
    enable_event_mediation: bool = True  # Use event-mediated construction
    enable_person_facets: bool = True  # Create PersonFacet nodes
    hub_penalty_weight: float = 0.1  # Base penalty for hub nodes
    time_validity_default_days: int = 365  # Default validity period for facts
    multi_community_threshold: float = 0.3  # Threshold for multi-community membership
    max_facet_edges_per_person: int = 50  # Max edges per person facet


@dataclass
class ExtractionConfig:
    """提取配置"""
    entity_extractor: str = "llm"  # llm | rule
    relation_extractor: str = "llm"
    enable_temporal_extraction: bool = True  # 时间实体识别
    use_enhanced_extraction: bool = True  # 使用增强版实体抽取（更丰富的描述）
    use_person_centered: bool = False  # Enable person-centered extraction
    use_person_facets: bool = False  # Enable PersonFacet nodes (requires use_person_centered)


@dataclass
class RetrievalConfig:
    gap_max_steps: int = 2
    fusion_semantic_weight: float = 0.6
    fusion_structure_weight: float = 0.4
    topk_candidates: int = 50
    # 新增：片段检索配置
    chunk_top_k: int = 10
    enable_chunk_retrieval: bool = True
    enable_reranking: bool = True


@dataclass
class CogMemConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    community: CommunityConfig = field(default_factory=CommunityConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    thresholds: Thresholds = field(default_factory=Thresholds)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    # 新增
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    person_centered: PersonCenteredConfig = field(default_factory=PersonCenteredConfig)

    # 持久化路径（目录）
    persist_dir: str = ".cogmem"
    graph_file: str = "graph.pkl"
    graph_changelog_file: str = "graph.changelog.pkl"
    faiss_index_file: str = "faiss.index"
    entity_store_file: str = "entities.npy"
    community_meta_file: str = "communities.json"
    community_centroid_file: str = "communities_centroids.npy"
    chunk_index_file: str = "chunks.index"
    chunk_meta_file: str = "chunks.pkl"
    evidence_store_file: str = "evidence_store.pkl"  # EvidenceStore持久化文件


DEFAULT_CONFIG = CogMemConfig()
