from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal
import re

@dataclass
class ModelConfig:
    # === 主要Provider配置 ===
    provider: str = "siliconflow"  # bailian | ollama | siliconflow | modelscope（默认）
    
    # === 分离配置：允许Chat和Embedding使用不同的Provider ===
    # 如果设置了embedding_provider，则embedding使用单独的provider，否则跟随主provider
    embedding_provider: Optional[str] = "siliconflow"  # None | ollama | siliconflow | modelscope
    
    # 可选：直接在配置中提供 OpenAI 兼容接口的 Base URL 与 API Key（优先级高于环境变量）
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    embedding_dim: Optional[int] = 1024  # Will be auto-detected

    # 聊天模型配置
    chat_model: Optional[str] = None  # 通用聊天模型
    ollama_chat_model: Optional[str] = "phi4"  # Ollama专用聊天模型


    siliconflow_chat_model: Optional[str] = "Qwen/Qwen2.5-14B-Instruct"  # SiliconFlow专用聊天模型
    modelscope_chat_model: Optional[str] = "qwen2.5-14b-instruct"  # ModelScope专用聊天模型

    # 嵌入模型配置
    embedding_model: Optional[str] = "BAAI/bge-m3"  # 通用嵌入模型
    ollama_embedding_model: Optional[str] = "bge-m3"  # Ollama专用嵌入模型
    siliconflow_embedding_model: Optional[str] = "BAAI/bge-m3"  # SiliconFlow专用嵌入模型
    modelscope_embedding_model: Optional[str] = "bge-m3"  # ModelScope专用嵌入模型

    # Ollama 配置
    ollama_base_url: Optional[str] = "http://localhost:11434"
    
    # SiliconFlow 配置
    siliconflow_api_key: Optional[str] = None  # 从环境变量 SILICONFLOW_API_KEY 读取
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    
    # ModelScope 配置
    modelscope_api_key: str = "empty"
    modelscope_base_url: str = "http://localhost:8000/v1"
    modelscope_embedding_base_url: Optional[str] = None  # 如果为 None 则使用 base_url

@dataclass
class FAISSConfig:
    metric: str = "cosine"  # cosine | l2
    enable_global_index: bool = False  # 全局索引开关（Pure Graph Mode建议关闭以节省内存）
    use_vector_search: bool = True  # 是否使用向量检索（False 时完全禁用 Stage 3）

@dataclass
class RuntimeConfig:
    batch_size: int = 2
    max_concurrency: int = 2
    request_timeout_s: int = 300

@dataclass
class ChunkConfig:
    """切分配置"""
    # 对话场景（注意：文档可能已经合并了多个turns，所以这里的chunk会更大）
    dialog_turns_per_chunk: int = 1
    dialog_max_length: int = 1000  # 增大以支持合并后的对话文档
    # 通用场景
    general_chunk_size: int = 300
    general_overlap: int = 50
    # Pure Graph Mode 核心开关
    enable_chunk_embedding: bool = False  # 默认为 False，关闭原文嵌入（Pure Graph Mode）
    keep_raw_text: bool = True  # 是否保留原文用于展示（不嵌入但存储，用于回溯到原文）

@dataclass
class ExtractionConfig:
    """提取配置 (v3.3 Enhanced)"""
    use_enhanced_extraction: bool = True  # 使用增强版实体抽取（更丰富的描述）
    use_person_centered: bool = True  # 默认启用人物中心模式
    skip_incomplete_facts: bool = True  # 跳过不完整的事实（缺少object和purpose）
    extract_facts_during_indexing: bool = True  # 是否在索引时提取Fact节点（False时只在记忆巩固时创建Fact）
    # v3.3: 时间解析增强
    resolve_relative_time: bool = True  # 是否将相对时间（yesterday）解析为绝对时间
    create_temporal_edges: bool = True  # 是否创建时间边（HAPPENED_BEFORE, HAPPENED_DURING）
    # v3.3: Concept层级
    build_concept_hierarchy: bool = True  # 是否构建Concept层级关系

# ============================================================================
# Question Type Classification (v3.3 New)
# ============================================================================

QuestionType = Literal["single_hop", "multi_hop", "temporal", "open_domain", "adversarial"]

def classify_question_type(query: str) -> QuestionType:
    """Classify question type using soft evidence from question structure.

    The goal is not to hard-route benchmark items, but to distinguish:
    - direct grounded lookup (`single_hop`)
    - shared / relational reasoning (`multi_hop`)
    - time-conditioned reasoning (`temporal`)
    - counterfactual / hypothesis / trait inference (`open_domain`)
    - likely-unanswerable checks (`adversarial`)
    """
    query_lower = (query or "").strip().lower()
    if not query_lower:
        return "single_hop"

    def _score(patterns: list[str]) -> int:
        return sum(1 for pattern in patterns if re.search(pattern, query_lower))

    multi_hop_patterns = [
        r"\b(both|common|shared|together|each other|relationship between)\b",
        r"\bwhat do .+ and .+ (have in common|share)\b",
        r"\bhow are .+ and .+ related\b",
        r"\bbetween .+ and .+\b",
    ]
    temporal_patterns = [
        r"\b(when|what time|what date)\b",
        r"\bhow long\b",
        r"\bhow many (days|weeks|months|years|times)\b",
        r"\b(before|after|during|since|until|ago)\b",
        r"\b(first|last|recent|recently|latest|earliest|current|currently|soon)\b",
        r"\b(yesterday|today|tomorrow|weekend|month|year|summer|spring|fall|winter)\b",
        r"\b\d{4}\b",
    ]
    adversarial_patterns = [
        r"\b(did|has|was|were)\b.+\b(ever|at any point)\b",
        r"\b(never|not|no)\b.+\b(mention|say|discuss|talk about)\b",
        r"\b(is there any|was there any)\b",
    ]
    inferential_patterns = [
        r"\b(would|could|might|should|likely|probably)\b",
        r"\b(why|what kind of|what type of|what personality|what political|what attributes)\b",
        r"\b(opinion|think|feel|believe|describe|considered)\b",
    ]
    direct_lookup_patterns = [
        r"\b(who|where|which|what)\b",
        r"\b(has|have|did|does|is|are)\b",
    ]

    multi_hop_score = _score(multi_hop_patterns)
    temporal_score = _score(temporal_patterns)
    adversarial_score = _score(adversarial_patterns)
    inferential_score = _score(inferential_patterns)
    direct_score = _score(direct_lookup_patterns)

    if multi_hop_score >= 1:
        return "multi_hop"

    if temporal_score >= 1:
        return "temporal"

    if adversarial_score >= 1 and inferential_score == 0:
        return "adversarial"

    # Counterfactual / hypothetical / trait attribution questions are open-domain,
    # but modal verbs alone should not override stronger temporal evidence.
    if inferential_score >= 2:
        return "open_domain"
    if inferential_score >= 1 and temporal_score == 0 and direct_score == 0:
        return "open_domain"

    return "single_hop"

@dataclass
class AgentConfig:
    """Agent 配置 (v3.3 Enhanced with Dynamic Configuration)"""
    max_iterations: int = 4  # SearchAgent 默认迭代次数；题型配置会进一步覆盖
    
    # ===== 搜索工具参数配置 =====
    # 基于真实索引数据分析（analyze_node_lengths.py）和模型上下文窗口设计
    # Qwen2.5-14B-Instruct: 128K context window
    # 真实数据（P95 tokens/节点）：
    #   - EVENT: 71 tokens（最长）
    #   - CLAIM: 18 tokens
    #   - FACT: 15 tokens
    #   - CONCEPT: 7 tokens
    #   - PERSON: 7 tokens
    # 
    # 上下文分配：
    #   - 系统提示词: ~3K tokens
    #   - 用户问题: ~0.5K tokens
    #   - 历史推理 (10 steps): ~10K tokens
    #   - 单次工具调用观察结果: ~5-10K tokens（保守估计）
    #   - 可用空间: ~115K tokens（但需要为多次调用留出空间）
    
    # semantic_search: 核心工具，用于内容检索（主要返回 FACT/EVENT/CLAIM）
    # 考虑最坏情况：全部是 EVENT（71 tokens/节点），10个节点 = 710 tokens
    # 但通常混合类型，平均约 30 tokens/节点，10个节点 = 300 tokens（安全）
    semantic_search_top_k: int = 10  # 默认10个结果，约300-700 tokens（适合内容问题）
    
    # find_anchors: 实体查找，用于初始锚点（主要返回 PERSON/CONCEPT）
    # PERSON/CONCEPT 平均 7 tokens/节点，5个节点 = 35 tokens（非常轻量）
    find_anchors_top_k: int = 5  # 默认5个结果，约35 tokens（足够找到主要实体）
    
    # traverse_neighbors: 关系遍历，用于探索连接（可能返回各种类型）
    # 保守估计：平均 30 tokens/节点，10个节点 = 300 tokens
    traverse_neighbors_max_results: int = 10  # 默认返回10个结果，约300 tokens
    traverse_neighbors_max_depth: int = 1  # 默认遍历深度为1（直接邻居）
    
    # find_common_ground: 多跳查询，用于查找共同节点（主要返回 CONCEPT/EVENT）
    # 混合类型平均约 20 tokens/节点，5个节点 = 100 tokens
    find_common_ground_top_k: int = 5  # 默认5个共同节点，约100 tokens
    
    # 向量搜索扩展倍数（用于提高召回率）
    # 在按类型搜索时，先检索更多候选，然后取 top_k
    vector_search_expansion_factor: float = 2.0  # semantic_search 按类型搜索时，每个类型检索 top_k * factor 个结果
    
    # v3.3: Evidence grounding 增强
    require_evidence_for_answer: bool = True  # 是否强制要求 get_evidence 后才能给出 final_answer
    auto_get_evidence: bool = True  # 是否在 semantic_search 后自动获取 evidence

    # Skill Runtime 开关（论文消融与 trace）
    enable_routing_skill: bool = True
    enable_evidence_skill: bool = True
    enable_temporal_skill: bool = True
    enable_common_ground_skill: bool = True
    enable_trace: bool = True
    enable_runtime_cache: bool = True

    def get_config_for_question_type(self, question_type: QuestionType) -> Dict[str, Any]:
        """Get optimized configuration for a specific question type (v3.3 Enhancement)
        
        Args:
            question_type: The classified question type
            
        Returns:
            Dict with overridden configuration values
        """
        configs: Dict[QuestionType, Dict[str, Any]] = {
            "single_hop": {
                "semantic_search_top_k": 4,  # 单跳问题不需要太多结果
                "max_iterations": 3,
                "traverse_neighbors_max_depth": 1,
                "require_evidence_for_answer": True,  # 单跳必须有证据
            },
            "multi_hop": {
                "semantic_search_top_k": 8,  # 多跳需要一定候选，但不能过宽
                "max_iterations": 5,
                "find_common_ground_top_k": 6,
                "traverse_neighbors_max_depth": 2,  # 允许更深遍历
            },
            "temporal": {
                "semantic_search_top_k": 6,
                "max_iterations": 4,
                "traverse_neighbors_max_results": 12,
                "traverse_neighbors_max_depth": 1,
            },
            "open_domain": {
                "semantic_search_top_k": 6,
                "max_iterations": 4,
                "require_evidence_for_answer": True,  # 推理类问题也需要明确证据支撑
            },
            "adversarial": {
                "semantic_search_top_k": 5,
                "max_iterations": 3,
                "require_evidence_for_answer": True,  # 对抗性问题必须有证据
            },
        }
        return configs.get(question_type, {})

@dataclass
class CogMemConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    # 持久化路径（目录）
    persist_dir: str = ".cogmem"
    graph_file: str = "graph.pkl"
    graph_changelog_file: str = "graph.changelog.pkl"
    faiss_index_file: str = "faiss.index"
    entity_store_file: str = "entities.npy"
    chunk_index_file: str = "chunks.index"
    chunk_meta_file: str = "chunks.pkl"
    evidence_store_file: str = "evidence_store.pkl"  # EvidenceStore持久化文件

DEFAULT_CONFIG = CogMemConfig()
