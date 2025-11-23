from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
import time
import numpy as np

from ..config import CogMemConfig
from ..models.llm_client import create_llm_client
from ..models.embedding import EmbeddingEncoder
from ..storage.graph_store import GraphStore
from ..storage.schema import (
    Entity, RetrievalResult, Person, Event, Fact, Preference,
    Topic, Relation, Metadata, NodeType
)
from ..storage.vector_store import VectorStore
from ..storage.community_store import CommunityStore
from ..storage.chunk_store import ChunkStore
from ..utils.system_monitor import monitor
from .query_processor import QueryProcessor, ConversationBuffer
from .evidence_templates import (
    TemporalEvidence, IdentityEvidence, RelationshipEvidence,
    LocationEvidence, MultiFactEvidence
)


class Retriever:
    """重构的检索器：基于 Topic-Scope 的智能检索

    核心架构（4阶段管道）：
    ----------------------
    1. Topic Matching: 向量检索 Top-K TopicNodes，获取语义背景
       - 对查询进行向量编码
       - 检索最相关的 Topic 节点
       - 提取 Topic 的 label, description, keywords

    2. Seed Finding: 在全量 Event/Fact 索引中检索种子节点
       - 在 Event 和 Fact 的向量索引中搜索
       - 获取高相似度的种子节点
       - 记录节点类型和相似度分数

    3. Graph Traversal: 从种子节点出发，优先遍历 Topic 邻居
       - 从种子节点开始图遍历
       - 优先访问属于命中 Topic 的邻居节点
       - 收集 Event, Fact, Episode 等节点
       - 基于 Topic 成员关系和相似度评分

    4. Evidence Formatting: 格式化证据，显式包含 [Time], [Topic], [Type]
       - 为每个证据添加时间标记 [Time: YYYY-MM-DD]
       - 关联 Topic 标签 [Topic: Work & Career]
       - 显式节点类型 [Type: Event/Fact/Preference]
       - 生成人类可读的摘要

    关键特性：
    ----------
    - Topic-Scoped Retrieval: 使用 Topic 作为语义过滤器
    - Multi-Stage Ranking: Topic -> Seed -> Traversal -> Format
    - Explicit Metadata: 证据包含明确的结构信息
    - AnswerAgent Ready: 优化格式便于 LLM 理解和回答
    """

    def __init__(
        self,
        cfg: CogMemConfig,
        graph: GraphStore,
        vstore: VectorStore,
        cstore: Optional[CommunityStore] = None,
        chunk_store: Optional[ChunkStore] = None
    ) -> None:
        """初始化检索器组件

        Args:
            cfg: 系统配置
            graph: 知识图谱存储
            vstore: 向量存储（用于 Topic 和节点检索）
            cstore: 社区存储（可选，用于实体扩展）
            chunk_store: Chunk 存储（可选）
        """
        self._cfg = cfg
        self._graph = graph
        self._vstore = vstore
        self._llm = create_llm_client(cfg)
        self._embed = EmbeddingEncoder(cfg)
        self._cstore = cstore
        self._chunk_store = chunk_store

        # 查询处理器（用于种子节点识别）
        self._conversation_buffer = ConversationBuffer(max_turns=5)
        self._query_processor = QueryProcessor(self._llm, self._embed, self._conversation_buffer)

        # Topic 缓存（避免重复计算）
        self._topic_cache: Dict[str, Topic] = {}
        self._topic_embeddings: Dict[str, List[float]] = {}

    # ============================================================================
    # 主检索入口
    # ============================================================================

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        """主检索入口（调用 retrieve_with_scope）

        Args:
            query: 用户查询
            top_k: 返回的证据数量

        Returns:
            RetrievalResult 包含实体和证据
        """
        print(f"[Retriever] Processing query: '{query}'")

        # 开始检索追踪
        retrieval_id = monitor.start_retrieval_tracking(query)

        try:
            # 调用新的 retrieve_with_scope 方法
            result = self.retrieve_with_scope(query, top_k=top_k)

            # 完成追踪
            monitor.complete_retrieval_tracking(
                retrieval_id, result.retrieval_path,
                len(result.entities), len(result.candidates) if hasattr(result, 'candidates') else 0,
                len(result.evidence), getattr(result, 'confidence', 0.0)
            )

            return result

        except Exception as e:
            print(f"[Retriever] Critical error: {e}")
            monitor.complete_retrieval_tracking(
                retrieval_id, ["error"], 0, 0, 0, 0.0, error=str(e)
            )
            raise

    def retrieve_with_scope(self, query: str, top_k: int = 10) -> RetrievalResult:
        """基于 Topic-Scope 的检索（4阶段管道）

        阶段:
        1. Topic Matching - 向量检索 Top-K TopicNodes
        2. Seed Finding - 在全量 Event/Fact 索引中检索种子
        3. Graph Traversal - 优先遍历命中 Topic 的邻居节点
        4. Evidence Formatting - 显式包含 [Time], [Topic], [Type]

        Args:
            query: 用户查询
            top_k: 返回的证据数量

        Returns:
            RetrievalResult 包含实体和格式化证据
        """
        print(f"[Retriever] [Scope] Starting topic-scoped retrieval for: '{query}'")

        result = RetrievalResult(query=query, retrieval_path=["retrieve-with-scope"])

        # ==================== Phase 1: Topic Matching ====================
        # 检索 Top-K TopicNodes，获取语义背景
        print(f"[Retriever] [Scope] [Phase 1] Topic matching...")
        matched_topics = self._match_topics(query, top_k=3)

        if matched_topics:
            print(f"[Retriever] [Scope] [Phase 1] Matched {len(matched_topics)} topics:")
            for topic, similarity in matched_topics:
                print(f"  - {topic.label} (sim: {similarity:.3f})")
        else:
            print(f"[Retriever] [Scope] [Phase 1] No topics matched, will use generic retrieval")

        result.matched_topics = matched_topics  # 存储匹配的 topics
        result.retrieval_path.append("topic-matching")

        # ==================== Phase 2: Seed Finding ====================
        # 在全量 Event/Fact 索引中检索种子节点
        print(f"[Retriever] [Scope] [Phase 2] Finding seed nodes...")
        seed_nodes = self._find_seeds(query, top_k=15)

        if not seed_nodes:
            print(f"[Retriever] [Scope] [Phase 2] No seeds found, falling back to entity-based retrieval")
            return self._fallback_retrieve(query, top_k)

        print(f"[Retriever] [Scope] [Phase 2] Found {len(seed_nodes)} seed nodes:")
        for node_id, node_type, similarity in seed_nodes[:5]:
            print(f"  - {node_id} ({node_type}, sim: {similarity:.3f})")

        result.seed_node_ids = [node_id for node_id, _, _ in seed_nodes]
        result.retrieval_path.append("seed-finding")

        # ==================== Phase 3: Graph Traversal ====================
        # 从种子节点出发，优先遍历 Topic 邻居
        print(f"[Retriever] [Scope] [Phase 3] Traversing graph with topic priority...")

        # 获取命中的 Topic IDs 用于优先遍历
        matched_topic_ids = {topic.id for topic, _ in matched_topics}

        traversed_nodes = self._traverse_with_topic_priority(
            seed_nodes=seed_nodes,
            topic_ids=matched_topic_ids,
            max_nodes=top_k * 3
        )

        if not traversed_nodes:
            print(f"[Retriever] [Scope] [Phase 3] No nodes traversed, using seeds only")
            traversed_nodes = seed_nodes

        print(f"[Retriever] [Scope] [Phase 3] Traversed {len(traversed_nodes)} nodes")
        result.retrieval_path.append("graph-traversal")

        # ==================== Phase 4: Evidence Formatting ====================
        # 格式化证据，显式包含 [Time], [Topic], [Type]
        print(f"[Retriever] [Scope] [Phase 4] Formatting evidence with metadata...")

        evidence = self._format_evidence_with_scope(
            nodes=traversed_nodes[:top_k],
            matched_topics=matched_topics
        )

        result.evidence = evidence
        result.confidence = self._calculate_scope_confidence(evidence, matched_topics)
        result.retrieval_path.append("evidence-formatting")

        print(f"[Retriever] [Scope] Completed: {len(evidence)} evidence items")
        for i, ev in enumerate(evidence[:3]):
            print(f"  [{i}] {ev[:150]}...")

        return result

    # ============================================================================
    # Phase 1: Topic Matching
    # ============================================================================

    def _match_topics(self, query: str, top_k: int = 3) -> List[Tuple[Topic, float]]:
        """向量检索 Top-K TopicNodes

        Args:
            query: 查询字符串
            top_k: 返回的 Topic 数量

        Returns:
            List of (Topic, similarity_score)
        """
        # 编码查询
        query_vec = self._embed.encode([query])
        if not query_vec:
            return []

        # 获取所有 Topic 节点
        topic_nodes = self._get_all_topics()
        if not topic_nodes:
            print(f"[Retriever] [Topic] No topics found in graph")
            return []

        print(f"[Retriever] [Topic] Searching across {len(topic_nodes)} topics...")

        # 计算与每个 Topic 的相似度
        from sklearn.metrics.pairwise import cosine_similarity

        topic_scores = []
        for topic in topic_nodes:
            # 获取 Topic 的文本表示
            topic_text = self._get_topic_text(topic)

            # 编码 Topic 文本
            topic_vec = self._embed.encode([topic_text])
            if not topic_vec:
                continue

            # 计算相似度
            similarity = cosine_similarity([query_vec[0]], [topic_vec[0]])[0][0]

            topic_scores.append((topic, similarity))

        # 按相似度排序
        topic_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回 top-k
        return topic_scores[:top_k]

    def _get_all_topics(self) -> List[Topic]:
        """从图中获取所有 Topic 节点

        Returns:
            List of Topic nodes
        """
        topics = []

        for node_id in self._graph._g.nodes():
            node_data = self._graph._g.nodes[node_id]

            # 检查是否是 Topic 节点
            if node_data.get("node_type") == NodeType.TOPIC.value:
                try:
                    topic = Topic(**node_data)
                    topics.append(topic)
                except Exception as e:
                    print(f"[Retriever] [Topic] Error parsing topic {node_id}: {e}")
                    continue

        return topics

    def _get_topic_text(self, topic: Topic) -> str:
        """获取 Topic 的文本表示用于向量化

        Args:
            topic: Topic 节点

        Returns:
            文本表示
        """
        parts = []

        if topic.label:
            parts.append(topic.label)

        if topic.description:
            parts.append(topic.description)

        if topic.keywords:
            parts.append(" ".join(topic.keywords))

        return " ".join(parts)

    # ============================================================================
    # Phase 2: Seed Finding
    # ============================================================================

    def _find_seeds(
        self,
        query: str,
        top_k: int = 15
    ) -> List[Tuple[str, str, float]]:
        """在全量 Event/Fact 索引中检索种子节点

        Args:
            query: 查询字符串
            top_k: 返回的种子节点数量

        Returns:
            List of (node_id, node_type, similarity_score)
        """
        # 编码查询
        query_vec = self._embed.encode([query])
        if not query_vec:
            return []

        seed_nodes = []

        # 搜索 Event 节点
        print(f"[Retriever] [Seed] Searching for Event seeds...")
        event_seeds = self._search_nodes_by_type(
            query_vec[0],
            node_type=NodeType.EVENT.value,
            top_k=top_k // 2
        )
        seed_nodes.extend(event_seeds)
        print(f"[Retriever] [Seed] Found {len(event_seeds)} Event seeds")

        # 搜索 Fact 节点
        print(f"[Retriever] [Seed] Searching for Fact seeds...")
        fact_seeds = self._search_nodes_by_type(
            query_vec[0],
            node_type=NodeType.FACT.value,
            top_k=top_k // 2
        )
        seed_nodes.extend(fact_seeds)
        print(f"[Retriever] [Seed] Found {len(fact_seeds)} Fact seeds")

        # 按相似度排序
        seed_nodes.sort(key=lambda x: x[2], reverse=True)

        return seed_nodes[:top_k]

    def _search_nodes_by_type(
        self,
        query_vec: List[float],
        node_type: str,
        top_k: int = 10
    ) -> List[Tuple[str, str, float]]:
        """按节点类型搜索节点

        Args:
            query_vec: 查询向量
            node_type: 节点类型（event, fact 等）
            top_k: 返回数量

        Returns:
            List of (node_id, node_type, similarity)
        """
        results = []

        # 从图中获取该类型的所有节点
        for node_id in self._graph._g.nodes():
            node_data = self._graph._g.nodes[node_id]

            if node_data.get("node_type") != node_type:
                continue

            # 获取节点的 embedding（如果有）
            embedding = node_data.get("embedding")
            if not embedding:
                # 如果没有 embedding，生成文本表示并编码
                node_text = self._get_node_text_for_search(node_data, node_type)
                if not node_text:
                    continue

                embedding = self._embed.encode([node_text])
                if not embedding:
                    continue

                embedding = embedding[0]

            # 计算相似度
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([query_vec], [embedding])[0][0]

            # 应用相似度阈值
            if similarity >= self._cfg.thresholds.min_similarity:
                results.append((node_id, node_type, similarity))

        # 排序并返回 top-k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def _get_node_text_for_search(self, node_data: Dict[str, Any], node_type: str) -> str:
        """获取节点文本表示用于搜索

        Args:
            node_data: 节点数据
            node_type: 节点类型

        Returns:
            文本表示
        """
        if node_type == NodeType.EVENT.value:
            # Event: title + description + location
            parts = []
            if node_data.get("title"):
                parts.append(node_data["title"])
            if node_data.get("description"):
                parts.append(node_data["description"])
            if node_data.get("location"):
                parts.append(node_data["location"])
            return " ".join(parts)

        elif node_type == NodeType.FACT.value:
            # Fact: subject + predicate + object/value
            subject = node_data.get("subject", "")
            predicate = node_data.get("predicate", "")
            object_val = node_data.get("object")
            value = node_data.get("value")

            obj = object_val or value or ""
            return f"{subject} {predicate} {obj}".strip()

        else:
            # 其他类型
            return node_data.get("name") or node_data.get("content") or ""

    # ============================================================================
    # Phase 3: Graph Traversal with Topic Priority
    # ============================================================================

    def _traverse_with_topic_priority(
        self,
        seed_nodes: List[Tuple[str, str, float]],
        topic_ids: Set[str],
        max_nodes: int = 30
    ) -> List[Tuple[str, str, float, int]]:
        """从种子节点出发，优先遍历命中 Topic 的邻居节点

        遍历策略：
        1. 从种子节点开始广度优先遍历
        2. 优先访问属于命中 Topic 的邻居节点
        3. 记录遍历深度和 Topic 匹配情况

        Args:
            seed_nodes: 种子节点列表 (node_id, node_type, similarity)
            topic_ids: 命中的 Topic ID 集合
            max_nodes: 最大遍历节点数

        Returns:
            List of (node_id, node_type, similarity, depth)
        """
        if not seed_nodes:
            return []

        graph = self._graph._g
        traversed = []
        visited = set()
        queue = []

        # 初始化队列（种子节点 depth=0）
        for node_id, node_type, similarity in seed_nodes:
            queue.append((node_id, node_type, similarity, 0))
            visited.add(node_id)

        # 广度优先遍历
        while queue and len(traversed) < max_nodes:
            node_id, node_type, similarity, depth = queue.pop(0)

            # 添加当前节点到结果
            traversed.append((node_id, node_type, similarity, depth))

            # 如果太深则停止扩展
            if depth >= 2:
                continue

            # 获取邻居节点
            neighbors = self._get_neighbors(node_id)

            # 按 Topic 优先级排序邻居
            # 优先处理属于命中 Topic 的邻居
            priority_neighbors = []
            normal_neighbors = []

            for neighbor_id in neighbors:
                if neighbor_id in visited:
                    continue

                # 检查邻居是否属于任何命中 Topic
                if self._is_node_in_topics(neighbor_id, topic_ids):
                    priority_neighbors.append(neighbor_id)
                else:
                    normal_neighbors.append(neighbor_id)

            # 先添加优先邻居
            for neighbor_id in priority_neighbors:
                if len(traversed) + len(queue) >= max_nodes:
                    break

                neighbor_type = self._get_node_type(neighbor_id)
                neighbor_sim = similarity * 0.9  # 衰减相似度

                queue.append((neighbor_id, neighbor_type, neighbor_sim, depth + 1))
                visited.add(neighbor_id)

            # 然后添加普通邻居
            for neighbor_id in normal_neighbors:
                if len(traversed) + len(queue) >= max_nodes:
                    break

                neighbor_type = self._get_node_type(neighbor_id)
                neighbor_sim = similarity * 0.8  # 更大衰减

                queue.append((neighbor_id, neighbor_type, neighbor_sim, depth + 1))
                visited.add(neighbor_id)

        return traversed

    def _get_neighbors(self, node_id: str) -> List[str]:
        """获取节点的所有邻居

        Args:
            node_id: 节点 ID

        Returns:
            邻居节点 ID 列表
        """
        graph = self._graph._g
        neighbors = set()

        # 出边邻居
        if node_id in graph:
            for _, neighbor in graph.out_edges(node_id):
                neighbors.add(neighbor)

            # 入边邻居
            for neighbor, _ in graph.in_edges(node_id):
                neighbors.add(neighbor)

        return list(neighbors)

    def _is_node_in_topics(self, node_id: str, topic_ids: Set[str]) -> bool:
        """检查节点是否属于任何命中 Topic

        Args:
            node_id: 节点 ID
            topic_ids: Topic ID 集合

        Returns:
            是否属于命中 Topic
        """
        # 检查从 Topic 到该节点的 COVERS 边
        for topic_id in topic_ids:
            if self._graph._g.has_edge(topic_id, node_id):
                edge_data = self._graph._g[topic_id][node_id]
                if edge_data.get("rel_type") == "COVERS":
                    return True

        return False

    def _get_node_type(self, node_id: str) -> str:
        """获取节点类型

        Args:
            node_id: 节点 ID

        Returns:
            节点类型字符串
        """
        if self._graph.get_person(node_id):
            return NodeType.PERSON.value
        elif self._graph.get_event(node_id):
            return NodeType.EVENT.value
        elif self._graph.get_fact(node_id):
            return NodeType.FACT.value
        elif self._graph.get_preference(node_id):
            return NodeType.PREFERENCE.value
        else:
            return NodeType.ENTITY.value

    # ============================================================================
    # Phase 4: Evidence Formatting with Scope
    # ============================================================================

    def _format_evidence_with_scope(
        self,
        nodes: List[Tuple[str, str, float, int]],
        matched_topics: List[Tuple[Topic, float]]
    ) -> List[str]:
        """格式化证据，显式包含 [Time], [Topic], [Type] 信息

        格式:
        [Time: YYYY-MM-DD] [Topic: Work & Career] [Type: Event] Event description...
        [Time: 2024-01-15] [Topic: Health & Fitness] [Type: Fact] Person has attribute...

        Args:
            nodes: 节点列表 (node_id, node_type, similarity, depth)
            matched_topics: 匹配的 Topic 列表

        Returns:
            格式化证据列表
        """
        evidence = []
        topic_id_to_label = {topic.id: topic.label for topic, _ in matched_topics}

        for node_id, node_type, similarity, depth in nodes:
            # 获取节点对象
            node_obj = self._get_node_object(node_id)
            if not node_obj:
                continue

            # 获取节点数据
            node_data = self._graph._g.nodes[node_id]

            # 格式化证据
            formatted_evidence = self._format_node_evidence(
                node_obj=node_obj,
                node_type=node_type,
                node_data=node_data,
                similarity=similarity,
                depth=depth,
                topic_id_to_label=topic_id_to_label
            )

            if formatted_evidence:
                evidence.append(formatted_evidence)

        return evidence

    def _format_node_evidence(
        self,
        node_obj: Any,
        node_type: str,
        node_data: Dict[str, Any],
        similarity: float,
        depth: int,
        topic_id_to_label: Dict[str, str]
    ) -> str:
        """格式化单个节点的证据

        Args:
            node_obj: 节点对象
            node_type: 节点类型
            node_data: 节点数据字典
            similarity: 相似度分数
            depth: 遍历深度
            topic_id_to_label: Topic ID 到标签的映射

        Returns:
            格式化证据字符串
        """
        parts = []

        # 1. 添加深度标记（调试用）
        parts.append(f"[Depth: {depth}]")

        # 2. 添加时间信息 [Time: ...]
        time_str = self._extract_time_info(node_data, node_type)
        if time_str:
            parts.append(f"[Time: {time_str}]")

        # 3. 添加 Topic 信息 [Topic: ...]
        topic_label = self._extract_topic_info(node_data, topic_id_to_label)
        if topic_label:
            parts.append(f"[Topic: {topic_label}]")

        # 4. 添加类型信息 [Type: ...]
        parts.append(f"[Type: {node_type.capitalize()}]")

        # 5. 添加节点内容
        content = self._extract_node_content(node_obj, node_type)
        if content:
            parts.append(content)

        # 6. 添加置信度
        parts.append(f"[Conf: {similarity:.2f}]")

        return " ".join(parts)

    def _extract_time_info(self, node_data: Dict[str, Any], node_type: str) -> Optional[str]:
        """提取时间信息

        Args:
            node_data: 节点数据
            node_type: 节点类型

        Returns:
            时间字符串或 None
        """
        if node_type == NodeType.EVENT.value:
            # Event: time_start, time_end
            time_start = node_data.get("time_start")
            time_end = node_data.get("time_end")

            if time_start:
                return time_start.split("T")[0]  # 只返回日期部分
            elif time_end:
                return time_end.split("T")[0]

        elif node_type == NodeType.FACT.value:
            # Fact: valid_from, valid_to
            valid_from = node_data.get("valid_from")
            if valid_from:
                return valid_from

        return node_data.get("timestamp")  # 通用时间戳

    def _extract_topic_info(
        self,
        node_data: Dict[str, Any],
        topic_id_to_label: Dict[str, str]
    ) -> Optional[str]:
        """提取 Topic 信息

        Args:
            node_data: 节点数据
            topic_id_to_label: Topic ID 到标签的映射

        Returns:
            Topic 标签或 None
        """
        # 检查节点的 topic_id 字段
        topic_id = node_data.get("topic_id")
        if topic_id and topic_id in topic_id_to_label:
            return topic_id_to_label[topic_id]

        # 可以扩展：检查 COVERS 边等
        return None

    def _extract_node_content(self, node_obj: Any, node_type: str) -> str:
        """提取节点内容

        Args:
            node_obj: 节点对象
            node_type: 节点类型

        Returns:
            内容字符串
        """
        if node_type == NodeType.EVENT.value:
            # Event: title + description + location
            title = getattr(node_obj, "title", "")
            description = getattr(node_obj, "description", "")
            location = getattr(node_obj, "location", "")

            parts = [title]
            if description:
                parts.append(description)
            if location:
                parts.append(f"@ {location}")

            return " ".join(parts)

        elif node_type == NodeType.FACT.value:
            # Fact: subject + predicate + object/value
            subject = getattr(node_obj, "subject", "")
            predicate = getattr(node_obj, "predicate", "")
            object_val = getattr(node_obj, "object", None)
            value = getattr(node_obj, "value", None)

            obj = object_val or value or ""
            return f"{subject} {predicate} {obj}".strip()

        elif node_type == NodeType.PREFERENCE.value:
            # Preference: holder + aspect + polarity
            holder = getattr(node_obj, "holder", "")
            aspect = getattr(node_obj, "aspect", "")
            polarity = getattr(node_obj, "polarity", "")

            return f"{holder} {polarity}s {aspect}"

        else:
            # 其他类型
            return getattr(node_obj, "name") or getattr(node_obj, "content") or str(node_obj)

    def _get_node_object(self, node_id: str) -> Optional[Any]:
        """获取节点对象

        Args:
            node_id: 节点 ID

        Returns:
            节点对象或 None
        """
        if self._graph.get_person(node_id):
            return self._graph.get_person(node_id)
        elif self._graph.get_event(node_id):
            return self._graph.get_event(node_id)
        elif self._graph.get_fact(node_id):
            return self._graph.get_fact(node_id)
        elif self._graph.get_preference(node_id):
            return self._graph.get_preference(node_id)
        else:
            return self._graph.get_entity(node_id)

    def _calculate_scope_confidence(
        self,
        evidence: List[str],
        matched_topics: List[Tuple[Topic, float]]
    ) -> float:
        """计算检索置信度

        因素：
        1. 证据数量（40%）
        2. Topic 匹配质量（30%）
        3. 证据多样性（30%）

        Args:
            evidence: 证据列表
            matched_topics: 匹配的 Topic 列表
        Returns:
            置信度分数 (0-1)
        """
        if not evidence:
            return 0.0

        # 因素1: 证据数量
        quantity_score = min(len(evidence) / 5.0, 1.0) * 0.4

        # 因素2: Topic 匹配质量
        topic_score = 0.0
        if matched_topics:
            avg_topic_sim = sum(sim for _, sim in matched_topics) / len(matched_topics)
            topic_score = avg_topic_sim * 0.3

        # 因素3: 证据多样性（不同类型）
        types = set()
        for ev in evidence:
            if "[Type: Event]" in ev:
                types.add("Event")
            elif "[Type: Fact]" in ev:
                types.add("Fact")
            elif "[Type: Preference]" in ev:
                types.add("Preference")

        diversity_score = min(len(types) / 3.0, 1.0) * 0.3

        return quantity_score + topic_score + diversity_score

    # ============================================================================
    # Fallback Methods
    # ============================================================================

    def _fallback_retrieve(self, query: str, top_k: int) -> RetrievalResult:
        """Fallback to entity-based retrieval when scope retrieval fails

        Args:
            query: 查询字符串
            top_k: 返回的证据数量

        Returns:
            RetrievalResult
        """
        print(f"[Retriever] [Fallback] Using entity-based retrieval for: '{query}'")

        # 使用旧的 retrieve 逻辑（简化版）
        # 提取实体
        entities = []
        entity_names = self._smart_extract_entities(query)

        for name in entity_names:
            entity_list = self._graph.get_entities_by_name(name)
            if entity_list:
                entities.extend(entity_list)

        if not entities:
            return RetrievalResult(query=query, evidence=[])

        # 收集候选节点
        candidates = []
        for entity in entities:
            if entity.id not in self._graph._g:
                continue

            # 1-hop Event/Fact
            for _, target_id, edge_data in self._graph._g.out_edges(entity.id, data=True):
                target_data = self._graph._g.nodes[target_id]
                node_type = target_data.get("node_type", "")

                if node_type in [NodeType.EVENT.value, NodeType.FACT.value]:
                    node_obj = self._get_node_object(target_id)
                    if node_obj:
                        candidates.append((target_id, node_type, 0.5, 1))

        # 格式化证据
        evidence = self._format_evidence_with_scope(
            nodes=candidates[:top_k],
            matched_topics=[]
        )

        return RetrievalResult(
            query=query,
            entities=entities,
            evidence=evidence,
            retrieval_path=["fallback-retrieval"]
        )

    # ============================================================================
    # Legacy Methods (kept for backward compatibility)
    # ============================================================================

    def _smart_extract_entities(self, query: str) -> List[str]:
        """从查询中提取实体名称（简化版）

        Args:
            query: 查询字符串

        Returns:
            实体名称列表
        """
        # 使用简单的实体缓存
        if not hasattr(self, "_entity_cache"):
            self._entity_cache = self._build_entity_cache()

        query_lower = query.lower()
        entities = []

        # 查找匹配的实体
        for entity_name in self._entity_cache:
            if entity_name.lower() in query_lower:
                entities.append(self._entity_cache[entity_name])

        return entities

    def _build_entity_cache(self) -> Dict[str, str]:
        """构建实体缓存

        Returns:
            实体名称小写 -> 原始名称 的映射
        """
        cache = {}

        for node_id, node_data in self._graph._g.nodes(data=True):
            node_type = node_data.get("node_type")

            if node_type == NodeType.PERSON.value:
                name = node_data.get("canonical_name", "")
                if name:
                    cache[name.lower()] = name

            elif node_type == NodeType.ENTITY.value:
                name = node_data.get("name", "")
                if name:
                    cache[name.lower()] = name

        return cache
