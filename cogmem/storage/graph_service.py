"""GraphService: Pure Python layer for atomic graph operations

This module provides a clean interface for graph operations without LLM involvement.
It serves as the foundation for the SearchAgent's reasoning capabilities.
"""
from cogmem.models.execution_errors import raise_if_execution_failure

from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from collections import deque
import re

from .graph_store import GraphStore
from .vector_store import VectorStore
from ..models.embedding import EmbeddingEncoder
from .schema import NodeTypeEnum
from ..utils.bm25_retriever import BM25Retriever, BM25_AVAILABLE


def _contains_identity_phrase(phrase: str, text: str) -> bool:
    """Match complete name tokens, including punctuation and possessives."""
    return bool(phrase and re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text))

@dataclass
class AnchorResult:
    """Result from find_anchors operation"""
    node_id: str
    node_type: str
    name: str
    score: float
    metadata: Dict[str, Any]

@dataclass
class TraversalResult:
    """Result from traverse_associations operation"""
    source_id: str
    target_id: str
    relation_type: str
    target_node: Dict[str, Any]
    weight: float
    confidence: float

@dataclass
class CommonGroundResult:
    """Result from find_common_ground operation"""
    common_nodes: List[Dict[str, Any]]
    intersection_size: int
    node_types: Dict[str, int]  # node_type -> count

class GraphService:
    """Atomic graph operations service for SearchAgent

    Provides low-level graph operations that are composable and testable.
    """

    def __init__(self, graph_store: GraphStore, vector_store: VectorStore, embedding_encoder: EmbeddingEncoder, use_vector_search: bool = True, chunk_store=None):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.embedding_encoder = embedding_encoder
        self.use_vector_search = use_vector_search  # 控制是否使用向量检索
        self.chunk_store = chunk_store  # ChunkStore实例，用于Naive RAG fallback
        
        # BM25 索引（延迟加载）
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._bm25_node_index: Dict[int, str] = {}  # corpus_index -> node_id
        self._bm25_initialized = False
        self._bm25_graph_state = None

    def find_anchors(self, query: str, types: Optional[List[str]] = None, top_k: int = 5,
                     exact_match_boost: float = 1.0) -> List[AnchorResult]:
        """Find anchor nodes in the graph using hybrid search (Three-Stage Rocket)

        Args:
            query: Natural language query to search for
            types: Optional list of node types to filter (e.g., ['person', 'concept'])
            top_k: Number of results to return
            exact_match_boost: Boost score for exact matches (default: 1.0 for perfect match)

        Returns:
            List of AnchorResult objects sorted by score (descending)

        Hybrid Search Strategy (Three-Stage Rocket):
            Stage 1: Heuristic ID Match (Rule-based) - Fastest
                - Normalize query: q = query.strip().lower().replace(" ", "_")
                - Try constructing IDs: p_{q}, c_{q}
                - Check if these IDs exist in graph_store
            Stage 2: Exact Name Search (In-Graph Search) - Fast for < 100k nodes
                - Iterate through Person and Concept nodes
                - Check if node.canonical_name.lower() == query.lower()
            Stage 3: Vector Search (Fallback) - Lower threshold (0.5) for recall
                - Use embedding for semantic search
                - Prefer False Positives over False Negatives
        """
        results_map = {}  # node_id -> AnchorResult (for deduplication)
        query_lower = query.lower().strip()
        query_normalized = query_lower.replace(" ", "_")
        
        # ===== Stage 1: Heuristic ID Match (Rule-based) =====
        # Normalize query: q = query.strip().lower().replace(" ", "_")
        # Try constructing IDs: p_{q}, c_{q}, event_{q}
        
        # Try Person ID: p_{normalized}
        person_id_candidates = [
            f"p_{query_normalized}",
            f"p_{query_lower.replace(' ', '_')}",
            f"person_{query_normalized}",
        ]
        
        for person_id_candidate in person_id_candidates:
            if person_id_candidate in self.graph_store._g:
                node_data = self.graph_store.get_node(person_id_candidate)
                if node_data and node_data.get('is_archived'):
                    continue
                is_person = node_data and node_data.get('node_type') == 'person'
                if is_person and (not types or 'person' in types):
                        results_map[person_id_candidate] = AnchorResult(
                            node_id=person_id_candidate,
                            node_type='person',
                            name=node_data.get('canonical_name', person_id_candidate) if node_data else person_id_candidate,
                            score=1.0,  # Perfect score for direct ID match
                            metadata=node_data or {"node_type": "person"}
                        )
                        break  # Found match, no need to try other variants
        
        # Try Concept ID: c_{normalized}
        concept_id_candidates = [
            f"c_{query_normalized}",
            f"c_{query_lower.replace(' ', '_')}",
            f"concept_{query_normalized}",
        ]
        
        for concept_id_candidate in concept_id_candidates:
            if concept_id_candidate in self.graph_store._g:
                node_data = self.graph_store.get_node(concept_id_candidate)
                if node_data and node_data.get('is_archived'):
                    continue
                is_concept = node_data and node_data.get('node_type') == 'concept'
                if is_concept and (not types or 'concept' in types):
                        results_map[concept_id_candidate] = AnchorResult(
                            node_id=concept_id_candidate,
                            node_type='concept',
                            name=node_data.get('name', concept_id_candidate.replace('c_', '').replace('_', ' ')) if node_data else concept_id_candidate,
                            score=1.0,  # Perfect score for direct ID match
                            metadata=node_data or {"node_type": "concept"}
                        )
                        break  # Found match, no need to try other variants
        
        # ===== Stage 2: Exact Name Search (In-Graph Search) =====
        # Iterate through Person and Concept nodes in the graph
        # If node.canonical_name.lower() == query.lower(): Return immediately (Score 1.0)
        # Note: This is fast for graph sizes < 100k nodes
        
        # ROOT FIX: If query is long (entire question), try to extract entities from it
        query_words = set(query_lower.split())
        is_long_query = len(query_words) > 4

        for node_id in self.graph_store._g.nodes():
            if node_id in results_map:
                continue
            node_data = self.graph_store.get_node(node_id)
            if not node_data:
                continue
            if node_data.get('is_archived'):
                continue
            node_type = node_data.get('node_type', 'unknown')
            if types and node_type not in types:
                continue
            # Check Person nodes
            if node_type == 'person':
                canonical_name = node_data.get('canonical_name', '')
                if canonical_name:
                    canonical_name_lower = canonical_name.lower()
                    # Exact Match (Score 1.0)
                    if canonical_name_lower == query_lower:
                        results_map[node_id] = AnchorResult(
                            node_id=node_id,
                            node_type=node_type,
                            name=canonical_name,
                            score=1.0,
                            metadata=node_data
                        )
                    # Substring Match (Boost Score 0.9)
                    elif (_contains_identity_phrase(query_lower, canonical_name_lower) or
                          _contains_identity_phrase(canonical_name_lower, query_lower)):
                        results_map[node_id] = AnchorResult(
                            node_id=node_id,
                            node_type=node_type,
                            name=canonical_name,
                            score=0.9,
                            metadata=node_data
                        )
                    # ROOT FIX: Even if name doesn't match, check if ID exists in long query (e.g. "p_maria")
                    elif is_long_query and _contains_identity_phrase(node_id, query_lower):
                        results_map[node_id] = AnchorResult(
                            node_id=node_id,
                            node_type=node_type,
                            name=canonical_name or node_id,
                            score=0.98,
                            metadata=node_data
                        )
                
                # Check aliases
                aliases = node_data.get('aliases', [])
                if aliases and isinstance(aliases, list):
                    for alias in aliases:
                        if not isinstance(alias, str) or not alias.strip():
                            continue
                        alias_lower = alias.lower().strip()
                        if alias_lower == query_lower:
                            alias_score = 1.0
                        elif (_contains_identity_phrase(query_lower, alias_lower) or
                              _contains_identity_phrase(alias_lower, query_lower)):
                            alias_score = 0.85
                        else:
                            continue
                        previous = results_map.get(node_id)
                        if previous is None or alias_score > previous.score:
                            results_map[node_id] = AnchorResult(
                                node_id=node_id, node_type=node_type,
                                name=canonical_name or node_id,
                                score=alias_score, metadata=node_data)
            
            # Check Concept nodes
            elif node_type == 'concept':
                concept_name = node_data.get('name', '')
                if concept_name:
                    concept_name_lower = concept_name.lower()
                    if concept_name_lower == query_lower:
                        results_map[node_id] = AnchorResult(
                            node_id=node_id,
                            node_type=node_type,
                            name=concept_name,
                            score=1.0,
                            metadata=node_data
                        )
                    elif query_lower in concept_name_lower or concept_name_lower in query_lower:
                        results_map[node_id] = AnchorResult(
                            node_id=node_id,
                            node_type=node_type,
                            name=concept_name,
                            score=0.9,
                            metadata=node_data
                        )
                    elif is_long_query and concept_name_lower in query_lower:
                        # Concepts in sentence get a boost
                        results_map[node_id] = AnchorResult(
                            node_id=node_id,
                            node_type=node_type,
                            name=concept_name,
                            score=0.85,
                            metadata=node_data
                        )
        
        # CRITICAL FIX: 不要提前返回，而是继续执行 Stage 3 补充语义相关节点
        # 只有在 Stage 1/2 已找到足够 top_k 个结果时才跳过 Stage 3
        # 这样可以确保向量检索能够补充语义相关但名称不完全匹配的节点

        # ===== Stage 2.5: BM25 Fuzzy Match =====
        # 当 Stage 1+2 找到的结果少于 top_k 时，使用 BM25 进行模糊匹配
        # 这可以捕获拼写变体和近义词，例如 "salsa" 能够匹配 "salsa_dancing"
        if len(results_map) < top_k:
            # 延迟初始化 BM25 索引（按需构建）
            if not self._bm25_initialized or self._bm25_graph_state != self._lexical_graph_state():
                self._build_bm25_index()

            # 使用 BM25 进行模糊搜索
            if self._bm25_retriever and self._bm25_retriever.corpus:
                try:
                    # 搜索更多候选结果（top_k * 2）以确保有足够的结果可以过滤
                    eligible = set()
                    for corpus_idx, node_id in self._bm25_node_index.items():
                        node = self.graph_store.get_node(node_id)
                        if node and not node.get('is_archived') and (not types or node.get('node_type') in types):
                            eligible.add(corpus_idx)
                    bm25_results = self._bm25_retriever.search(
                        query, top_k=top_k * 2, allowed_indices=eligible)

                    # 处理 BM25 搜索结果
                    for corpus_idx, score in bm25_results:
                        node_id = self._bm25_node_index.get(corpus_idx)
                        if not node_id:
                            continue

                        # 跳过已找到的节点（避免重复）
                        if node_id in results_map:
                            continue

                        node_data = self.graph_store.get_node(node_id)
                        if not node_data:
                            continue
                        if node_data.get('is_archived'):
                            continue
                        node_type = node_data.get('node_type', 'unknown')
                        if types and node_type not in types:
                            continue
                        # BM25 score，转换为 0.5-0.7 的评分范围
                        # 低于向量检索的精确匹配分数，但高于向量检索的低分
                        bm25_score = min(score * 0.7, 0.7)

                        # 获取显示名称
                        if node_type == 'person':
                            name = node_data.get('canonical_name', node_id)
                        elif node_type == 'concept':
                            name = node_data.get('name', node_id.replace('c_', '').replace('_', ' '))
                        else:
                            name = node_data.get('name', node_id)

                        results_map[node_id] = AnchorResult(
                            node_id=node_id,
                            node_type=node_type,
                            name=name,
                            score=bm25_score,
                            metadata=node_data
                        )

                        # 如果已找到足够结果，提前终止
                        if len(results_map) >= top_k:
                            break
                except Exception as e:
                    raise_if_execution_failure(e)
                    # 如果 BM25 搜索失败，记录日志但不中断主流程
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"BM25 search failed in find_anchors: {e}")

        # ===== Stage 3: Vector Search (Semantic Supplement) =====
        # ROOT FIX: Always execute vector search to supplement results
        # REFINED: Increase top_k during search to improve recall for specific keywords
        query_embedding = None
        if self.use_vector_search:
            vector_top_k = top_k * 3
            try:
                query_embeddings = self.embedding_encoder.encode([query])
                query_embedding = query_embeddings[0]

                vector_results = []
                if types:
                    for node_type in types:
                        if hasattr(self.vector_store, 'search_by_type'):
                            type_results = self.vector_store.search_by_type(query_embedding, node_type, vector_top_k)
                            vector_results.extend(type_results)
                    vector_results.sort(key=lambda x: x[1], reverse=True)
                    vector_results = vector_results[:vector_top_k]
                else:
                    vector_results = self.vector_store.search(query_embedding, top_k=vector_top_k)

                for node_id, score in vector_results:
                    node_data = self.graph_store.get_node(node_id)
                    if not node_data:
                        continue
                    if node_data.get('is_archived'):
                        continue

                    node_type = node_data.get('node_type', 'unknown')
                    if types and node_type not in types:
                        continue

                    score_float = float(score)
                    # Person/concept score thresholds
                    if node_type == 'person':
                        if score_float <= 0.65:
                            continue
                    else:
                        if score_float < 0.5:
                            continue

                    # Merge all returned candidates before final top-k. A weak
                    # earlier channel must not hide the same node's better score.
                    if node_id in results_map:
                        results_map[node_id].score = max(results_map[node_id].score, score_float)
                        continue

                    # Get display name
                    if node_type == 'person':
                        name = node_data.get('canonical_name', node_id)
                    elif node_type == 'concept':
                        name = node_data.get('name', node_id.replace('c_', '').replace('_', ' '))
                    else:
                        name = node_data.get('name', node_id)

                    results_map[node_id] = AnchorResult(
                        node_id=node_id,
                        node_type=node_type,
                        name=name,
                        score=score_float,
                        metadata=node_data
                    )
            except Exception as e:
                raise_if_execution_failure(e)
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Vector search failed in find_anchors: {e}")

        # ===== Stage 4: Raw Chunk Search (Fallback for HotpotQA) =====
        # P0 FIX: Use ChunkStore.search() instead of TypedVectorStore.search_by_type('chunk')
        # (TypedVectorStore does not support 'chunk' type; chunks live in ChunkStore.)
        if (not types or 'chunk' in types) and (len(results_map) < top_k or all(r.score < 0.8 for r in results_map.values())):
            try:
                if query_embedding is None:
                    query_embeddings = self.embedding_encoder.encode([query])
                    query_embedding = query_embeddings[0]
                if self.chunk_store and self.chunk_store.size() > 0:
                    chunk_results = self.chunk_store.search(query_embedding, top_k=min(5, top_k + 2))
                    for chunk, score in chunk_results:
                        # Use chunk.id as node_id; metadata holds text for display
                        cid = chunk.id if hasattr(chunk, 'id') else str(id(chunk))
                        if cid in results_map:
                            continue
                        node_data = {"text": getattr(chunk, 'text', '')[:500], "doc_id": getattr(chunk, 'doc_id', '')}
                        results_map[cid] = AnchorResult(
                            node_id=cid,
                            node_type='chunk',
                            name="Raw Evidence",
                            score=float(score) * 0.95,
                            metadata=node_data
                        )
            except Exception as execution_error:
                raise_if_execution_failure(execution_error)
                pass

        # ===== Sort and Return =====
        results = list(results_map.values())
        results.sort(key=lambda x: x.score, reverse=True)
        final_results = results[:top_k]

        # ===== Enrichment (Single Hop Optimization) =====
        # For Person nodes, pre-fetch key attributes and events to solve single-hop questions in one go
        for result in final_results:
            if result.node_type == 'person':
                try:
                    person_id = result.node_id
                    facts = []
                    
                    # 1. Static Facts/Attributes (Increase K to 20 to ensure core identity isn't missed)
                    attr_edges = self.traverse_associations(
                        [person_id], 
                        relation_types=["HAS_ATTRIBUTE", "WORKS_AT", "LIVES_IN", "PREFERS", "LIKES"],
                        max_depth=1
                    )
                    for edge in attr_edges[:20]:
                        target = edge.target_node
                        if target:
                            # ROOT FIX: Convert Concept IDs to readable names (c_salsa -> salsa)
                            raw_val = target.get('name') or target.get('content') or target.get('canonical_name') or ""
                            val = raw_val.replace('c_', '').replace('_', ' ') if raw_val.startswith('c_') else raw_val
                            if val: facts.append(f"{edge.relation_type}: {val}")

                    # 2. Key Participation (Pull top 5 event titles as previews)
                    event_edges = self.traverse_associations(
                        [person_id],
                        relation_types=["PARTICIPATED_IN", "INVOLVED_IN"],
                        max_depth=1
                    )
                    recent_previews = []
                    for edge in event_edges[:5]:
                        target = edge.target_node
                        if target and target.get('title'):
                            recent_previews.append(f"[{target['title']}]")

                    # 3. Key Claims (Short ones)
                    claim_edges = self.traverse_associations(
                        [person_id],
                        relation_types=["CLAIMS"],
                        max_depth=1
                    )
                    for edge in claim_edges[:5]:
                        target = edge.target_node
                        if target and target.get('content'):
                            content = target['content']
                            if len(content) < 80:
                                facts.append(f"Claim: {content}")

                    if facts:
                        result.metadata["known_facts"] = " | ".join(facts)
                    if recent_previews:
                        result.metadata["event_previews"] = " | ".join(recent_previews)
                except Exception as execution_error:
                    raise_if_execution_failure(execution_error)
                    pass

        return final_results

    def _lexical_graph_state(self):
        """Track public graph upserts and reloads without persistent schema changes."""
        graph = self.graph_store._g
        return (id(graph), len(graph), len(getattr(self.graph_store, '_changelog', ())))

    def _build_bm25_index(self):
        """构建 BM25 索引（延迟加载，按需构建）
        
        构建包含所有节点的完整索引，类型过滤在搜索时进行。
        """
        self._bm25_graph_state = self._lexical_graph_state()
        self._bm25_retriever = None
        self._bm25_node_index = {}
        if not BM25_AVAILABLE:
            self._bm25_initialized = True
            return
        
        try:
            corpus = []
            node_ids = []
            
            # 遍历所有节点，构建文本语料库（不进行类型过滤，构建完整索引）
            for node_id in self.graph_store._g.nodes():
                node_data = self.graph_store.get_node(node_id)
                if not node_data:
                    continue
                
                node_type = node_data.get('node_type', 'unknown')
                
                # 构建节点文本（用于 BM25 检索）
                text_parts = []
                
                # Person: canonical_name + aliases
                if node_type == 'person':
                    if 'canonical_name' in node_data:
                        text_parts.append(str(node_data['canonical_name']))
                    if 'aliases' in node_data and node_data['aliases']:
                        if isinstance(node_data['aliases'], list):
                            text_parts.extend([str(a) for a in node_data['aliases']])
                        else:
                            text_parts.append(str(node_data['aliases']))
                
                # Storage addresses are opaque; index the semantic fields.
                elif node_type == 'concept':
                    if node_data.get('name'):
                        text_parts.append(str(node_data['name']))
                    if 'description' in node_data:
                        text_parts.append(str(node_data['description']))
                
                # Event: title + description
                elif node_type == 'event':
                    if 'title' in node_data:
                        text_parts.append(str(node_data['title']))
                    if 'description' in node_data:
                        text_parts.append(str(node_data['description']))
                
                # Fact content contains the proposition; its ID is provenance.
                elif node_type == 'fact':
                    if node_data.get('content'):
                        text_parts.append(str(node_data['content']))
                    if 'description' in node_data:
                        text_parts.append(str(node_data['description']))
                
                # Claim: content
                elif node_type == 'claim':
                    if 'content' in node_data:
                        text_parts.append(str(node_data['content']))
                
                # 其他类型：使用 name 或 ID
                else:
                    if 'name' in node_data:
                        text_parts.append(str(node_data['name']))
                    else:
                        text_parts.append(node_id)
                
                # 组合文本
                node_text = ' '.join(text_parts).strip()
                if node_text:
                    corpus.append(node_text)
                    node_ids.append(node_id)
            
            # 构建 BM25 索引
            if corpus:
                self._bm25_retriever = BM25Retriever(corpus)
                self._bm25_node_index = {i: node_id for i, node_id in enumerate(node_ids)}
            
            self._bm25_initialized = True
            
        except Exception as e:
            raise_if_execution_failure(e)
            # 如果构建失败，标记为已初始化但不可用
            self._bm25_retriever = None
            self._bm25_initialized = True

    def traverse_associations(self, start_node_ids: List[str], relation_types: Optional[List[str]] = None,
                             max_depth: int = 1, direction: str = "out", keyword_filter: Optional[str] = None) -> List[TraversalResult]:
        """Traverse graph from start nodes following specific relation types

        Args:
            start_node_ids: List of node IDs to start traversal from
            relation_types: Optional list of relation types to follow (e.g., ['CLAIMS', 'ABOUT'])
            max_depth: Maximum traversal depth (default: 1 for 1-hop)
            direction: Direction of traversal ("out", "in", or "both")
            keyword_filter: Optional keyword to filter nodes by title or content (e.g., "job" to find job-related events)

        Returns:
            List of TraversalResult objects representing discovered connections

        Strategies:
            - "看法"/"评价" (opinions): Use ['CLAIMS'] to find claims made by a person
            - "爱好"/"属性" (preferences): Use ['PREFERS', 'LIKES', 'USES'] for preferences
            - "参与" (participation): Use ['INVOLVED_IN', 'PARTICIPATED_IN'] for events
            - "断言" (assertions): Use ['ASSERTS'] for facts derived from events
            - "证据链" (evidence chain): Use ['SUPPORTED_BY'] to find events that support a generalized Fact (Memory Consolidation)
            - Keyword filtering: Use keyword_filter to find specific events (e.g., keyword_filter="job" to find job loss events)
        """
        if direction not in {'in', 'out', 'both'}:
            raise ValueError('Traversal direction must be in, out or both')
        if max_depth <= 0:
            return []
        results = []
        graph = self.graph_store._g
        for start_id in dict.fromkeys(start_node_ids):
            start = self.graph_store.get_node(start_id)
            if not start or start.get('is_archived'):
                continue
            # Expansion state is per root and separate from returned edge records.
            # BFS visits each neighbor at its shortest distance from this root.
            queue = deque([(start_id, 0)])
            scheduled = {start_id}
            while queue:
                current, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                edges = []
                if direction in {'out', 'both'}:
                    edges.extend((tail, attrs) for _, tail, attrs in graph.out_edges(current, data=True))
                if direction in {'in', 'both'}:
                    edges.extend((head, attrs) for head, _, attrs in graph.in_edges(current, data=True))
                for neighbor, attrs in edges:
                    relation = attrs.get('rel_type', 'related')
                    if relation_types and relation not in relation_types:
                        continue
                    data = self.graph_store.get_node(neighbor)
                    if not data or data.get('is_archived'):
                        continue
                    if keyword_filter:
                        haystack = ' '.join(str(value or '') for value in
                            [data.get('content'), data.get('title'), attrs.get('edge_title')]).lower()
                        if not any(token in haystack for token in keyword_filter.lower().split()):
                            continue
                    # target_id and target_node always identify the same visited
                    # neighbor, including when following an incoming physical edge.
                    results.append(TraversalResult(source_id=current, target_id=neighbor,
                        relation_type=relation, target_node=data,
                        weight=attrs.get('weight', 1.), confidence=attrs.get('confidence', 1.)))
                    if neighbor not in scheduled and depth + 1 < max_depth:
                        scheduled.add(neighbor)
                        queue.append((neighbor, depth + 1))
        return results

    def find_common_ground(self, entity_ids: List[str], node_types: Optional[List[str]] = None, max_depth: int = 2) -> CommonGroundResult:
        """Find common ground between two or more entities (核心杀手锏)

        This is the core method for answering questions like:
        - "What do Jon and Gina have in common?"
        - "How do they both destress?"

        Algorithm:
            1. Get all neighbors (up to max_depth hops) for each entity
            2. Compute intersection of neighbor sets
            3. Filter by node types (focus on concepts and events)
            4. Return common nodes with metadata

        ROOT FIX: Support multi-hop traversal to find indirect common ground
        - max_depth=1: Direct neighbors only (original behavior)
        - max_depth=2: 2-hop neighbors (e.g., Person -> Event -> Concept)
        - max_depth=3: 3-hop neighbors (e.g., Person -> Event -> Concept -> Event)

        Args:
            entity_ids: List of person/concept IDs (e.g., ['p_jon', 'p_gina'])
            node_types: Optional filter for node types (e.g., ['concept', 'event'])
            max_depth: Maximum traversal depth (default: 2 for 2-hop, supports indirect common ground)

        Returns:
            CommonGroundResult with common nodes and statistics
        """
        if len(entity_ids) < 2:
            return CommonGroundResult(common_nodes=[], intersection_size=0, node_types={})

        # ROOT FIX: Get neighbors for each entity with multi-hop support
        # CRITICAL FIX: Initialize neighbor_sets to avoid UnboundLocalError
        neighbor_sets = []
        entity_neighbors = {}  # entity_id -> {neighbor_id: neighbor_data}

        for entity_id in entity_ids:
                if entity_id not in self.graph_store._g:
                    neighbor_sets.append(set())
                    entity_neighbors[entity_id] = {}
                    continue

                # Get all neighbors (both in and out edges) with multi-hop traversal
                neighbors = set()
                neighbors_data = {}
                visited = set()  # Track visited nodes to avoid cycles within this entity's traversal

                # BFS traversal up to max_depth
                queue = [(entity_id, 0)]  # (node_id, depth)
                visited.add(entity_id)

                while queue:
                    current_id, depth = queue.pop(0)
                    
                    if depth >= max_depth:
                        continue

                    # Get neighbors at current depth
                    # Outgoing neighbors
                    for neighbor_id in self.graph_store._g.successors(current_id):
                        # Add to neighbors set (even if visited, we want to include it)
                        neighbors.add(neighbor_id)
                        if neighbor_id not in neighbors_data:
                            neighbors_data[neighbor_id] = self.graph_store.get_node(neighbor_id)
                        # Add to queue for deeper traversal (only if not visited to avoid cycles)
                        if neighbor_id not in visited and depth + 1 < max_depth:
                            queue.append((neighbor_id, depth + 1))
                            visited.add(neighbor_id)

                    # Incoming neighbors
                    for neighbor_id in self.graph_store._g.predecessors(current_id):
                        # Add to neighbors set (even if visited, we want to include it)
                        neighbors.add(neighbor_id)
                        if neighbor_id not in neighbors_data:
                            neighbors_data[neighbor_id] = self.graph_store.get_node(neighbor_id)
                        # Add to queue for deeper traversal (only if not visited to avoid cycles)
                        if neighbor_id not in visited and depth + 1 < max_depth:
                            queue.append((neighbor_id, depth + 1))
                            visited.add(neighbor_id)

                neighbor_sets.append(neighbors)
                entity_neighbors[entity_id] = neighbors_data

        # Compute intersection
        # CRITICAL FIX: Check if we have at least 2 neighbor sets before computing intersection
        if not neighbor_sets or len(neighbor_sets) < 2:
            return CommonGroundResult(common_nodes=[], intersection_size=0, node_types={})

        common_neighbors = neighbor_sets[0].intersection(*neighbor_sets[1:])

        # Filter and format results
        common_nodes = []
        node_type_counts = {}

        for neighbor_id in sorted(common_neighbors):
            # Get node data (try from any entity's neighbor map)
            neighbor_data = None
            for entity_id in entity_ids:
                if neighbor_id in entity_neighbors[entity_id]:
                    neighbor_data = entity_neighbors[entity_id][neighbor_id]
                    break

            # CRITICAL FIX: If neighbor_data is None or missing 'id', fetch from graph_store
            if not neighbor_data:
                neighbor_data = self.graph_store.get_node(neighbor_id)
            
            if not neighbor_data:
                continue

            # CRITICAL FIX: Ensure the node data includes 'id' field
            # The node_id IS the key, not an attribute inside the node object
            if 'id' not in neighbor_data:
                neighbor_data = neighbor_data.copy()
                neighbor_data['id'] = neighbor_id

            node_type = neighbor_data.get('node_type', 'unknown')

            # Filter by node types if specified
            if node_types and node_type not in node_types:
                continue

            # Focus on concepts and events (most meaningful common ground)
            if node_type in ['concept', 'event']:
                # ROOT FIX: 统一使用 node_type 字段，返回完整的节点数据以支持格式化
                # 确保数据模型一致性：所有字段都使用标准命名（node_type 而非 type）
                node_info = neighbor_data.copy() if neighbor_data else {}
                
                # 确保关键字段存在（数据模型统一性）
                node_info['id'] = neighbor_data.get('id', neighbor_id)
                node_info['node_type'] = node_type  # 统一使用 node_type 字段（而非 type）
                
                # 移除可能存在的旧字段名（type），确保只使用 node_type
                if 'type' in node_info and node_info['type'] != node_type:
                    # 如果 type 字段存在且与 node_type 不一致，移除它（避免混淆）
                    node_info.pop('type', None)
                
                # 补充缺失的字段以支持格式化（优雅降级）
                if node_type == 'concept':
                    if 'name' not in node_info:
                        # 从 ID 推断名称（优雅降级）
                        node_info['name'] = neighbor_data.get('name', neighbor_id.replace('c_', '').replace('_', ' '))
                    if 'description' not in node_info and 'description' in neighbor_data:
                        node_info['description'] = neighbor_data.get('description', '')
                elif node_type == 'event':
                    if 'title' not in node_info:
                        node_info['title'] = neighbor_data.get('title', neighbor_id)
                    # CRITICAL FIX: Display fuzzy time (time_start_raw) if time_start is None
                    time_start = neighbor_data.get('time_start')
                    time_start_raw = neighbor_data.get('time_start_raw')
                    if time_start:
                        node_info['time_start'] = time_start
                    elif time_start_raw:
                        node_info['time_start'] = time_start_raw  # Preserve fuzzy time like "recently"
                    if 'description' not in node_info and 'description' in neighbor_data:
                        node_info['description'] = neighbor_data.get('description', '')
                    # 确保包含其他重要字段
                    if 'location' in neighbor_data:
                        node_info['location'] = neighbor_data.get('location')
                    if 'participants' in neighbor_data:
                        node_info['participants'] = neighbor_data.get('participants', [])
                    if 'session_time' in neighbor_data:
                        node_info['session_time'] = neighbor_data.get('session_time')
                
                common_nodes.append(node_info)
                node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

        return CommonGroundResult(
            common_nodes=common_nodes,
            intersection_size=len(common_nodes),
            node_types=node_type_counts
        )

    def get_node_details(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a node including its neighborhood"""
        if node_id not in self.graph_store._g:
            return None

        node_data = self.graph_store.get_node(node_id)
        if not node_data:
            return None

        # Get incoming and outgoing edges
        incoming = []
        outgoing = []

        for predecessor_id in self.graph_store._g.predecessors(node_id):
            edges = self.graph_store._g[predecessor_id][node_id]
            for edge_key, edge_data in edges.items():
                incoming.append({
                    'source_id': predecessor_id,
                    'rel_type': edge_data.get('rel_type', 'related'),
                    'weight': edge_data.get('weight', 1.0)
                })

        for successor_id in self.graph_store._g.successors(node_id):
            edges = self.graph_store._g[node_id][successor_id]
            for edge_key, edge_data in edges.items():
                outgoing.append({
                    'target_id': successor_id,
                    'rel_type': edge_data.get('rel_type', 'related'),
                    'weight': edge_data.get('weight', 1.0)
                })

        return {
            'node_data': node_data,
            'incoming_edges': incoming,
            'outgoing_edges': outgoing,
            'degree': len(incoming) + len(outgoing)
        }

    def find_claims_about_target(self, source_id: Optional[str] = None,
                                 target_id: Optional[str] = None,
                                 sentiment: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find claims made about a target (for opinion/questions about others)

        Args:
            source_id: Optional filter for who made the claim
            target_id: Optional filter for who/what the claim is about
            sentiment: Optional filter for sentiment (positive/negative/neutral)

        Returns:
            List of claim nodes with full context
        """
        claims = []
        seen = set()

        # Find CLAIMS edges
        for edge in self.graph_store.get_edges_by_type('CLAIMS'):
            claim_id = edge['tail_id']
            claim_node = self.graph_store.get_node(claim_id)

            if (not claim_node or claim_node.get('node_type') != 'claim'
                    or claim_node.get('is_archived') or claim_id in seen):
                continue
            # Attribution is part of the proposition, not a replaceable hint.
            # A mismatched speaker edge cannot authorize a different speaker.
            if claim_node.get('source_id') != edge['head_id']:
                continue

            # Filter by source if specified
            if source_id and edge['head_id'] != source_id:
                continue

            # Filter by target if specified
            if target_id and claim_node.get('target_id') != target_id:
                continue

            # Filter by sentiment
            if sentiment and claim_node.get('sentiment') != sentiment:
                continue

            seen.add(claim_id)
            claims.append(claim_node)

        return claims
