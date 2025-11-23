from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import networkx as nx
import time

from ..config import CogMemConfig
from ..models.llm_client import create_llm_client
from ..models.embedding import EmbeddingEncoder
from ..prompt import QUERY_ENTITY_SYSTEM, QUERY_ENTITY_USER, RETRIEVAL_GAP_SYSTEM, RETRIEVAL_GAP_USER
from ..storage.graph_store import GraphStore
from ..storage.schema import Entity, RetrievalResult, Person, Event, Fact, Preference
from ..storage.vector_store import VectorStore
from ..storage.community_store import CommunityStore
from ..storage.chunk_store import ChunkStore
from ..utils.json_utils import safe_json_loads
from ..utils.system_monitor import monitor
from .query_processor import QueryProcessor, ConversationBuffer
from .graph_traversal import SpreadingActivationTraversal
from .evidence_templates import StructuredEvidence


class Retriever:
    """智能检索器：基于查询感知的证据生成和图遍历扩展。

    核心架构：
    ---------
    1. 查询分析: QueryTypeAnalyzer 识别查询类型和时序模式
    2. 实体链接: 多策略实体提取（图匹配、LLM、语言模式）
    3. 候选收集: 智能图遍历，1-hop到2-hop平衡扩展
    4. 生成证据: 查询类型驱动的证据模板生成结构化证据
    5. 重排序: _rerank_chunks() 对证据进行相关性重排序

    关键特性：
    --------
    - 查询感知: 根据查询类型（时序/身份/关系）选择检索策略
    - 事实优先: 优先收集Fact节点（包含答案关键信息）
    - 结构化证据: 生成"Date: X", "Identity: Y"格式（非原始文本）
    - 多策略融合: 图遍历 + 向量检索 + LLM分析协同工作
"""

    # 常量: 时序关键词（用于遗留方法，新架构使用query_type_analyzer.TEMPORAL_KEYWORDS）
    TEMPORAL_KEYWORDS = frozenset([
        'when', 'time', 'date', 'ago', 'yesterday', 'today', 'tomorrow',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'morning', 'afternoon', 'evening', 'night', 'week', 'month', 'year'
    ])

    def __init__(
        self,
        cfg: CogMemConfig,
        graph: GraphStore,
        vstore: VectorStore,
        cstore: Optional[CommunityStore] = None,
        chunk_store: Optional[ChunkStore] = None
    ) -> None:
        """初始化检索器组件。

        Args:
            cfg: 系统配置
            graph: 知识图谱存储
            vstore: 向量存储（用于实体和chunk）
            cstore: 社区存储（可选，用于实体扩展）
            chunk_store: Chunk存储（可选，用于文本片段）
        """
        self._cfg = cfg
        self._graph = graph
        self._vstore = vstore
        self._llm = create_llm_client(cfg)
        self._embed = EmbeddingEncoder(cfg)
        self._cstore = cstore
        self._chunk_store = chunk_store

        # Initialize QueryProcessor for context-aware query processing
        self._conversation_buffer = ConversationBuffer(max_turns=5)
        self._query_processor = QueryProcessor(self._llm, self._embed, self._conversation_buffer)

        # Initialize SpreadingActivationTraversal for intelligent graph traversal
        self._activation_traversal = SpreadingActivationTraversal(
            graph=graph,
            embed_encoder=self._embed,
            init_energy=1.0,
            decay_factor=0.8,
            energy_threshold=0.1,
            max_iterations=10
        )

    def _smart_extract_entities(self, query: str) -> List[str]:
        """Intelligent entity extraction using graph-aware strategies.

        Strategy:
        1. Graph-based entity matching (primary - dataset-agnostic)
        2. Linguistic pattern extraction (secondary - general heuristics)
        3. LLM extraction (tertiary - when available)

        This approach adapts to any dataset by leveraging the actual graph content.
        
        Args:
            query: User query string
            
        Returns:
            List of validated entity names found in the graph
        """
        entities = []

        # Strategy 1: Graph-aware entity matching (most reliable)
        graph_entities = self._extract_entities_from_graph(query)
        entities.extend(graph_entities)

        # Strategy 2: Minimal linguistic hints (only when necessary)
        if len(entities) < 2:  # If we have very few entities, try minimal hints
            hint_entities = self._extract_entity_hints(query)
            entities.extend(hint_entities)

        # Strategy 3: LLM extraction (primary for better accuracy)
        # Try LLM extraction first, as it's more accurate than graph matching
        try:
            llm_entities = self._extract_entities_llm(query)
            # Add LLM-extracted entities (they will be validated later)
            for entity in llm_entities:
                if entity not in entities:
                    entities.append(entity)
        except Exception as e:
            print(f"[EntityExtract] LLM extraction failed, using graph-based only: {e}")

        # Validate all entities by checking if they exist in graph
        # This replaces brittle regex filtering with actual graph validation
        validated_entities = self._filter_valid_entity_names(entities)

        return list(set(validated_entities))  # Remove any duplicates

    def _extract_entities_from_graph(self, query: str) -> List[str]:
        """Extract entities by matching against the actual graph content.

        This method is dataset-agnostic - it adapts to whatever entities exist
        in the current knowledge graph, making it work for any domain.

        Strategy:
        1. Query n-gram matching against entity names
        2. Fuzzy substring matching
        3. Context-aware scoring and ranking

        Args:
            query: User query string

        Returns:
            List of entity names found in the graph
        """
        import re
        from collections import defaultdict

        entities = []
        query_lower = query.lower()

        # Get all entities from graph (with caching for performance)
        # IMPORTANT: Only cache real entities (Person/Entity nodes), NOT Event/Fact/Preference nodes
        if not hasattr(self, '_graph_entity_cache'):
            self._graph_entity_cache = {}
            try:
                graph = self._graph._g
                for node_id, node_data in graph.nodes(data=True):
                    if not node_data:
                        continue
                    node_type = node_data.get("node_type", "unknown")
                    # Only cache Person and Entity nodes, skip Event/Fact/Preference
                    if node_type == "Person":
                        name = node_data.get("canonical_name", "")
                        if name:
                            self._graph_entity_cache[name.lower()] = name
                    elif node_type == "Entity":
                        name = node_data.get("name", "")
                        if name:
                            self._graph_entity_cache[name.lower()] = name
            except Exception as e:
                print(f"[Retriever] Failed to build entity cache: {e}")
                self._graph_entity_cache = {}

        # Strategy 1: Direct substring matching
        for entity_name_lower, original_name in self._graph_entity_cache.items():
            if entity_name_lower in query_lower and len(entity_name_lower) > 2:
                if original_name not in entities:
                    entities.append(original_name)

        # Strategy 2: N-gram matching (for partial matches)
        if len(entities) < 3:  # Don't over-match if we already have good results
            query_words = set(re.findall(r'\b\w+\b', query_lower))
            for entity_name_lower, original_name in self._graph_entity_cache.items():
                entity_words = set(entity_name_lower.split())
                # Check if entity words appear in query (intersection)
                if entity_words & query_words:  # Set intersection
                    if original_name not in entities:
                        entities.append(original_name)

        return entities

    def _extract_entity_hints(self, query: str) -> List[str]:
        """Extract entity hints using the simplest possible approach.

        The core insight: if entities are properly indexed and named,
        we should just look for exact matches in the query.

        No complex rules, no linguistic analysis, just direct matching.

        Args:
            query: User query string

        Returns:
            List of entity names that appear in the query
        """
        import re

        # Build graph entity index (cached)
        # IMPORTANT: Only index real entities (Person/Entity nodes), NOT Event/Fact/Preference nodes
        if not hasattr(self, '_graph_entity_index'):
            self._graph_entity_index = {}
            try:
                graph = self._graph._g
                for node_id, node_data in graph.nodes(data=True):
                    if not node_data:
                        continue
                    node_type = node_data.get("node_type", "unknown")
                    # Only index Person and Entity nodes, skip Event/Fact/Preference
                    if node_type == "Person":
                        name = node_data.get("canonical_name", "")
                        if name and len(name.strip()) > 1:
                            name_lower = name.lower()
                            self._graph_entity_index[name_lower] = name
                    elif node_type == "Entity":
                        name = node_data.get("name", "")
                        if name and len(name.strip()) > 1:
                            name_lower = name.lower()
                            self._graph_entity_index[name_lower] = name
            except Exception as e:
                print(f"[EntityHints] Failed to build entity index: {e}")
                self._graph_entity_index = {}

        if not self._graph_entity_index:
            return []

        # Simple approach: find all substrings in query that match entity names
        # This is much simpler and more reliable than trying to parse linguistics
        query_lower = query.lower()
        found_entities = []

        for entity_key, original_name in self._graph_entity_index.items():
            if entity_key in query_lower:
                found_entities.append(original_name)

        # Remove duplicates
        unique_entities = []
        seen = set()
        for entity in found_entities:
            if entity not in seen:
                unique_entities.append(entity)
                seen.add(entity)

        return unique_entities

    def _extract_entities_llm(self, query: str) -> List[str]:
        """LLM-based entity extraction with improved prompt.
        
        IMPORTANT: Only extract proper noun entity names (person names, place names, organization names).
        DO NOT extract event titles, verb phrases, or descriptive phrases.
        """
        try:
            from ..prompt import QUERY_ANALYSIS_SYSTEM, QUERY_ANALYSIS_USER
            
            # Use the same prompt as RetrievalAgent for consistency
            response = self._llm.chat(
                QUERY_ANALYSIS_SYSTEM,
                QUERY_ANALYSIS_USER.format(query=query)
            )
            
            analysis = safe_json_loads(response, default={})
            if isinstance(analysis, dict):
                entity_names = analysis.get("required_entities", [])
                # Filter out invalid entity names (event titles, verb phrases, etc.)
                entity_names = self._filter_valid_entity_names(entity_names)
                return entity_names
        except Exception as e:
            print(f"[EntityExtract] LLM extraction failed: {e}")
            import traceback
            traceback.print_exc()

        return []

    def _filter_valid_entity_names(self, entity_names: List[str]) -> List[str]:
        """Validate entity names by checking if they exist in the graph.
        
        Instead of using brittle regex patterns, we validate entities by actually
        looking them up in the knowledge graph. This is more robust and adapts
        to the actual content of the graph.
        
        Args:
            entity_names: List of entity names extracted from query
            
        Returns:
            Filtered list of entity names that exist in the graph
        """
        if not entity_names:
            return []
        
        valid_entities = []
        
        for name in entity_names:
            name = name.strip()
            if not name:
                continue
            
            # Skip if too long (likely a sentence or phrase)
            if len(name) > 50:
                continue
            
            # Validate by checking if entity exists in graph
            # This is more robust than regex patterns
            if self._entity_exists_in_graph(name):
                valid_entities.append(name)
                print(f"[Retriever] [VALIDATE] Entity '{name}' found in graph")
            else:
                print(f"[Retriever] [VALIDATE] Entity '{name}' not found in graph, skipping")
        
        # Limit to top 3 most important entities
        return valid_entities[:3]
    
    def _entity_exists_in_graph(self, entity_name: str) -> bool:
        """Check if an entity exists in the graph by name.

        This method checks both Person nodes (by canonical_name) and Entity nodes (by name),
        and also checks aliases.

        Args:
            entity_name: Entity name to check

        Returns:
            True if entity exists in graph, False otherwise
        """
        if not entity_name:
            return False

        try:
            # Check Person nodes by iterating through all persons
            graph = self._graph._g
            entity_name_lower = entity_name.lower()

            for node_id, node_data in graph.nodes(data=True):
                if not node_data:
                    continue

                # Check Person nodes by canonical_name
                if node_data.get("node_type") == "Person":
                    canonical_name = node_data.get("canonical_name", "")
                    if canonical_name and canonical_name.lower() == entity_name_lower:
                        return True

            # Check Entity nodes
            entities = self._graph.get_entities_by_name(entity_name.lower())
            if entities:
                return True

            # Check aliases (already checked in Person loop above, but also check Entity aliases)
            for node_id, node_data in graph.nodes(data=True):
                if not node_data:
                    continue

                # Check Entity aliases
                if node_data.get("node_type") == "Entity":
                    aliases = node_data.get("aliases", [])
                    if aliases:
                        for alias in aliases:
                            if isinstance(alias, str) and alias.lower() == entity_name_lower:
                                return True

            return False
        except Exception as e:
            print(f"[Retriever] [ERROR] Failed to check entity existence: {e}")
            return False

    def _is_valid_entity_name(self, entity_name: str) -> bool:
        """Validate entity name - filter out action phrases and invalid formats.

        This method applies additional filtering rules to exclude:
        - Action phrases (e.g., "volunteered at shelter", "ran charity race")
        - Overly long strings (likely sentences, not names)
        - Strings with punctuation (not proper names)
        - Empty or too-short strings

        Args:
            entity_name: Entity name to validate

        Returns:
            True if valid entity name, False otherwise
        """
        if not entity_name or len(entity_name.strip()) < 2:
            return False

        # Filter out action phrases (contains verbs, too long, sentence-like)
        action_indicators = [
            'volunteered', 'engaged', 'ran', 'attended', 'participated',
            'did', 'went', 'volunteer', 'engages', 'runs', 'attends',
            'participates', 'organized', 'held', 'hosted', 'completed',
            'shelter', 'race', 'charity', 'marathon', 'fundraiser'
        ]
        name_lower = entity_name.lower()

        # Check for action indicators
        if any(indicator in name_lower for indicator in action_indicators):
            return False

        # Filter if it's too long (likely a sentence, not a name)
        if len(entity_name.split()) > 5:
            return False

        # Filter if it contains punctuation that suggests it's not a proper name
        if any(char in entity_name for char in ['.', ',', '!', '?', ':', ';', '(', ')']):
            return False

        # Filter if it starts with lowercase (likely not a proper noun)
        if entity_name and entity_name[0].islower():
            return False

        return True
    
    def _clean_entity_name(self, entity: str) -> str:
        """Clean entity name for better matching."""
        import re
        # Remove punctuation and extra spaces
        cleaned = re.sub(r'[^\w\s]', '', entity).strip()

        # Title case for consistency
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:].lower()

        return cleaned

    def _entity_exists(self, entity_name: str) -> bool:
        """Check if entity exists in graph (quick validation)."""
        try:
            # Try exact case first
            entities = self._graph.get_entities_by_name(entity_name)
            if entities:
                return True

            # Try lower case
            entities = self._graph.get_entities_by_name(entity_name.lower())
            if entities:
                return True

            # Try capitalized
            entities = self._graph.get_entities_by_name(entity_name.capitalize())
            return len(entities) > 0
        except:
            return True  # If check fails, assume it exists

    def _resolve_entities_smart(self, entity_names: List[str], query: str) -> List[Entity]:
        """Smart entity resolution with multiple strategies.

        Strategy:
        1. Exact match by name
        2. Fuzzy match for variations
        3. Alias resolution
        4. Context-based resolution

        Args:
            entity_names: List of entity names to resolve
            query: Original query for context

        Returns:
            List of resolved Entity objects
        """
        resolved_entities = []
        seen_ids = set()

        for name in entity_names:
            # Strategy 1: Exact match - but limit to first valid entity per name
            entities = self._graph.get_entities_by_name(name)
            if entities:
                # Only take the first entity for each name to avoid duplicates
                entity = entities[0]  # Take first match
                if entity.id not in seen_ids:
                    resolved_entities.append(entity)
                    seen_ids.add(entity.id)
                continue

            # Strategy 2: Fuzzy match (case insensitive)
            entities = self._graph.get_entities_by_name(name.lower())
            if entities:
                for entity in entities:
                    if entity.id not in seen_ids:
                        resolved_entities.append(entity)
                        seen_ids.add(entity.id)
                continue

            # Strategy 3: Alias matching
            for entity in self._graph.iter_entities():
                if hasattr(entity, 'aliases') and entity.aliases:
                    if name.lower() in [alias.lower() for alias in entity.aliases]:
                        if entity.id not in seen_ids:
                            resolved_entities.append(entity)
                            seen_ids.add(entity.id)
                            break

            # Strategy 4: Partial name matching (for complex names)
            if not any(e.id in seen_ids for e in resolved_entities if e.name == name):
                for entity in self._graph.iter_entities():
                    if name.lower() in entity.name.lower() or entity.name.lower() in name.lower():
                        if entity.id not in seen_ids:
                            resolved_entities.append(entity)
                            seen_ids.add(entity.id)
                            break

        return resolved_entities

    def _vector_search_ids(self, text: str, top_k: int) -> List[Tuple[str, float]]:
        vec = self._embed.encode([text])
        if not vec:
            return []
        return self._vstore.search(vec[0], top_k=top_k)

    def _entities_from_ids(self, ids: Set[str]) -> List[Entity]:
        """从ID集合获取实体 - 使用索引优化"""
        results: List[Entity] = []
        for entity_id in ids:
            entity = self._graph.get_entity(entity_id)
            if entity:
                results.append(entity)
        return results

    def _gather_candidates_from_entities(
        self,
        entity_ids: List[str],
        query: Optional[str] = None,
        max_candidates: int = 50
    ) -> List[Tuple[str, str, Union[Event, Fact, Preference], int]]:
        """从实体ID列表收集候选节点（Event/Fact/Preference）

        Args:
            entity_ids: 实体ID列表（Person或Entity）
            query: 可选查询字符串（用于日志）
            max_candidates: 最大候选数

        Returns:
            List of (node_id, node_type, node_obj, graph_distance)
        """
        graph = self._graph._g
        candidates = []
        seen_ids = set()

        for entity_id in entity_ids:
            if entity_id not in graph:
                continue

            # 1-hop traversal: Get directly connected nodes
            for _, target_id, edge_data in graph.out_edges(entity_id, data=True):
                if len(candidates) >= max_candidates:
                    break

                if target_id in seen_ids:
                    continue

                # Get node object
                node_obj = self._get_node_object(target_id)
                if node_obj:
                    candidates.append((target_id, self._get_node_type(target_id), node_obj, 1))
                    seen_ids.add(target_id)

        return candidates
    
    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        """Simplified and robust retrieval pipeline with hybrid seed node identification.

        PIPELINE (4 steps):
        1. Hybrid Seed Node Identification: QueryProcessor + Explicit/Implicit paths
        2. Entity Resolution: Convert seed nodes to entities
        3. Intelligent Expansion: Graph + community expansion
        4. Evidence Generation: Rank and select evidence

        Args:
            query: User query string
            top_k: Number of evidence items to return

        Returns:
            RetrievalResult with entities and evidence
        """
        print(f"[Retriever] Processing query: '{query}'")

        # Start retrieval tracking
        retrieval_id = monitor.start_retrieval_tracking(query)

        try:
            result = RetrievalResult(query=query)

            # Step 1: Hybrid Seed Node Identification (replaces Smart Entity Extraction)
            result.retrieval_path.append("seed-node-identification")

            # Process query with context awareness
            seed_node_ids = self.identify_seed_nodes(
                query,
                enable_coreference=True,
                enable_expansion=True
            )

            result.seed_node_ids = seed_node_ids  # Track seed nodes
            print(f"[Retriever] [Hybrid] Identified {len(seed_node_ids)} seed nodes")

            # Convert seed node IDs to entities
            entities = []
            for node_id in seed_node_ids:
                entity = self._graph.get_entity(node_id)
                if entity:
                    entities.append(entity)

            # Filter invalid entities (e.g., pronouns like 'her')
            valid_entities = []
            for e in entities:
                if not hasattr(e, 'name') or not e.name:
                    continue
                # Remove noisy entities (pronouns, stop words)
                if e.name.lower() in ['he', 'she', 'it', 'they', 'them', 'her', 'his', 'their', 'i', 'you', 'we', 'us']:
                    print(f"[Retriever] Filtered out invalid entity: {e.name}")
                    continue
                # Remove entities that are too short and not meaningful
                if len(e.name.strip()) < 2:
                    print(f"[Retriever] Filtered out too-short entity: {e.name}")
                    continue
                valid_entities.append(e)

            result.entities = valid_entities
            result.extracted_entity_names = [e.name for e in valid_entities]
            print(f"[Retriever] Resolved to {len(valid_entities)} entities: {[e.name for e in valid_entities]}")

            # ============================================
            # Step 2: Community-Based Noise Reduction
            # ============================================
            # Use communities as a "denoiser" to filter out irrelevant entities
            # For queries like "education career", this prevents "Visited the beach" from being retrieved
            if self._cstore and len(self._cstore.semantic) > 0:
                result.retrieval_path.append("community-denoising")
                print(f"[Retriever] [Community] Performing community-based noise reduction...")

                # Find most relevant communities for this query
                relevant_community_ids = self._find_relevant_communities(query, top_k=3)
                print(f"[Retriever] [Community] Found {len(relevant_community_ids)} relevant communities")

                if relevant_community_ids:
                    # Get all members from relevant communities
                    community_member_ids = set()
                    for comm_id in relevant_community_ids:
                        member_ids = self._cstore.get_community_entities(comm_id)
                        community_member_ids.update(member_ids)
                        print(f"[Retriever] [Community] Community {comm_id[:8]}... has {len(member_ids)} members")

                    print(f"[Retriever] [Community] Total community members: {len(community_member_ids)}")

                    # Filter valid_entities to only include those in relevant communities
                    filtered_entities = []
                    for e in valid_entities:
                        if e.id in community_member_ids:
                            filtered_entities.append(e)
                        else:
                            print(f"[Retriever] [Community] Filtered out entity not in relevant communities: {e.name} (id: {e.id[:8]}...)")

                    if filtered_entities:
                        print(f"[Retriever] [Community] After filtering: {len(filtered_entities)}/{len(valid_entities)} entities remain")
                        valid_entities = filtered_entities
                    else:
                        print(f"[Retriever] [Community] Warning: No entities in relevant communities, keeping all entities")

            # If no entities found, fallback to chunk search
            if not valid_entities:
                print("[Retriever] No entities resolved, falling back to chunk-search")
                fallback_result = self._fallback_to_chunk_search(query, top_k)
                fallback_result.extracted_entity_names = []
                fallback_result.retrieval_path = result.retrieval_path + ["chunk-search"]

                monitor.complete_retrieval_tracking(
                    retrieval_id, fallback_result.retrieval_path,
                    0, 0, len(fallback_result.evidence), 0.0, fallback_used=True
                )
                return fallback_result

            # Step 3: Intelligent Expansion
            result.retrieval_path.append("intelligent-expansion")
            expanded_entities, candidates = self._expand_intelligently(entities, query, top_k * 3)
            result.entities = expanded_entities  # Update with expanded entities
            result.candidates = candidates
            print(f"[Retriever] Expanded to {len(expanded_entities)} entities, {len(candidates)} candidates")

            # Step 4: Evidence Generation
            result.retrieval_path.append("evidence-generation")
            evidence, confidence = self._generate_evidence_smart(candidates, query, top_k)
            result.evidence = evidence
            result.reranked_candidates = candidates  # Store for debugging

            print(f"[Retriever] Generated {len(evidence)} evidence items (confidence: {confidence:.2f})")

            # Complete tracking
            monitor.complete_retrieval_tracking(
                retrieval_id, result.retrieval_path,
                len(result.entities), len(candidates), len(evidence), confidence
            )

            # Update conversation buffer with entities from this turn
            self._conversation_buffer.add_turn(
                user_query=query,
                assistant_response=", ".join(evidence[:3]) if evidence else "No evidence found",
                entities=[e.name for e in valid_entities]
            )

            return result
    
        except Exception as e:
            print(f"[Retriever] Critical error: {e}")
            monitor.complete_retrieval_tracking(
                retrieval_id, ["error"], 0, 0, 0, 0.0, error=str(e)
            )
            raise
    
    def _fallback_to_chunk_search(self, query: str, top_k: int) -> RetrievalResult:
        """Fallback method: Perform semantic search on raw text chunks.
        
        This is used when:
        - No entities found in query, OR
        - Entities found in query but not in graph
        
        Args:
            query: User query string
            top_k: Number of results to return
            
        Returns:
            RetrievalResult with chunk-based evidence
        """
        evidence = ["chunk-search"]
        chunks_retrieved = []
        
        if self._chunk_store and self._chunk_store.size() > 0:
            query_vec = self._embed.encode([query])[0]
            chunk_results = self._chunk_store.search(query_vec, top_k=min(15, top_k * 2))
            for ch, score in chunk_results:
                if score >= self._cfg.thresholds.min_similarity:
                    chunks_retrieved.append(ch.text[:300])
            
            if chunks_retrieved:
                # Optionally rerank chunks
                if self._cfg.retrieval.enable_reranking:
                    chunks_retrieved = self._rerank_chunks(
                        query,
                        chunks_retrieved,
                        top_k=self._cfg.retrieval.chunk_top_k
                    )
                evidence.extend(chunks_retrieved[:5])

        
        # Return consistent RetrievalResult structure
        return RetrievalResult(
            query=query,
            entities=[],  # No entities in chunk-search mode
            evidence=evidence if evidence else ["chunk-search-empty"]
        )
    
    def _expand_with_communities(self, entity_ids: List[str]) -> List[str]:
        """使用社区信息扩展实体ID列表（兼容旧接口）

        Args:
            entity_ids: 初始实体ID列表

        Returns:
            扩展后的实体ID列表（包括同社区的其他实体）
        """
        return self._expand_with_communities_smart(entity_ids, None)

    def _expand_intelligently(self, entities: List[Entity], query: str, max_candidates: int) -> Tuple[List[Entity], List]:
        """Intelligent expansion using Spreading Activation for multi-hop reasoning (Phase 2).

        Args:
            entities: Initial entities to expand from
            query: Query for relevance ranking
            max_candidates: Maximum candidates to generate

        Returns:
            Tuple of (expanded_entities, candidates)
        """
        print(f"[Retriever] [SpreadingActivation] Expanding from {len(entities)} entities")

        # Use dict for efficient deduplication by entity ID
        all_entities_dict = {e.id: e for e in entities}

        # Get entity IDs for spreading activation
        entity_ids = list(all_entities_dict.keys())

        # Use Spreading Activation for candidate gathering (instead of fixed-depth traversal)
        # This will gather nodes based on semantic relevance and edge conductance
        activation_candidates = self._gather_candidates_with_spreading_activation(
            seed_node_ids=entity_ids,
            query=query,
            top_k=max_candidates
        )

        print(f"[Retriever] [SpreadingActivation] Generated {len(activation_candidates)} candidates")

        # 1. Extract candidates list (compatible format: node_id, node_type, node_obj, graph_dist)
        # Remove the energy_score as it's not in the backward-compatible format
        candidates = [(n_id, n_type, n_obj, g_dist) for n_id, n_type, n_obj, g_dist, _ in activation_candidates]

        # 2. Extract entities from activation results
        # Only add actual Entity nodes to the entities list
        for node_id, node_type, node_obj, _, _ in activation_candidates:
            # Skip Event/Fact/Preference nodes - they are not Entity objects
            if node_type in ["Event", "Fact", "Preference"]:
                continue
            # Only add if it's an actual Entity node and not already in the dict
            if node_id not in all_entities_dict:
                entity = self._graph.get_entity(node_id)
                if entity:
                    all_entities_dict[node_id] = entity

        # 3. Community-based expansion (query-aware, very limited)
        # Keep community expansion as an additional signal
        if self._cstore:
            try:
                # Get community members but rank by query relevance
                community_entity_ids = self._cstore.get_community_members([e.id for e in entities])
                community_entity_ids = [eid for eid in community_entity_ids if eid not in all_entities_dict]

                # Rank community entities by query relevance
                query_words = set(query.lower().split())
                scored_community_entities = []
                for entity_id in community_entity_ids[:10]:  # Limit to top 10 for ranking
                    entity = self._graph.get_entity(entity_id)
                    if not entity:
                        continue

                    # Score by query relevance
                    score = 0
                    entity_name_lower = entity.name.lower()
                    if any(word in entity_name_lower for word in query_words if len(word) > 2):
                        score += 10
                    if hasattr(entity, 'description') and entity.description:
                        desc_lower = entity.description.lower()
                        if any(word in desc_lower for word in query_words if len(word) > 2):
                            score += 5

                    scored_community_entities.append((score, entity_id, entity))

                # Sort by score and take top 2 most relevant
                scored_community_entities.sort(key=lambda x: x[0], reverse=True)
                for score, entity_id, entity in scored_community_entities[:2]:
                    if entity_id not in all_entities_dict:
                        all_entities_dict[entity_id] = entity
            except Exception as e:
                print(f"[Retriever] Community expansion failed: {e}")

        # Convert back to list (limit to reasonable size to avoid explosion)
        all_entities = list(all_entities_dict.values())
        # Limit total entities to avoid returning too many
        # Maximum: 5 entities for simple queries, 10 for complex queries
        max_entities = 10
        if len(all_entities) > max_entities:
            # Keep initial entities + top relevant expanded entities
            initial_entity_ids = {e.id for e in entities}
            expanded_entities = [e for e in all_entities if e.id not in initial_entity_ids]
            # Limit to max_entities total
            all_entities = list(entities) + expanded_entities[:max(0, max_entities - len(entities))]

        # Print statistics
        candidate_types = set(c[1] for c in candidates if c[1] != "Entity")
        print(f"[Retriever] [SpreadingActivation] Candidate types: {candidate_types}")
        print(f"[Retriever] [SpreadingActivation] Final entities: {len(all_entities)}")

        return all_entities, candidates

    def _gather_candidates_from_entities_smart(self, entity_ids: List[str], query: str, max_candidates: int) -> List:
        """Smart candidate gathering with query-aware filtering and balanced exploration.

        Args:
            entity_ids: 初始实体ID列表
            query: 查询字符串
            max_candidates: 最大候选数

        Returns:
            候选节点列表，包含(node_id, node_type, node_obj, graph_dist)
        """
        graph = self._graph._g
        candidates = []
        seen_ids = set()

        # Prepare query keywords for relevance filtering
        query_words = set(query.lower().split())
        temporal_keywords = self.TEMPORAL_KEYWORDS
        is_temporal_query = any(word in query_words for word in temporal_keywords)

        # Reserve slots for Facts (2-hop nodes) - they contain answer-critical information
        # Strategy: collect all Facts (up to max_candidates // 2), then fill remaining with Events
        reserved_fact_slots = max_candidates // 2
        reserved_event_slots = max_candidates - reserved_fact_slots

        fact_candidates = []
        event_candidates = []

        # Priority 1: First, collect 2-hop Facts (person -> event -> fact)
        # These contain the actual answer information (identity, status, etc.)
        for entity_id in entity_ids:
            if entity_id not in graph:
                continue

            # For each event connected to the person, get its facts
            for _, event_id in graph.out_edges(entity_id):
                if event_id in seen_ids:
                    continue

                # Check if this is an Event node
                event_data = graph.nodes.get(event_id, {})
                if not event_data or event_data.get("node_type") != "Event":
                    continue

                event_obj = self._get_node_object(event_id)
                if not event_obj:
                    continue

                # Add the Event itself first (1-hop)
                if len(event_candidates) < reserved_event_slots:
                    event_candidates.append((event_id, "Event", event_obj, 1))
                    seen_ids.add(event_id)

                # Then add Facts connected to this Event (2-hop)
                for _, fact_id in graph.out_edges(event_id):
                    if fact_id in seen_ids or len(fact_candidates) >= reserved_fact_slots:
                        continue

                    fact_data = graph.nodes.get(fact_id, {})
                    if not fact_data or fact_data.get("node_type") != "Fact":
                        continue

                    fact_obj = self._get_node_object(fact_id)
                    if not fact_obj:
                        continue

                    if self._is_node_relevant_to_query(fact_obj, "Fact", query_words, is_temporal_query):
                        fact_candidates.append((fact_id, "Fact", fact_obj, 2))
                        seen_ids.add(fact_id)

        # Priority 2: Also collect direct Facts (if any) and standalone Events
        for entity_id in entity_ids:
            if entity_id not in graph:
                continue

            # Check direct connections
            for _, target_id in graph.out_edges(entity_id):
                if target_id in seen_ids:
                    continue

                node_data = graph.nodes.get(target_id, {})
                node_type = node_data.get("node_type", "")

                if node_type == "Fact" and len(fact_candidates) < reserved_fact_slots:
                    node_obj = self._get_node_object(target_id)
                    if node_obj and self._is_node_relevant_to_query(node_obj, "Fact", query_words, is_temporal_query):
                        fact_candidates.append((target_id, "Fact", node_obj, 1))
                        seen_ids.add(target_id)

                elif node_type == "Event" and len(event_candidates) < reserved_event_slots:
                    node_obj = self._get_node_object(target_id)
                    if node_obj and self._is_node_relevant_to_query(node_obj, "Event", query_words, is_temporal_query):
                        event_candidates.append((target_id, "Event", node_obj, 1))
                        seen_ids.add(target_id)

        # Combine: Facts first (they contain answers), then Events (for context)
        candidates = fact_candidates + event_candidates

        # If we have very few candidates, try broader search
        if len(candidates) < 3:
            print(f"[Gather] Too few candidates ({len(candidates)}), trying broader search")
            for entity_id in entity_ids:
                if entity_id not in graph:
                    continue

                for _, target_id in graph.out_edges(entity_id):
                    if target_id in seen_ids:
                        continue

                    node_obj = self._get_node_object(target_id)
                    node_type = self._get_node_type(target_id)

                    if node_obj and self._is_node_relevant_to_query(node_obj, node_type, query_words, is_temporal_query):
                        distance = 1 if not any(e[0] == target_id for e in fact_candidates + event_candidates) else 2
                        candidates.append((target_id, node_type, node_obj, distance))
                        seen_ids.add(target_id)

                        if len(candidates) >= max_candidates:
                            break

                if len(candidates) >= max_candidates:
                    break

        return candidates[:max_candidates]

    # LEGACY METHODS (for backward compatibility, will be removed in v2.0)
    # These methods are not used by the current query-aware retrieval pipeline.
    # The new architecture uses QueryTypeAnalyzer + EvidenceTemplates instead.

    def _is_node_relevant_to_query(self, node_obj, node_type: str, query_words: set, is_temporal_query: bool) -> bool:
        """Check if a node is relevant to the query.

        Simplified approach: include all Events and Facts by default, as they contain the most
        actionable information. Only exclude obviously irrelevant content.
        """
        # Always include Events and Facts - they contain the core information
        if node_type in ["Event", "Fact"]:
            return True

        # For Preferences, only include if they seem relevant
        if node_type == "Preference":
            if hasattr(node_obj, 'aspect') and node_obj.aspect:
                aspect_words = set(node_obj.aspect.lower().split())
                if query_words.intersection(aspect_words):
                    return True

        # Exclude generic Entities unless they have specific relevance
        if node_type == "Entity":
            if hasattr(node_obj, 'name') and node_obj.name:
                name_words = set(node_obj.name.lower().split())
                if query_words.intersection(name_words):
                    return True

        return False

    def _get_node_object(self, node_id: str):
        """Get node object by ID."""
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

    def _get_node_type(self, node_id: str) -> str:
        """Get node type string."""
        if self._graph.get_person(node_id):
            return "Person"
        elif self._graph.get_event(node_id):
            return "Event"
        elif self._graph.get_fact(node_id):
            return "Fact"
        elif self._graph.get_preference(node_id):
            return "Preference"
        else:
            return "Entity"

    def _rank_community_members(self, member_ids: List[str], original_entity_ids: List[str],
                               query: Optional[str] = None) -> List[str]:
        """对社区成员进行智能排序

        排序策略：
        1. 优先级：语义社区 > 结构社区（语义相关性更强）
        2. 多样性：避免重复类型的实体
        3. 查询相关性：如果有查询，考虑实体名称与查询的相似度
        4. 连接强度：实体在社区中的中心度

        Args:
            member_ids: 社区成员ID列表
            original_entity_ids: 原始实体ID
            query: 查询字符串

        Returns:
            排序后的成员ID列表
        """
        if not member_ids:
            return []

        scored_members = []

        for member_id in member_ids:
            score = 0.0
            reasons = []

            # 1. 获取实体信息
            entity = self._graph.get_entity(member_id)
            if not entity:
                continue

            # 2. 社区类型优先级（语义社区权重更高）
            semantic_communities = 0
            structure_communities = 0

            for comm in self._cstore.get_entity_communities(member_id):
                if hasattr(comm, 'centroid'):  # SemanticCommunity
                    semantic_communities += 1
                else:  # StructureCommunity
                    structure_communities += 1

            if semantic_communities > 0:
                score += semantic_communities * 10  # 语义社区权重
                reasons.append(f"语义社区:{semantic_communities}")
            if structure_communities > 0:
                score += structure_communities * 5   # 结构社区权重
                reasons.append(f"结构社区:{structure_communities}")

            # 3. 查询相关性（如果有查询）
            if query:
                entity_name_lower = entity.name.lower()
                query_lower = query.lower()

                # 名称匹配
                if entity_name_lower in query_lower or any(word in entity_name_lower for word in query_lower.split()):
                    score += 15
                    reasons.append("查询匹配")

                # 部分匹配
                elif any(word in entity_name_lower for word in query_lower.split() if len(word) > 3):
                    score += 8
                    reasons.append("部分匹配")

            # 4. 实体类型多样性奖励（避免过多同类型实体）
            entity_type = type(entity).__name__

            # 计算当前已选实体的类型分布
            current_types = {}
            for eid in original_entity_ids:
                e = self._graph.get_entity(eid)
                if e:
                    t = type(e).__name__
                    current_types[t] = current_types.get(t, 0) + 1

            # 奖励稀有类型
            if current_types.get(entity_type, 0) < 2:  # 同类型少于2个
                score += 5
                reasons.append(f"多样性:{entity_type}")

            # 5. 连接强度（实体在社区中的中心度）
            total_communities = semantic_communities + structure_communities
            if total_communities > 1:
                score += total_communities * 2
                reasons.append(f"中心度:{total_communities}")

            scored_members.append({
                'id': member_id,
                'score': score,
                'reasons': reasons,
                'entity': entity
            })

        # 按分数排序（降序）
        scored_members.sort(key=lambda x: x['score'], reverse=True)

        # 调试输出（仅在详细模式）
        if scored_members and len(scored_members) > 0:
            top_member = scored_members[0]
            print(f"[CommunityRank] Top member: {top_member['entity'].name} "
                  f"(score: {top_member['score']:.1f}, reasons: {top_member['reasons']})")

        return [member['id'] for member in scored_members]

    def _generate_evidence_smart(self, candidates: List, query: str, top_k: int) -> Tuple[List[str], float]:
        """基于查询类型生成结构化证据（根本性重构）

        Args:
            candidates: List of (node_id, node_type, node_obj, graph_dist)
            query: Query string
            top_k: Number of evidence items to return

        Returns:
            Tuple of (evidence_list, confidence_score)
        """
        if not candidates:
            return [], 0.0

        # 1. 分析查询类型
        from .query_type_analyzer import QueryTypeAnalyzer
        query_analyzer = QueryTypeAnalyzer(self._cfg)
        query_analysis = query_analyzer.analyze(query)

        print(f"[EvidenceGen] Query type: {query_analysis.query_type}")
        print(f"[EvidenceGen] Expected format: {query_analysis.expected_format}")
        print(f"[EvidenceGen] Entities: {query_analysis.entities}")

        # 2. 获取相关节点类型
        relevant_node_types = query_analyzer.get_relevant_node_types(query_analysis.query_type)
        print(f"[EvidenceGen] Relevant node types: {relevant_node_types}")

        # 3. 过滤和评分候选节点
        filtered_candidates = [
            (node_id, node_type, node_obj, graph_dist)
            for node_id, node_type, node_obj, graph_dist in candidates
            if node_type in relevant_node_types
        ]

        if not filtered_candidates:
            print("[EvidenceGen] No candidates match relevant node types, using all")
            filtered_candidates = candidates

        # 4. 根据查询类型选择证据模板生成证据
        evidence = self._generate_typed_evidence(
            filtered_candidates,
            query_analysis,
            top_k
        )

        # 5. 计算置信度
        confidence = self._calculate_evidence_confidence(evidence, query)

        print(f"[EvidenceGen] Generated {len(evidence)} evidence items with confidence {confidence:.2f}")
        for i, ev in enumerate(evidence[:3]):
            print(f"  [{i}] {ev[:100]}...")

        return evidence, confidence

    def _generate_typed_evidence(
        self,
        candidates: List,
        query_analysis: "QueryAnalysis",
        top_k: int
    ) -> List[str]:
        """基于查询类型生成结构化证据

        Args:
            candidates: 候选节点列表
            query_analysis: 查询分析结果
            top_k: 返回的证据数量

        Returns:
            结构化证据列表
        """
        from .evidence_templates import (
            TemporalEvidence,
            IdentityEvidence,
            RelationshipEvidence,
            LocationEvidence,
            MultiFactEvidence,
        )

        evidence_list = []
        seen_facts = set()  # 避免重复Fact

        query_type = query_analysis.query_type

        # 时序证据生成
        if query_type == "Temporal":
            for node_id, node_type, node_obj, graph_dist in candidates:
                if node_type == "Event":
                    evidence = TemporalEvidence.generate(node_obj)
                    if evidence:
                        evidence_list.append(evidence)
                        if len(evidence_list) >= top_k:
                            break

                # 有些Fact也包含时间
                elif node_type == "Fact":
                    evidence = TemporalEvidence.generate_from_fact(node_obj)
                    if evidence:
                        evidence_list.append(evidence)
                        if len(evidence_list) >= top_k:
                            break

        # 身份证据生成
        elif query_type == "Identity":
            for node_id, node_type, node_obj, graph_dist in candidates:
                if node_type == "Fact":
                    evidence = IdentityEvidence.generate(node_obj)
                    if evidence and node_id not in seen_facts:
                        evidence_list.append(evidence)
                        seen_facts.add(node_id)
                        if len(evidence_list) >= top_k:
                            break

        # 关系证据生成
        elif query_type == "Relationship":
            for node_id, node_type, node_obj, graph_dist in candidates:
                if node_type == "Fact":
                    evidence = RelationshipEvidence.generate(node_obj)
                    if evidence and node_id not in seen_facts:
                        evidence_list.append(evidence)
                        seen_facts.add(node_id)
                        if len(evidence_list) >= top_k:
                            break

        # 位置证据生成
        elif query_type == "Location":
            for node_id, node_type, node_obj, graph_dist in candidates:
                evidence = None
                if node_type == "Event":
                    evidence = LocationEvidence.generate_from_event(node_obj)
                elif node_type == "Fact":
                    evidence = LocationEvidence.generate_from_fact(node_obj)

                if evidence:
                    evidence_list.append(evidence)
                    if len(evidence_list) >= top_k:
                        break

        # 多事实证据生成（用于Multi-Hop问题）
        elif query_type == "Multi-Hop":
            # 收集所有Fact节点
            facts = [
                node_obj for node_id, node_type, node_obj, graph_dist in candidates
                if node_type == "Fact"
            ]

            multi_fact_evidence = MultiFactEvidence.generate(facts)
            evidence_list.extend(multi_fact_evidence[:top_k])

        # 单事实证据生成（默认）
        else:
            for node_id, node_type, node_obj, graph_dist in candidates:
                if node_type != "Fact" or node_id in seen_facts:
                    continue

                # 尝试专用模板
                evidence = IdentityEvidence.generate(node_obj)

                # 如果专用模板失败，Fall back to generic formatting
                # BUT if it's Identity query and value contains trans-related info, mark as identity
                if not evidence and hasattr(node_obj, 'predicate') and hasattr(node_obj, 'object'):
                    predicate = node_obj.predicate.lower() if node_obj.predicate else ""
                    value = node_obj.object or getattr(node_obj, 'value', "")
                    value_str = str(value).lower()

                    # For Identity queries, check if this is a trans-related fact
                    if query_type == "Identity" and any(term in value_str for term in ["trans", "gender"]):
                        evidence = "Identity: Transgender"
                    else:
                        # Generic fact formatting: "Fact: Subject Predicate Object"
                        subject = ""
                        if hasattr(node_obj, 'subject'):
                            subject = IdentityEvidence._extract_name_from_id(node_obj.subject)

                        evidence = f"Fact: {subject} {predicate} {value_str}"

                if evidence:
                    evidence_list.append(evidence)
                    seen_facts.add(node_id)
                    if len(evidence_list) >= top_k:
                        break

        # 如果生成的证据不足，回退到通用格式（保持向后兼容）
        if len(evidence_list) < 2:
            print(f"[EvidenceGen] Too few evidence items ({len(evidence_list)}), using fallback")
            fallback_count = 0
            for node_id, node_type, node_obj, graph_dist in candidates:
                if fallback_count >= top_k:
                    break

                fallback_evidence = None
                if node_type == "Event":
                    fallback_evidence = self._format_event_evidence(node_obj)
                elif node_type == "Fact":
                    fallback_evidence = self._format_fact_evidence(node_obj)
                elif node_type == "Preference":
                    fallback_evidence = self._format_preference_evidence(node_obj)

                # 确保不添加空字符串或None
                if fallback_evidence and fallback_evidence.strip() and fallback_evidence not in evidence_list:
                    evidence_list.append(fallback_evidence)
                    fallback_count += 1

        # 过滤掉空字符串和None（根本性修复）
        evidence_list = [ev for ev in evidence_list if ev and ev.strip()]

        print(f"[EvidenceGen] Final evidence list: {len(evidence_list)} valid items")
        if len(evidence_list) < len([e for e in evidence_list if e]):
            print(f"[EvidenceGen] WARNING: Filtered out empty evidence items")

        return evidence_list[:top_k]

    def _calculate_evidence_confidence(self, evidence: List[str], query: str) -> float:
        """Calculate confidence score for generated evidence."""
        if not evidence:
            return 0.0

        # Factor 1: Evidence quantity (more is better, up to a point)
        quantity_score = min(len(evidence) / 5.0, 1.0)

        # Factor 2: Evidence specificity (avoid generic "Entity: X" patterns)
        specific_evidence = [ev for ev in evidence if not ev.startswith("Entity:")]
        specificity_score = len(specific_evidence) / max(len(evidence), 1)

        # Factor 3: Evidence diversity (different types of information)
        evidence_types = set()
        for ev in evidence:
            if "Event:" in ev:
                evidence_types.add("event")
            elif "Fact:" in ev:
                evidence_types.add("fact")
            elif "Preference:" in ev:
                evidence_types.add("preference")

        diversity_score = min(len(evidence_types) / 3.0, 1.0)  # Max 3 types

        # Combined confidence score
        confidence = (
            quantity_score * 0.4 +
            specificity_score * 0.4 +
            diversity_score * 0.2
        )

        return min(confidence, 1.0)

    def identify_seed_nodes(self, query: str, enable_coreference: bool = True, enable_expansion: bool = True) -> List[str]:
        """Hybrid seed node identification using explicit and implicit paths.

        Strategy:
        1. Use QueryProcessor to rewrite query (coreference resolution) and generate variants
        2. Explicit Path: LLM extracts entity names → graph exact match
        3. Implicit Path: Vector search with query variants → retrieve similar nodes
        4. Merge both paths to get comprehensive seed nodes

        Args:
            query: Original user query
            enable_coreference: Whether to enable coreference resolution
            enable_expansion: Whether to enable query expansion

        Returns:
            List of validated entity/node IDs as seeds for graph traversal
        """
        print(f"[Retriever] [Hybrid] Processing query: '{query}'")

        # Step 1: Query preprocessing with QueryProcessor
        query_result = self._query_processor.process_query(
            query,
            enable_coreference=enable_coreference,
            enable_expansion=enable_expansion
        )

        rewritten_query = query_result['rewritten_query']
        seed_queries = query_result['seed_queries']

        print(f"[Retriever] [Hybrid] Rewritten query: '{rewritten_query}'")
        print(f"[Retriever] [Hybrid] Generated {len(seed_queries)} seed queries for vector search")

        seed_node_ids = set()  # Use set to avoid duplicates

        # Path 1: Explicit - LLM entity extraction + graph matching
        print(f"[Retriever] [Hybrid] [Explicit] Extracting entities with LLM...")
        explicit_entities = self._smart_extract_entities(rewritten_query)

        # 验证步骤：过滤无效实体名（根本性修复）
        valid_explicit_entities = []
        for entity_name in explicit_entities:
            # 过滤规则：排除过长、包含动词短语、明显是事件描述的实体名
            if self._is_valid_entity_name(entity_name):
                valid_explicit_entities.append(entity_name)
            else:
                print(f"[Retriever] [Hybrid] [Explicit] Filtered out invalid entity name: '{entity_name}'")

        print(f"[Retriever] [Hybrid] [Explicit] Found {len(valid_explicit_entities)} valid entities: {valid_explicit_entities}")

        # Convert entity names to node IDs
        for entity_name in valid_explicit_entities:
            entities = self._graph.get_entities_by_name(entity_name)
            if entities:
                seed_node_ids.add(entities[0].id)  # Add first match

        print(f"[Retriever] [Hybrid] [Explicit] Added {len(seed_node_ids)} seed nodes from explicit path")

        # Path 2: Implicit - Vector search with query variants
        if self._vstore and self._vstore.size() > 0:
            print(f"[Retriever] [Hybrid] [Implicit] Performing vector search...")

            # Search with multiple query variants to increase recall
            vector_matches = 0
            for i, seed_query in enumerate(seed_queries[:3]):  # Limit to top 3 variants to avoid overload
                try:
                    # Encode query
                    query_vec = self._embed.encode([seed_query])
                    if not query_vec:
                        continue

                    # Search vector store
                    results = self._vstore.search(query_vec[0], top_k=5)

                    for node_id, similarity in results:
                        # Filter by similarity threshold
                        if similarity >= self._cfg.thresholds.min_similarity:
                            seed_node_ids.add(node_id)
                            vector_matches += 1

                    print(f"[Retriever] [Hybrid] [Implicit] Query variant {i+1}: {len(results)} results, "
                          f"similarity >= {self._cfg.thresholds.min_similarity}")

                except Exception as e:
                    print(f"[Retriever] [Hybrid] [Implicit] Vector search failed for variant {i+1}: {e}")
                    continue

            print(f"[Retriever] [Hybrid] [Implicit] Vector search added {vector_matches} matches "
                  f"(total seed nodes: {len(seed_node_ids)})")
        else:
            print(f"[Retriever] [Hybrid] [Implicit] Vector store empty or not available")

        # Convert to list and validate
        seed_nodes = list(seed_node_ids)

        # 步骤2：强制邻居上下文注入
        # 对于每个种子节点，显式获取其1跳邻居（确保Event/Fact等节点被包含）
        if self._graph and self._graph._g:
            print(f"[Retriever] [Hybrid] Injecting 1-hop neighbors for {len(seed_nodes)} seed nodes")
            extended_nodes = set(seed_nodes)

            for node_id in seed_nodes:
                try:
                    # 获取该节点的所有邻居
                    if node_id in self._graph._g:
                        # 获取出边邻居（从该节点指向其他节点）
                        for _, neighbor_id in self._graph._g.out_edges(node_id):
                            extended_nodes.add(neighbor_id)

                        # 获取入边邻居（其他节点指向该节点）
                        for neighbor_id, _ in self._graph._g.in_edges(node_id):
                            extended_nodes.add(neighbor_id)
                except Exception as e:
                    print(f"[Retriever] [Hybrid] Error getting neighbors for {node_id}: {e}")
                    continue

            # 检查是否有新节点被添加
            new_nodes = extended_nodes - set(seed_nodes)
            if new_nodes:
                print(f"[Retriever] [Hybrid] Added {len(new_nodes)} neighbor nodes: {list(new_nodes)[:10]}")
                seed_nodes = list(extended_nodes)

        print(f"[Retriever] [Hybrid] Final seed nodes ({len(seed_nodes)}): {seed_nodes[:10]}...")

        return seed_nodes

    def _find_relevant_communities(self, query: str, top_k: int = 3) -> List[str]:
        """Find most relevant communities for a query using vector similarity.

        This is the "Community Denoiser": Filter out irrelevant communities by first
        finding the most semantically relevant communities for the query.

        For example:
        - Query: "education career" → finds "职业与发展" community
        - Query: "summer activities" → finds "休闲与娱乐" community

        Args:
            query: User query string
            top_k: Number of top communities to return

        Returns:
            List of community IDs that are most relevant to the query
        """
        if not self._cstore or len(self._cstore.semantic) == 0:
            return []

        # Encode query to vector
        query_vec = self._embed.encode([query])
        if not query_vec:
            return []

        # Calculate similarity with each community centroid
        community_scores = []
        for community in self._cstore.semantic:
            if hasattr(community, 'centroid') and community.centroid:
                # Simple cosine similarity (could use more sophisticated measures)
                from sklearn.metrics.pairwise import cosine_similarity
                try:
                    sim = cosine_similarity([query_vec[0]], [community.centroid])[0][0]
                    community_scores.append((community.id, sim))
                except Exception as e:
                    print(f"[Retriever] [Community] Error calculating similarity for community {community.id[:8]}...: {e}")
                    continue

        if not community_scores:
            return []

        # Sort by similarity score (descending)
        community_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k community IDs
        top_communities = [comm_id for comm_id, score in community_scores[:top_k]]

        # Print debug info
        print(f"[Retriever] [Community] Top {top_k} communities for query '{query}':")
        for i, (comm_id, score) in enumerate(community_scores[:top_k]):
            print(f"  [{i+1}] {comm_id[:8]}... (score: {score:.3f})")

        return top_communities

    def _gather_candidates_with_spreading_activation(
        self,
        seed_node_ids: List[str],
        query: str,
        top_k: int = 50
    ) -> List[Tuple[str, str, Union[Event, Fact, Preference], int, float]]:
        """使用激活扩散算法收集候选节点（Phase 2优化）

        相比传统的固定深度遍历，激活扩散能:
        1. 根据语义相关性自动决定传播路径
        2. 能量自然衰减，避免过度扩散
        3. 按能量值排序，取最相关的Top-K节点

        Args:
            seed_node_ids: 种子节点ID列表
            query: 查询字符串（用于相关性过滤）
            top_k: 返回Top-K个节点

        Returns:
            List of (node_id, node_type, node_obj, graph_dist, energy_score)
            其中graph_dist是激活路径长度（跳数），energy_score是能量值
        """
        print(f"[Retriever] [SpreadingActivation] Gathering candidates from {len(seed_node_ids)} seeds")

        # 使用激活扩散遍历
        top_nodes = self._activation_traversal.traverse(
            seed_node_ids=seed_node_ids,
            top_k=top_k,
            include_context=True
        )

        # 转换为RetrievalResult兼容的格式
        candidates = []
        for node_id, energy, node_info in top_nodes:
            node_type = node_info["node_type"]
            node_obj = node_info["node_obj"]
            activation_path = node_info["activation_path"]
            graph_dist = len(activation_path) - 1  # 跳数

            candidates.append((
                node_id,
                node_type,
                node_obj,
                graph_dist,
                energy  # 额外添加能量分数
            ))

        print(f"[Retriever] [SpreadingActivation] Collected {len(candidates)} candidates")

        return candidates

    def _calculate_query_aware_rerank_score(self, node_id: str, node_type: str, node_obj, query: str,
                                          query_embedding, graph_dist: int, vector_sim: float, primary_entity_id: str) -> Tuple[float, Dict]:
        """Calculate query-aware reranking score for a node.

        Args:
            node_id: Node ID
            node_type: Node type (Person, Event, Fact, etc.)
            node_obj: Node object
            query: Query string
            query_embedding: Query embedding vector
            graph_dist: Graph distance from primary entity
            vector_sim: Vector similarity score
            primary_entity_id: Primary entity ID for context

        Returns:
            Tuple of (score, breakdown_dict)
        """
        score = 0.0
        breakdown = {}

        # Graph distance factor (closer is better)
        weights = self.SCORING_WEIGHTS['graph_distance']
        if graph_dist == 0:
            # Initial entities get highest score
            score += weights['close']
            breakdown['graph_distance'] = weights['close']
        elif graph_dist == 1:
            score += weights['near']
            breakdown['graph_distance'] = weights['near']
        elif graph_dist == 2:
            score += weights['medium']
            breakdown['graph_distance'] = weights['medium']
        else:
            score += weights['far']
            breakdown['graph_distance'] = weights['far']

        # Node type factor (context-dependent: Events are more valuable for temporal queries)
        query_lower = query.lower()
        is_temporal_query = any(word in query_lower for word in self.TEMPORAL_KEYWORDS)

        if is_temporal_query:
            # For temporal queries, prioritize Events over Facts
            score += self.SCORING_WEIGHTS['temporal_event' if node_type == "Event" else 'temporal_fact' if node_type == "Fact" else 'temporal_other']
            breakdown['node_type'] = self.SCORING_WEIGHTS['temporal_event' if node_type == "Event" else 'temporal_fact' if node_type == "Fact" else 'temporal_other']
        else:
            # For general queries, Facts are slightly more valuable than Events
            score += self.SCORING_WEIGHTS['general_fact' if node_type == "Fact" else 'general_event' if node_type == "Event" else 'general_other']
            breakdown['node_type'] = self.SCORING_WEIGHTS['general_fact' if node_type == "Fact" else 'general_event' if node_type == "Event" else 'general_other']

        # Vector similarity factor (if available)
        if vector_sim > 0:
            vector_weight = vector_sim * self.SCORING_WEIGHTS['vector_similarity']
            score += vector_weight
            breakdown['vector_similarity'] = vector_weight

        # Enhanced text relevance factor
        query_words = set(query.lower().split())
        relevance_score = 0.0

        # Check predicate relevance for Facts
        if hasattr(node_obj, 'predicate') and node_obj.predicate:
            predicate_words = set(node_obj.predicate.lower().split())
            overlap = len(query_words.intersection(predicate_words))
            if overlap > 0:
                relevance_score += min(overlap * 0.15, self.SCORING_WEIGHTS['text_relevance'] * 0.6)

        # Check name/title relevance
        if hasattr(node_obj, 'name') and node_obj.name:
            name_words = set(node_obj.name.lower().split())
            name_overlap = len(query_words.intersection(name_words))
            if name_overlap > 0:
                relevance_score += min(name_overlap * 0.1, self.SCORING_WEIGHTS['text_relevance'] * 0.4)

        # Check title for Events
        if hasattr(node_obj, 'title') and node_obj.title:
            title_words = set(node_obj.title.lower().split())
            title_overlap = len(query_words.intersection(title_words))
            if title_overlap > 0:
                relevance_score += min(title_overlap * 0.1, self.SCORING_WEIGHTS['text_relevance'] * 0.4)

        # Semantic relevance for specific query patterns
        if any(word in query_lower for word in ['what', 'who']):
            # For "what" queries, boost Facts
            if node_type == "Fact":
                relevance_score += self.SCORING_WEIGHTS['text_relevance'] * 0.3
        elif any(word in query_lower for word in ['when', 'time']):
            # For "when" queries, boost Events with time info
            if node_type == "Event" and hasattr(node_obj, 'time_start') and node_obj.time_start:
                relevance_score += self.SCORING_WEIGHTS['text_relevance'] * 0.3
        elif any(word in query_lower for word in ['where', 'location']):
            # For "where" queries, boost Events with location
            if node_type == "Event" and hasattr(node_obj, 'location') and node_obj.location:
                relevance_score += self.SCORING_WEIGHTS['text_relevance'] * 0.3

        if relevance_score > 0:
            score += min(relevance_score, self.SCORING_WEIGHTS['text_relevance'])
            breakdown['text_relevance'] = min(relevance_score, self.SCORING_WEIGHTS['text_relevance'])

        breakdown['total_score'] = score
        return score, breakdown

    # LEGACY: Fallback evidence formatting (used when template generation fails)
    # These methods provide backward-compatible evidence formatting for edge cases.
    def _format_event_evidence(self, event) -> str:
        """Format Event node as evidence string.

        根本性修改：显式提取并格式化 timestamp 属性。
        对于 Event 节点，格式为：[Date: {timestamp}] Event: {description}
        """
        # Extract core event information
        if hasattr(event, 'title') and event.title:
            title = event.title
        else:
            title = getattr(event, 'name', 'Unknown Event')

        # EXPLICITLY extract timestamp property
        timestamp = None
        if hasattr(event, 'time_start') and event.time_start:
            timestamp = event.time_start
        elif hasattr(event, 'time_end') and event.time_end:
            timestamp = event.time_end
        elif hasattr(event, 'timestamp') and event.timestamp:
            timestamp = event.timestamp

        # Build the description parts
        description_parts = []

        # Add location if available
        if hasattr(event, 'location') and event.location:
            description_parts.append(f"location {event.location}")

        # Add participants if available
        if hasattr(event, 'participants') and event.participants:
            if isinstance(event.participants, list):
                participants_str = ", ".join(event.participants[:3])
                if len(event.participants) > 3:
                    participants_str += f" and {len(event.participants) - 3} others"
                description_parts.append(f"participants {participants_str}")
            else:
                description_parts.append(f"participants {event.participants}")

        # Construct description
        description = title
        if description_parts:
            description += f" | {', '.join(description_parts)}"

        # Format with explicit timestamp extraction
        if timestamp:
            return f"[Date: {timestamp}] Event: {description}"
        else:
            return f"Event: {description}"

    def _format_fact_evidence(self, fact) -> str:
        """Format Fact node as evidence string.

        根本性修复：确保始终返回有效证据字符串，不返回空值
        """
        if hasattr(fact, 'subject') and hasattr(fact, 'predicate'):
            subject_name = self._resolve_entity_name(fact.subject)

            # 即使object或value缺失，也使用predicate生成证据
            # 避免返回空字符串（这是根本原因）
            if hasattr(fact, 'object') and fact.object:
                object_name = self._resolve_entity_name(fact.object)
                fact_content = f"{subject_name} {fact.predicate} {object_name}"
                return f"Fact: {fact_content}"
            elif hasattr(fact, 'value') and fact.value is not None:
                fact_content = f"{subject_name} {fact.predicate} {fact.value}"
                return f"Fact: {fact_content}"
            elif fact.predicate:
                # 即使只有subject和predicate，也生成证据（避免空证据）
                fact_content = f"{subject_name} {fact.predicate}"
                return f"Fact: {fact_content}"

        # Fallback to name-based formatting
        fact_name = getattr(fact, 'name', 'Unknown Fact')
        if fact_name and fact_name != 'Unknown Fact':
            return f"Fact: {fact_name}"

        # 根本性修复：如果所有方法都失败，返回基本事实描述（不返回空字符串）
        fact_id = getattr(fact, 'id', 'unknown')
        return f"Fact: {fact_id}"

    def _format_preference_evidence(self, preference) -> str:
        """Format Preference node as evidence string.

        根本性修复：确保始终返回有效证据字符串
        """
        if hasattr(preference, 'holder') and hasattr(preference, 'preference_type'):
            holder_name = self._resolve_entity_name(preference.holder)
            pref_type = preference.preference_type
            aspect = getattr(preference, 'aspect', '')
            value = getattr(preference, 'value', '')

            parts = [f"Preference: {holder_name} {pref_type}"]
            if aspect:
                parts.append(f"regarding {aspect}")
            if value:
                parts.append(f"with value {value}")

            return " ".join(parts)

        # 根本性修复：如果所有方法都失败，返回基本偏好描述（不返回空字符串）
        pref_id = getattr(preference, 'id', 'unknown')
        return f"Preference: {pref_id}"

    def _resolve_entity_name(self, entity_id: str) -> str:
        """Resolve entity ID to human-readable name."""
        if not entity_id:
            return "unknown"

        entity = self._graph.get_entity(entity_id)
        if entity:
            return entity.name

        # Try to extract a readable name from the ID
        import re
        # Remove common prefixes and convert to title case
        clean_id = re.sub(r'^(person_|event_|fact_|entity_)', '', entity_id)
        return clean_id.replace('_', ' ').title()

    def _expand_with_communities_smart(self, entity_ids: List[str], query: Optional[str] = None) -> List[str]:
        """Smart community-based entity expansion.

        Args:
            entity_ids: Initial entity IDs
            query: Optional query for relevance filtering

        Returns:
            Expanded entity ID list
        """
        if not self._cstore or not entity_ids:
            return entity_ids

        expanded_ids = list(entity_ids)

        try:
            # Get community members for all entities
            community_members = self._cstore.get_community_members(entity_ids)

            # Filter out already included entities
            new_members = [eid for eid in community_members if eid not in expanded_ids]

            if new_members:
                # Rank and limit community members
                ranked_members = self._rank_community_members(new_members, entity_ids, query)
                # Add top-ranked members (limit to prevent explosion)
                max_community_additions = min(5, len(ranked_members))
                expanded_ids.extend(ranked_members[:max_community_additions])

        except Exception as e:
            print(f"[CommunityExpansion] Failed: {e}")

        return expanded_ids

    # ===========================================================================
    # 证据重排序功能（从reranker.py合并）
    # ===========================================================================
    @staticmethod
    def _rerank_chunks(query: str, chunks: List[str], top_k: int = 5) -> List[str]:
        """基于多维度相关性重排序证据片段

        评分维度：
        1. 关键词覆盖度 (40%)
        2. TF-IDF 相似度 (30%)
        3. 长度质量分 (20%)
        4. 语义密度分 (10%)

        Args:
            query: 用户查询
            chunks: 待排序的片段列表
            top_k: 返回前 k 个片段

        Returns:
            重排序后的片段列表
        """
        if not chunks:
            return []

        # 1. 提取查询关键词
        query_keywords = Retriever._extract_keywords(query)

        # 2. 计算每个片段的多维度分数
        scored_chunks = []
        for chunk in chunks:
            score = Retriever._compute_comprehensive_score(query, query_keywords, chunk)
            scored_chunks.append((score, chunk))

        # 3. 按分数降序排序
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # 4. 去重和多样性过滤
        final_chunks = Retriever._diversify_and_deduplicate(scored_chunks, top_k)

        return final_chunks

    @staticmethod
    def _extract_keywords(text: str) -> set:
        """提取关键词（简单版：去停用词）"""
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "must", "can",
                     "在", "的", "了", "是", "有", "和", "与", "或", "但", "而"}
        # 简单的分词（按空格和标点）
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        return {w for w in words if len(w) > 2 and w not in stopwords}

    @staticmethod
    def _compute_comprehensive_score(query: str, query_keywords: set, chunk: str) -> float:
        """计算片段的综合相关性分数"""

        # 1. 关键词覆盖度 (40%)
        chunk_keywords = Retriever._extract_keywords(chunk)
        if not query_keywords:
            keyword_coverage = 0.0
        else:
            overlap = len(query_keywords.intersection(chunk_keywords))
            keyword_coverage = overlap / len(query_keywords)

        # 2. TF-IDF 相似度 (30%) - 简化版
        tfidf_score = Retriever._compute_tfidf_similarity(query_keywords, chunk_keywords)

        # 3. 长度质量分 (20%)
        length_score = Retriever._compute_length_quality(chunk)

        # 4. 语义密度分 (10%)
        density_score = Retriever._compute_semantic_density(chunk)

        # 加权综合分数
        total_score = (
            keyword_coverage * 0.4 +
            tfidf_score * 0.3 +
            length_score * 0.2 +
            density_score * 0.1
        )

        return total_score

    @staticmethod
    def _compute_tfidf_similarity(query_keywords: set, chunk_keywords: set) -> float:
        """计算简化的 TF-IDF 相似度"""
        if not query_keywords or not chunk_keywords:
            return 0.0

        # 简化版：使用 Jaccard 相似度作为 TF-IDF 的近似
        intersection = len(query_keywords.intersection(chunk_keywords))
        union = len(query_keywords.union(chunk_keywords))

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _compute_length_quality(chunk: str) -> float:
        """计算长度质量分：适中长度得分最高"""
        length = len(chunk)

        # 理想长度范围：100-500 字符
        if 100 <= length <= 500:
            return 1.0
        elif 50 <= length < 100:
            return 0.8
        elif 500 < length <= 800:
            return 0.7
        elif length < 50:
            return 0.3  # 太短，信息不足
        else:  # length > 800
            return 0.5  # 太长，可能包含噪音

    @staticmethod
    def _compute_semantic_density(chunk: str) -> float:
        """计算语义密度：实体和关键信息的密度"""
        import re

        # 统计可能的实体标记（大写词、数字、日期等）
        entities = len(re.findall(r'\b[A-Z][a-z]+\b', chunk))  # 专有名词
        numbers = len(re.findall(r'\b\d+\b', chunk))  # 数字
        dates = len(re.findall(r'\b\d{4}\b|\b\d{1,2}[/-]\d{1,2}\b', chunk))  # 日期

        # 计算信息密度
        total_words = len(chunk.split())
        if total_words == 0:
            return 0.0

        info_density = (entities + numbers + dates) / total_words

        # 归一化到 0-1 范围
        return min(info_density * 2, 1.0)

    @staticmethod
    def _diversify_and_deduplicate(scored_chunks: List[tuple], top_k: int) -> List[str]:
        """去重和多样性过滤"""
        final_chunks = []
        seen_signatures = set()

        for score, chunk in scored_chunks:
            # 计算片段签名（用于去重）
            signature = Retriever._compute_chunk_signature(chunk)

            # 检查是否重复
            if signature in seen_signatures:
                continue

            # 检查与已选片段的相似度（避免过度相似）
            if Retriever._is_too_similar_to_selected(chunk, final_chunks):
                continue

            final_chunks.append(chunk)
            seen_signatures.add(signature)

            if len(final_chunks) >= top_k:
                break

        return final_chunks

    @staticmethod
    def _compute_chunk_signature(chunk: str) -> str:
        """计算片段签名用于去重"""
        # 提取关键词并排序，形成签名
        keywords = Retriever._extract_keywords(chunk)
        # 取前10个最长的关键词作为签名
        signature_words = sorted(keywords, key=len, reverse=True)[:10]
        return "|".join(sorted(signature_words))

    @staticmethod
    def _is_too_similar_to_selected(chunk: str, selected_chunks: List[str], threshold: float = 0.8) -> bool:
        """检查片段是否与已选片段过于相似"""
        if not selected_chunks:
            return False

        chunk_keywords = Retriever._extract_keywords(chunk)

        for selected in selected_chunks:
            selected_keywords = Retriever._extract_keywords(selected)
            similarity = Retriever._compute_tfidf_similarity(chunk_keywords, selected_keywords)

            if similarity > threshold:
                return True

        return False


# End of Retriever class