"""智能检索Agent：根据问题特征自动选择最佳检索策略"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cogmem import CogMem
    from ..storage.schema import RetrievalResult

from ..config import CogMemConfig
from ..models.llm_client import create_llm_client
from ..prompt import (
    RETRIEVAL_STRATEGY_SELECTION_SYSTEM, RETRIEVAL_STRATEGY_SELECTION_USER,
    QUERY_ENTITY_SYSTEM, QUERY_ENTITY_USER,
    QUERY_ANALYSIS_SYSTEM, QUERY_ANALYSIS_USER,
    EVIDENCE_ASSESSMENT_SYSTEM, EVIDENCE_ASSESSMENT_USER
)
from ..utils.json_utils import safe_json_loads
from ..storage.schema import RetrievalResult, Entity, NodeTypeEnum
from typing import List, Dict, Any
import json


class RetrievalAgent:
    """智能检索Agent：根据问题特征自动选择最佳检索策略"""
    
    def __init__(self, cogmem: CogMem):
        """
        Args:
            cogmem: CogMem实例，提供检索能力
        """
        self.cogmem = cogmem
        self.llm = create_llm_client(cogmem.cfg)
        self.cfg = cogmem.cfg
        
        # 可用的检索策略
        self.strategies = {
            'exact_qa_match': self._exact_qa_match,
            'semantic_qa_search': self._semantic_search,
            'entity_based_retrieval': self._entity_based_retrieval,
            'community_expansion': self._community_expansion,
            'multi_hop_reasoning': self._multi_hop_reasoning
        }
    
    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        """Intelligent multi-step Think-Act retrieval loop.
        
        CONTROL FLOW:
        1. [THINK] Analyze Query: Understand question structure and requirements
        2. [ACT] Initial Retrieval: Fast 1-hop retrieval from main entities
        3. [THINK] Assess Sufficiency: Evaluate if evidence is sufficient
        4. [LOOP] Conditional Expansion: Expand based on assessment (max 2 iterations)
        5. Assemble & Return: Combine and deduplicate evidence
        
        Args:
            query: User query string
            top_k: Number of results to return
            
        Returns:
            RetrievalResult with entities and evidence
        """
        print(f"[RetrievalAgent] [START] Starting intelligent retrieval for query: '{query}'")
        
        # ============================================
        # [THINK] STEP 1: Analyze Query
        # ============================================
        print("[RetrievalAgent] [THINK] Analyzing query structure and requirements...")
        query_analysis = self._analyze_query(query)
        print(f"[RetrievalAgent] [THINK] Query analysis result: {json.dumps(query_analysis, indent=2, ensure_ascii=False)}")
        
        if not query_analysis:
            print("[RetrievalAgent] [WARNING] Query analysis failed, falling back to simple retrieval")
            return self._fallback_retrieval(query, top_k)
        
        required_entities = query_analysis.get("required_entities", [])
        # Filter out invalid entity names (event titles, verb phrases, etc.)
        required_entities = self._filter_valid_entity_names(required_entities)
        question_type = query_analysis.get("question_type", "single_fact")
        complexity_level = query_analysis.get("complexity_level", "simple")
        
        # ============================================
        # [ACT] STEP 2: Initial Retrieval using Retriever
        # ============================================
        print(f"[RetrievalAgent] [ACT] Performing initial retrieval for entities: {required_entities}")
        
        # Lookup main entities in graph
        found_entities = []
        if required_entities:
            found_entities = self._lookup_entities_in_graph(required_entities)
            print(f"[RetrievalAgent] [ACT] Found {len(found_entities)} entities in graph: {[e.name for e in found_entities[:3]]}")
        
        # Use the main Retriever to perform retrieval (it handles all the graph traversal internally)
        # This is more reliable than trying to manually traverse the graph
        initial_evidence = []
        all_entities = list(found_entities)
        
        if found_entities:
            # Use the retriever's retrieve method which handles Topic-Scoped retrieval
            try:
                retrieval_result = self.cogmem.retriever.retrieve(query, top_k=top_k)
                initial_evidence = retrieval_result.evidence if retrieval_result.evidence else []
                all_entities = retrieval_result.entities if retrieval_result.entities else found_entities
                # 保存检索路径以便后续使用
                if hasattr(retrieval_result, 'retrieval_path') and retrieval_result.retrieval_path:
                    # 将在最终结果中合并检索路径
                    pass
                print(f"[RetrievalAgent] [ACT] Retrieved {len(initial_evidence)} evidence items using Retriever")
            except Exception as e:
                print(f"[RetrievalAgent] [ERROR] Retriever failed: {e}, falling back to simple graph traversal")
                # Fallback: simple graph traversal
                graph = self.cogmem.graph._g
                for entity in found_entities:
                    if entity.id in graph:
                        # Get connected nodes (1-hop)
                        for _, target_id, key, edge_data in graph.out_edges(entity.id, keys=True, data=True):
                            fact = self.cogmem.graph.get_fact(target_id)
                            if fact:
                                fact_text = f"Fact: {fact.subject} {fact.predicate} {fact.object}"
                                initial_evidence.append(fact_text)
                            else:
                                event = self.cogmem.graph.get_event(target_id)
                                if event:
                                    event_text = f"Event: {event.title} (Participants: {', '.join(event.participants)})"
                                    if event.time_start:
                                        event_text += f" [Time: {event.time_start}]"
                                    initial_evidence.append(event_text)
        
        print(f"[RetrievalAgent] [ACT] Initial retrieval complete: {len(initial_evidence)} evidence items")
        
        # ============================================
        # [THINK] STEP 3: Assess Evidence Sufficiency
        # ============================================
        print("[RetrievalAgent] [THINK] Assessing evidence sufficiency...")
        assessment = self._assess_evidence_sufficiency(query_analysis, initial_evidence)
        print(f"[RetrievalAgent] [THINK] Assessment result: is_sufficient={assessment.get('is_sufficient')}, "
              f"reason='{assessment.get('reason')}', next_step='{assessment.get('next_step_suggestion')}'")
        
        # ============================================
        # [LOOP] STEP 4: Conditional Expansion
        # ============================================
        all_evidence = list(initial_evidence)
        max_iterations = 2
        iteration = 0
        
        while not assessment.get('is_sufficient', False) and iteration < max_iterations:
            iteration += 1
            next_step = assessment.get('next_step_suggestion', 'terminate')
            
            print(f"[RetrievalAgent] [LOOP] Iteration {iteration}/{max_iterations}: Expanding with strategy '{next_step}'")
            
            if next_step == 'terminate':
                print("[RetrievalAgent] [LOOP] Assessment suggests termination, exiting loop")
                break
            
            # Perform expansion based on suggestion
            expansion_evidence = self._perform_expansion(
                query, query_analysis, found_entities, next_step, all_evidence
            )
            
            if expansion_evidence:
                # Deduplicate evidence
                seen_evidence = set(all_evidence)
                for ev in expansion_evidence:
                    if ev not in seen_evidence:
                        all_evidence.append(ev)
                        seen_evidence.add(ev)
                
                print(f"[RetrievalAgent] [LOOP] Expansion added {len(expansion_evidence)} new evidence items "
                      f"(total: {len(all_evidence)})")
                
                # Re-assess with expanded evidence
                print("[RetrievalAgent] [THINK] Re-assessing evidence after expansion...")
                assessment = self._assess_evidence_sufficiency(query_analysis, all_evidence)
                print(f"[RetrievalAgent] [THINK] Re-assessment: is_sufficient={assessment.get('is_sufficient')}, "
                      f"reason='{assessment.get('reason')}'")
            else:
                print("[RetrievalAgent] [LOOP] Expansion produced no new evidence, exiting loop")
                break
        
        # ============================================
        # STEP 5: Assemble & Return
        # ============================================
        print(f"[RetrievalAgent] [FINAL] Assembling final result: {len(all_entities)} entities, {len(all_evidence)} evidence items")
        
        # Limit entities to reasonable number (max 5 for most queries, 10 for complex queries)
        max_entities = 10 if complexity_level == "complex" else 5
        final_entities = all_entities[:max_entities]
        
        # Limit evidence to top_k
        final_evidence = all_evidence[:top_k * 2]  # Allow more evidence than entities
        
        # 构建检索路径
        retrieval_path = ["retrieval-agent"]
        if initial_evidence:
            retrieval_path.append("initial-retrieval")
        if len(all_evidence) > len(initial_evidence):
            retrieval_path.append("expansion")
        
        result = RetrievalResult(
            query=query,
            entities=final_entities,
            evidence=final_evidence,
            retrieval_path=retrieval_path
        )
        result.extracted_entity_names = required_entities
        
        print(f"[RetrievalAgent] [END] Retrieval complete. Returning {len(result.entities)} entities, {len(result.evidence)} evidence items")
        return result
    
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
                print(f"[RetrievalAgent] [VALIDATE] Entity '{name}' found in graph")
            else:
                print(f"[RetrievalAgent] [VALIDATE] Entity '{name}' not found in graph, skipping")
        
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
            graph = self.cogmem.graph._g
            entity_name_lower = entity_name.lower()
            
            for node_id, node_data in graph.nodes(data=True):
                if not node_data:
                    continue
                
                # Check Person nodes by canonical_name
                # Use NodeTypeEnum.PERSON.value ("person") instead of "Person"
                node_type = node_data.get("node_type")
                if node_type == "person" or node_type == NodeTypeEnum.PERSON.value:
                    canonical_name = node_data.get("canonical_name", "")
                    if canonical_name and canonical_name.lower() == entity_name_lower:
                        return True
                    
                    # Check aliases
                    aliases = node_data.get("aliases", [])
                    if aliases:
                        for alias in aliases:
                            if isinstance(alias, str) and alias.lower() == entity_name_lower:
                                return True
            
            # Check Entity nodes
            entities = self.cogmem.graph.get_entities_by_name(entity_name.lower())
            if entities:
                return True
            
            return False
        except Exception as e:
            print(f"[RetrievalAgent] [ERROR] Failed to check entity existence: {e}")
            return False
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query structure and requirements using QUERY_ANALYSIS_SYSTEM.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with question_type, required_entities, complexity_level, temporal_requirement
        """
        try:
            response = self.llm.chat(
                QUERY_ANALYSIS_SYSTEM,
                QUERY_ANALYSIS_USER.format(query=query)
            )
            
            analysis = safe_json_loads(response, default={})
            if not isinstance(analysis, dict):
                return {}
            
            return analysis
        except Exception as e:
            print(f"[RetrievalAgent] [ERROR] Query analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _assess_evidence_sufficiency(
        self, 
        query_analysis: Dict[str, Any], 
        evidence: List[str]
    ) -> Dict[str, Any]:
        """Assess whether retrieved evidence is sufficient using EVIDENCE_ASSESSMENT_SYSTEM.
        
        Args:
            query_analysis: Query analysis result from _analyze_query
            evidence: List of evidence strings
            
        Returns:
            Dictionary with is_sufficient, reason, next_step_suggestion
        """
        try:
            # Format evidence for prompt
            evidence_text = "\n".join([f"  - '{ev}'" for ev in evidence[:20]])  # Limit to 20 items
            if not evidence_text:
                evidence_text = "  (no evidence retrieved)"
            
            query_analysis_str = json.dumps(query_analysis, ensure_ascii=False)
            
            response = self.llm.chat(
                EVIDENCE_ASSESSMENT_SYSTEM,
                EVIDENCE_ASSESSMENT_USER.format(
                    query_analysis=query_analysis_str,
                    evidence=evidence_text
                )
            )
            
            assessment = safe_json_loads(response, default={})
            if not isinstance(assessment, dict):
                return {"is_sufficient": False, "reason": "Assessment parsing failed", "next_step_suggestion": "graph_traversal"}
            
            # Ensure required fields exist
            if "is_sufficient" not in assessment:
                assessment["is_sufficient"] = False
            if "reason" not in assessment:
                assessment["reason"] = "No reason provided"
            if "next_step_suggestion" not in assessment:
                assessment["next_step_suggestion"] = "graph_traversal"
            
            return assessment
        except Exception as e:
            print(f"[RetrievalAgent] [ERROR] Evidence assessment failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "is_sufficient": False,
                "reason": f"Assessment error: {str(e)}",
                "next_step_suggestion": "graph_traversal"
            }
    
    def _perform_expansion(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        entities: List[Entity],
        next_step: str,
        current_evidence: List[str]
    ) -> List[str]:
        """Perform evidence expansion based on next_step_suggestion.
        
        Args:
            query: Original query
            query_analysis: Query analysis result
            entities: List of entities found
            next_step: Expansion strategy (graph_traversal, query_rewrite, etc.)
            current_evidence: Current evidence list
            
        Returns:
            List of new evidence items
        """
        expansion_evidence = []
        
        if next_step == "graph_traversal":
            print("[RetrievalAgent] [EXPAND] Performing deeper graph traversal (2-3 hop)...")
            # Perform deeper traversal (2-3 hop)
            for entity in entities:
                person = self.cogmem.graph.get_person(entity.id)
                if person:
                    if hasattr(self.cogmem.retriever, '_traverse_person_structured_nodes'):
                        person_evidence = self.cogmem.retriever._traverse_person_structured_nodes(
                            entity,
                            max_depth=3,  # Deeper traversal
                            max_nodes=20,  # More nodes
                            query=query
                        )
                        expansion_evidence.extend(person_evidence)
                        print(f"[RetrievalAgent] [EXPAND] Deep traversal found {len(person_evidence)} items for {entity.name}")
        
        elif next_step == "multi_hop_expansion":
            print("[RetrievalAgent] [EXPAND] Performing multi-hop expansion...")
            # Use multi-hop reasoning strategy
            if entities:
                from ..retrieval.graph_traversal import GraphTraversal
                traversal = GraphTraversal(self.cogmem.graph)
                expanded_entities = traversal.expand_with_importance_scoring(
                    entities[:3],
                    k_hop=3,
                    max_entities_per_hop=15
                )
                # Extract evidence from expanded entities
                for e in expanded_entities:
                    if e.id not in [ent.id for ent in entities]:
                        # Get connected nodes
                        graph = self.cogmem.graph._g
                        if e.id in graph:
                            for _, target_id, key, edge_data in graph.out_edges(e.id, keys=True, data=True):
                                fact = self.cogmem.graph.get_fact(target_id)
                                if fact:
                                    expansion_evidence.append(self.cogmem.retriever._format_fact_evidence(fact))
                                else:
                                    event = self.cogmem.graph.get_event(target_id)
                                    if event:
                                        expansion_evidence.append(self.cogmem.retriever._format_event_evidence(event))
                print(f"[RetrievalAgent] [EXPAND] Multi-hop expansion found {len(expansion_evidence)} items")
        
        elif next_step == "expand_entities":
            print("[RetrievalAgent] [EXPAND] Expanding to related entities...")
            # Find related entities through community or graph connections
            if entities and self.cogmem.community_store:
                # Get entities from same communities
                for entity in entities[:3]:
                    communities = self.cogmem.community_store.get_entity_communities(entity.id)
                    for comm in communities[:2]:  # Limit to 2 communities
                        members = self.cogmem.community_store.get_community_entities(comm.id)
                        for member_id in members[:5]:  # Limit to 5 members per community
                            if member_id != entity.id:
                                member_entity = self.cogmem.graph.get_entity(member_id)
                                if member_entity:
                                    # Get evidence from related entity
                                    person = self.cogmem.graph.get_person(member_id)
                                    if person and hasattr(self.cogmem.retriever, '_traverse_person_structured_nodes'):
                                        related_evidence = self.cogmem.retriever._traverse_person_structured_nodes(
                                            member_entity,
                                            max_depth=1,
                                            max_nodes=5,
                                            query=query
                                        )
                                        expansion_evidence.extend(related_evidence)
            print(f"[RetrievalAgent] [EXPAND] Entity expansion found {len(expansion_evidence)} items")
        
        elif next_step == "temporal_refinement":
            print("[RetrievalAgent] [EXPAND] Refining temporal search...")
            # For temporal questions, search for events with specific timestamps
            for entity in entities:
                person = self.cogmem.graph.get_person(entity.id)
                if person:
                    if hasattr(self.cogmem.retriever, '_traverse_person_structured_nodes'):
                        # Use deeper traversal to find temporal events
                        temporal_evidence = self.cogmem.retriever._traverse_person_structured_nodes(
                            entity,
                            max_depth=2,
                            max_nodes=15,
                            query=query
                        )
                        # Filter for evidence with time information
                        for ev in temporal_evidence:
                            if "Time:" in ev or any(keyword in ev.lower() for keyword in ["2023", "2024", "date", "time"]):
                                expansion_evidence.append(ev)
            print(f"[RetrievalAgent] [EXPAND] Temporal refinement found {len(expansion_evidence)} items")
        
        elif next_step == "query_rewrite":
            print("[RetrievalAgent] [EXPAND] Query rewrite not implemented, using graph traversal instead")
            # Fallback to graph traversal if query rewrite is not implemented
            for entity in entities:
                person = self.cogmem.graph.get_person(entity.id)
                if person:
                    if hasattr(self.cogmem.retriever, '_traverse_person_structured_nodes'):
                        person_evidence = self.cogmem.retriever._traverse_person_structured_nodes(
                            entity,
                            max_depth=2,
                            max_nodes=15,
                            query=query
                        )
                        expansion_evidence.extend(person_evidence)
        
        return expansion_evidence
    
    def _fallback_retrieval(self, query: str, top_k: int) -> RetrievalResult:
        """Fallback retrieval method when query analysis fails.
        
        Args:
            query: User query
            top_k: Number of results
            
        Returns:
            RetrievalResult using simple entity-based retrieval
        """
        print("[RetrievalAgent] [FALLBACK] Using fallback retrieval method")
        entity_names = self._extract_entity_names_from_query(query)
        
        if entity_names:
            found_entities = self._lookup_entities_in_graph(entity_names)
            if found_entities:
                return self._entity_based_retrieval_with_entities(query, found_entities, top_k)
        
        # Final fallback to semantic search
        return self._semantic_search(query, top_k)
    
    def _select_retrieval_strategy(self, query: str) -> dict:
        """使用LLM智能选择检索策略"""
        response = self.llm.chat(
            RETRIEVAL_STRATEGY_SELECTION_SYSTEM,
            RETRIEVAL_STRATEGY_SELECTION_USER.format(query=query)
        )
        
        decision = safe_json_loads(response)
        return {
            'strategy': decision.get('strategy', 'semantic_qa_search'),
            'confidence': decision.get('confidence', 0.5),
            'reason': decision.get('reason', '')
        }
    
    def _exact_qa_match(self, query: str, top_k: int) -> RetrievalResult:
        """精确匹配：在知识库中查找包含查询关键词的实体"""
        # 在图中查找包含查询关键词的实体
        entities = []
        query_lower = query.lower()
        for entity in self.cogmem.graph.iter_entities():
            if query_lower in entity.name.lower() or query_lower in entity.description.lower():
                entities.append(entity)
        
        return RetrievalResult(
            query=query,
            entities=entities[:top_k],
            evidence=self._extract_evidence_from_entities(entities[:top_k])
        )
    
    def _semantic_search(self, query: str, top_k: int) -> RetrievalResult:
        """语义搜索：使用向量相似度搜索相关文档（增强时序问题处理）"""
        # 检测是否是时间问题
        is_temporal_query = any(keyword in query.lower() for keyword in ["when", "time", "date", "什么时候"])
        
        # 使用现有的向量检索
        query_vec = self.cogmem._embed.encode([query])[0]
        
        # 从chunk_store检索
        if self.cogmem.chunk_store and len(self.cogmem.chunk_store._chunks) > 0:
            # 对于时间问题，检索更多结果以便筛选
            search_k = top_k * 2 if is_temporal_query else top_k
            results = self.cogmem.chunk_store.search(query_vec, top_k=search_k)
            
            evidence = []
            for chunk, score in results:
                if score >= self.cfg.thresholds.min_similarity:
                    chunk_text = chunk.text
                    
                    # 对于时间问题，优先包含Session Date或时间信息的证据
                    if is_temporal_query:
                        import re
                        # 检查是否包含时间信息
                        has_time_info = (
                            "[Session Date:" in chunk_text or
                            re.search(r'\d{4}', chunk_text) is not None or  # 年份
                            re.search(r'\d{1,2} \w+ \d{4}', chunk_text) is not None or  # 日期
                            any(keyword in chunk_text.lower() for keyword in ['yesterday', 'last week', 'last year', 'ago'])
                        )
                        
                        if has_time_info:
                            # 时间相关的证据优先（插入到前面）
                            evidence.insert(0, chunk_text)
                        elif len(evidence) < top_k:
                            # 如果还没达到top_k，也加入
                            evidence.append(chunk_text)
                    else:
                        evidence.append(chunk_text)
                    
                    if len(evidence) >= top_k:
                        break
            
            # 从证据中提取实体
            entities = self._extract_entities_from_evidence(evidence)
            
            return RetrievalResult(query=query, entities=entities, evidence=evidence[:top_k])
        
        # 降级到实体向量检索
        return self.cogmem.retrieve(query, top_k)
    
    def _extract_entity_names_from_query(self, query: str) -> List[str]:
        """Extract entity names from query using QUERY_ENTITY_SYSTEM prompt.
        
        This method ONLY extracts entity names from the query text.
        It does NOT perform graph lookup - that's done separately.
        
        Args:
            query: User query string
            
        Returns:
            List of entity names (e.g., ["Caroline", "Melanie"])
        """
        try:
            # Use QUERY_ENTITY_SYSTEM to extract entity names
            response = self.llm.chat(
                QUERY_ENTITY_SYSTEM,
                QUERY_ENTITY_USER.format(query=query)
            )
            
            # Parse entity names
            entity_names = safe_json_loads(response, default=[])
            if not isinstance(entity_names, list):
                return []
            
            # Return cleaned entity names
            return [str(name).strip() for name in entity_names if isinstance(name, str) and str(name).strip()]
        except Exception as e:
            print(f"[RetrievalAgent] [ERROR] Error extracting entity names from query: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _lookup_entities_in_graph(self, entity_names: List[str]) -> List[Entity]:
        """Look up entity names in the graph by directly traversing all nodes.
        
        This method searches both Person nodes (with canonical_name) and Entity nodes (with name).
        It directly accesses the graph's underlying NetworkX structure to find all node types.
        
        Args:
            entity_names: List of entity names to look up
            
        Returns:
            List of Entity objects found in graph (Person nodes are converted to Entity format)
        """
        found_entities = []
        name_set = {name.strip().lower() for name in entity_names if isinstance(name, str)}
        
        # Directly traverse all nodes in the graph
        graph = self.cogmem.graph._g
        total_nodes = graph.number_of_nodes()
        
        print(f"[RetrievalAgent] [DEBUG] Looking up {entity_names} in graph with {total_nodes} total nodes")
        
        # First, try using get_person and get_entity methods which handle parsing correctly
        for name in entity_names:
            name_lower = name.strip().lower()
            
            # Try to find Person nodes by iterating and using get_person
            for node_id in graph.nodes():
                try:
                    person = self.cogmem.graph.get_person(node_id)
                    if person:
                        # Check canonical_name
                        if person.canonical_name.lower() == name_lower:
                            # Generate description from persona traits or aliases
                            description_parts = []
                            if person.persona_traits:
                                description_parts.append(f"Traits: {', '.join(person.persona_traits[:3])}")
                            if person.aliases:
                                description_parts.append(f"Aliases: {', '.join(person.aliases)}")
                            description = " | ".join(description_parts) if description_parts else f"Person: {person.canonical_name}"
                            
                            entity = Entity(
                                id=person.id,
                                name=person.canonical_name,
                                description=description,
                                type="world",
                                subtype="person",
                                aliases=person.aliases,
                                confidence=person.confidence
                            )
                            found_entities.append(entity)
                            print(f"[RetrievalAgent] [DEBUG] Found Person via canonical_name: {person.canonical_name} (id: {person.id})")
                            continue
                        
                        # Check aliases
                        if person.aliases:
                            for alias in person.aliases:
                                if isinstance(alias, str) and alias.strip().lower() == name_lower:
                                    # Generate description from persona traits or aliases
                                    description_parts = []
                                    if person.persona_traits:
                                        description_parts.append(f"Traits: {', '.join(person.persona_traits[:3])}")
                                    if person.aliases:
                                        description_parts.append(f"Aliases: {', '.join(person.aliases)}")
                                    description = " | ".join(description_parts) if description_parts else f"Person: {person.canonical_name}"
                                    
                                    entity = Entity(
                                        id=person.id,
                                        name=person.canonical_name,
                                        description=description,
                                        type="world",
                                        subtype="person",
                                        aliases=person.aliases,
                                        confidence=person.confidence
                                    )
                                    found_entities.append(entity)
                                    print(f"[RetrievalAgent] [DEBUG] Found Person via alias: {alias} -> {person.canonical_name} (id: {person.id})")
                                    break
                except Exception:
                    # Not a Person node, continue
                    pass
            
            # Try to find Entity nodes
            for node_id in graph.nodes():
                try:
                    entity = self.cogmem.graph.get_entity(node_id)
                    if entity:
                        # Check name
                        if entity.name.lower() == name_lower:
                            found_entities.append(entity)
                            print(f"[RetrievalAgent] [DEBUG] Found Entity: {entity.name} (id: {entity.id})")
                            continue
                        
                        # Check aliases
                        if entity.aliases:
                            for alias in entity.aliases:
                                if isinstance(alias, str) and alias.strip().lower() == name_lower:
                                    found_entities.append(entity)
                                    print(f"[RetrievalAgent] [DEBUG] Found Entity via alias: {alias} -> {entity.name} (id: {entity.id})")
                                    break
                except Exception:
                    # Not an Entity node, continue
                    pass
        
        # Remove duplicates
        seen_ids = set()
        unique_entities = []
        for e in found_entities:
            if e.id not in seen_ids:
                unique_entities.append(e)
                seen_ids.add(e.id)
        
        print(f"[RetrievalAgent] [DEBUG] Total found: {len(unique_entities)} unique entities")
        return unique_entities
    
    def _extract_entities_from_query(self, query: str) -> List[Entity]:
        """Legacy method - combines entity name extraction and graph lookup.
        
        DEPRECATED: Use _extract_entity_names_from_query + _lookup_entities_in_graph instead.
        This method is kept for backward compatibility.
        """
        entity_names = self._extract_entity_names_from_query(query)
        if not entity_names:
            return []
        return self._lookup_entities_in_graph(entity_names)
    
    def _entity_based_retrieval_with_entities(self, query: str, entities: List[Entity], top_k: int) -> RetrievalResult:
        """基于实体的检索：使用已提供的实体进行图搜索
        
        Args:
            query: User query
            entities: List of Entity objects already found in graph
            top_k: Number of results to return
            
        Returns:
            RetrievalResult with entities and structured evidence
        """
        if not entities:
            # No entities provided, return empty result
            print("[RetrievalAgent] [DEBUG] _entity_based_retrieval_with_entities called with empty entities list")
            return RetrievalResult(query=query, entities=[], evidence=[])
        
        print(f"[RetrievalAgent] [DEBUG] _entity_based_retrieval_with_entities called with {len(entities)} entities")
        
        # Use the retriever's graph traversal for Person entities
        all_entities = list(entities)
        structured_evidence = []
        
        # For each Person entity, traverse graph to get structured nodes
        for entity in entities:
            person = self.cogmem.graph.get_person(entity.id)
            if person:
                # Use retriever's traversal method if available
                if hasattr(self.cogmem.retriever, '_traverse_person_structured_nodes'):
                    person_evidence = self.cogmem.retriever._traverse_person_structured_nodes(
                        entity,
                        max_depth=2,
                        max_nodes=20,
                        query=query
                    )
                    structured_evidence.extend(person_evidence)
                    if person_evidence:
                        print(f"[RetrievalAgent] [DEBUG] Found {len(person_evidence)} structured evidence items for {entity.name}")
            else:
                # For non-Person entities, try to find connected Facts/Events
                graph = self.cogmem.graph._g
                if entity.id in graph:
                    for _, target_id, key, edge_data in graph.out_edges(entity.id, keys=True, data=True):
                        fact = self.cogmem.graph.get_fact(target_id)
                        if fact:
                            structured_evidence.append(self.cogmem.retriever._format_fact_evidence(fact))
                        else:
                            event = self.cogmem.graph.get_event(target_id)
                            if event:
                                structured_evidence.append(self.cogmem.retriever._format_event_evidence(event))
        
        # If no structured evidence, try to get evidence from entities
        if not structured_evidence:
            print("[RetrievalAgent] [DEBUG] No structured evidence found, using entity evidence extraction")
            evidence = self._extract_evidence_from_entities(entities)
        else:
            evidence = structured_evidence[:10]
            print(f"[RetrievalAgent] [DEBUG] Using {len(evidence)} structured evidence items")
        
        return RetrievalResult(query=query, entities=all_entities[:top_k], evidence=evidence)
    
    def _entity_based_retrieval(self, query: str, top_k: int) -> RetrievalResult:
        """基于实体的检索：使用已提取的实体进行图搜索
        
        DEPRECATED: This method re-extracts entities. Use _entity_based_retrieval_with_entities instead.
        """
        # Extract entities from query
        extracted_entities = self._extract_entities_from_query(query)
        
        if not extracted_entities:
            # No entities found, return empty result
            return RetrievalResult(query=query, entities=[], evidence=[])
        
        return self._entity_based_retrieval_with_entities(query, extracted_entities, top_k)
    
    def _community_expansion(self, query: str, top_k: int) -> RetrievalResult:
        """社区扩展：基于社区关系扩展检索"""
        # 先获取基础实体
        base_result = self._entity_based_retrieval(query, top_k // 2)
        
        if not base_result.entities:
            return base_result
        
        # 使用社区扩展
        if self.cogmem.community_store:
            from ..retrieval.graph_traversal import GraphTraversal
            traversal = GraphTraversal(self.cogmem.graph)
            
            expanded_entities = traversal.expand_with_importance_scoring(
                base_result.entities[:5],
                k_hop=2,
                max_entities_per_hop=10
            )
            
            # 合并结果
            all_entities = list(base_result.entities)
            seen_ids = {e.id for e in all_entities}
            for e in expanded_entities:
                if e.id not in seen_ids:
                    all_entities.append(e)
                    seen_ids.add(e.id)
            
            evidence = self._extract_evidence_from_entities(all_entities[:top_k])
            
            return RetrievalResult(
                query=query,
                entities=all_entities[:top_k],
                evidence=evidence
            )
        
        return base_result
    
    def _multi_hop_reasoning(self, query: str, top_k: int) -> RetrievalResult:
        """多跳推理：使用图遍历进行复杂推理"""
        # 使用社区扩展作为基础，但增加跳数
        base_result = self._community_expansion(query, top_k)
        
        if not base_result.entities:
            return base_result
        
        # 更深的图遍历
        from ..retrieval.graph_traversal import GraphTraversal
        traversal = GraphTraversal(self.cogmem.graph)
        
        expanded = traversal.expand_with_importance_scoring(
            base_result.entities[:3],
            k_hop=3,
            max_entities_per_hop=15
        )
        
        all_entities = list(base_result.entities)
        seen_ids = {e.id for e in all_entities}
        for e in expanded:
            if e.id not in seen_ids:
                all_entities.append(e)
                seen_ids.add(e.id)
        
        evidence = self._extract_evidence_from_entities(all_entities[:top_k])
        
        return RetrievalResult(
            query=query,
            entities=all_entities[:top_k],
            evidence=evidence
        )
    
    def _extract_evidence_from_entities(self, entities: list[Entity]) -> list[str]:
        """从实体中提取证据"""
        if not entities:
            return []
        
        evidence = []
        entity_ids = [e.id for e in entities]
        
        # 从evidence store获取
        if self.cogmem.indexer and hasattr(self.cogmem.indexer, 'evidence'):
            evidence_chunks = self.cogmem.indexer.evidence.build_answer_evidence(entity_ids)
            evidence.extend(evidence_chunks)
        
        return evidence[:10]  # 限制数量
    
    def _extract_entities_from_evidence(self, evidence: list[str]) -> list[Entity]:
        """从证据中提取实体"""
        # 简化实现：使用现有检索器提取实体
        if not evidence:
            return []
        
        # 合并证据文本
        combined_text = "\n".join(evidence[:5])
        
        # 使用LLM提取实体
        from ..extraction.entity_extractor import EntityExtractor
        extractor = EntityExtractor(self.cfg.extraction, self.llm)
        entities = extractor.extract(combined_text, context_type="dialog")
        
        return entities
    
    def _merge_results(self, result1: RetrievalResult, result2: RetrievalResult, top_k: int) -> RetrievalResult:
        """合并两个检索结果"""
        # 合并实体（去重）
        all_entities = list(result1.entities)
        seen_ids = {e.id for e in all_entities}
        
        for e in result2.entities:
            if e.id not in seen_ids:
                all_entities.append(e)
                seen_ids.add(e.id)
        
        # 合并证据（去重）
        all_evidence = list(result1.evidence)
        seen_evidence = set(result1.evidence)
        
        for ev in result2.evidence:
            if ev not in seen_evidence:
                all_evidence.append(ev)
                seen_evidence.add(ev)
        
        return RetrievalResult(
            query=result1.query,
            entities=all_entities[:top_k],
            evidence=all_evidence[:10]
        )
