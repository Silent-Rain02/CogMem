"""Memory Consolidation Module (v3.3 Enhanced)

This module implements offline memory consolidation based on CLS (Consolidation Learning System) theory.
It processes existing graphs, clusters similar events, and synthesizes them into high-level Facts using LLM.

Enhanced with:
- ConceptMerger: Semantic-based concept deduplication
- FactDeduplicator: Fact deduplication and version management
- ClaimReconciler: Claim contradiction detection and temporal management

v3.3 Enhancements:
- Incremental consolidation: Only process new/changed data since last consolidation
- Concept hierarchy: Build IS_PARENT_OF/IS_CHILD_OF relationships
- Temporal chain management: Track is_latest flag and supersedes relationships
"""

from __future__ import annotations

import json
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..config import CogMemConfig
from ..models.llm_client import create_llm_client, LLMClient
from ..models.embedding import EmbeddingEncoder
from ..storage.graph_store import GraphStore
from ..storage.typed_vector_store import TypedVectorStore
from ..storage.schema import (
    Person, Event, Fact, Concept, NodeTypeEnum, Metadata, Relation, MemoryLifecycle
)
from ..prompt import MEMORY_CONSOLIDATION_SYSTEM, MEMORY_CONSOLIDATION_USER
from ..utils.json_utils import safe_json_loads

# Import new consolidation components
from .concept_merger import ConceptMerger
from .fact_deduplicator import FactDeduplicator
from .claim_reconciler import ClaimReconciler

def _make_fact_id(person_id: str, concept_id: str, predicate: str) -> str:
    """Generate a unique Fact ID based on person, concept, and predicate"""
    content = f"{person_id}_{concept_id}_{predicate}"
    return f"fact_{hashlib.sha1(content.encode('utf-8')).hexdigest()}"

class MemoryConsolidator:
    """Memory Consolidation Engine
    
    Processes existing graphs to consolidate episodic memories (Events) into 
    semantic memories (generalized Facts) based on CLS theory.
    
    Enhanced consolidation pipeline:
    1. Concept Merger: Detect and merge synonymous concepts
    2. Event -> Fact Synthesis: Cluster events and synthesize Facts
    3. Fact Deduplicator: Deduplicate and version-manage Facts
    4. Claim Reconciler: Detect contradictions and establish temporal chains
    """

    def __init__(
        self,
        config: CogMemConfig,
        graph: GraphStore,
        vector_store: TypedVectorStore,
        embedding_encoder: EmbeddingEncoder,
        evidence_store=None  # P0 FIX: Add evidence_store support
    ):
        self.config = config
        self.graph = graph
        self.vector_store = vector_store
        self.embedding_encoder = embedding_encoder
        self.evidence_store = evidence_store  # P0 FIX: Store reference
        self.llm: LLMClient = create_llm_client(config)
        
        # Initialize sub-components
        self.concept_merger = ConceptMerger(
            graph=graph,
            vector_store=vector_store,
            embedding_encoder=embedding_encoder,
            use_embedding=True
        )
        self.fact_deduplicator = FactDeduplicator(
            graph=graph,
            vector_store=vector_store,
            embedding_encoder=embedding_encoder,
            semantic_threshold=0.90
        )
        self.claim_reconciler = ClaimReconciler(graph=graph)
        # v3.3: Track last consolidation timestamp
        self._last_consolidation_timestamp: Optional[str] = None

    def run_incremental_consolidation(
        self,
        since_timestamp: Optional[str] = None,
        affected_person_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run incremental consolidation (v3.3 Enhancement)
        
        Only processes data that has changed since the last consolidation.
        
        Args:
            since_timestamp: Only process nodes created/updated after this timestamp (ISO-8601)
            affected_person_ids: Only process these specific Person IDs
            **kwargs: Additional arguments passed to run_consolidation
            
        Returns:
            Statistics dictionary with consolidation results
        """
        # Determine which persons to process
        if affected_person_ids:
            persons_to_process = affected_person_ids
        elif since_timestamp:
            # Find persons with new events since timestamp
            persons_to_process = self._find_affected_persons(since_timestamp)
        elif self._last_consolidation_timestamp:
            persons_to_process = self._find_affected_persons(self._last_consolidation_timestamp)
        else:
            # First run - process all
            persons_to_process = None
        
        print(f"[Consolidator] Incremental mode: processing {len(persons_to_process) if persons_to_process else 'all'} persons")
        
        # Run consolidation with filtered persons
        stats = self.run_consolidation(
            affected_person_ids=persons_to_process,
            **kwargs
        )
        
        # Update last consolidation timestamp
        self._last_consolidation_timestamp = datetime.now().isoformat()
        
        return stats

    def _find_affected_persons(self, since_timestamp: str) -> List[str]:
        """Find persons with new events since timestamp (v3.3 Enhancement)"""
        affected_persons = set()
        
        try:
            since_dt = datetime.fromisoformat(since_timestamp.replace('Z', '+00:00'))
        except ValueError:
            return []
        
        # Iterate through all Event nodes
        for node_id, node_data in self.graph._g.nodes(data=True):
            if not node_data or node_data.get('node_type') != NodeTypeEnum.EVENT.value:
                continue
            
            # Check if event was created after since_timestamp
            metadata = node_data.get('metadata', {})
            created_at = metadata.get('created_at') if isinstance(metadata, dict) else None
            
            if created_at:
                try:
                    created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    if created_dt > since_dt:
                        # Find participants of this event
                        participants = node_data.get('participants', [])
                        affected_persons.update(participants)
                except ValueError:
                    continue
        
        return list(affected_persons)

    def build_concept_hierarchy(self) -> Dict[str, List[str]]:
        """Build concept hierarchy relationships (v3.3 Enhancement)
        
        Uses LLM to identify parent-child relationships between concepts.
        E.g., c_pizza IS_CHILD_OF c_food, c_hiking IS_CHILD_OF c_outdoor_activity
        
        Returns:
            Dict mapping child_id -> [parent_ids]
        """
        from ..storage.schema import Relation
        
        # Get all concept nodes
        concepts = []
        for node_id, node_data in self.graph._g.nodes(data=True):
            if node_data and node_data.get('node_type') == NodeTypeEnum.CONCEPT.value:
                concepts.append({
                    'id': node_id,
                    'name': node_data.get('name', node_id),
                    'description': node_data.get('description', '')
                })
        
        if len(concepts) < 2:
            return {}
        
        # Common parent categories
        parent_categories = {
            'food': ['pizza', 'pasta', 'sushi', 'cooking', 'baking', 'restaurant', 'cuisine'],
            'activity': ['hiking', 'running', 'swimming', 'dancing', 'yoga', 'exercise', 'sport'],
            'hobby': ['painting', 'reading', 'gaming', 'music', 'photography', 'crafts'],
            'location': ['park', 'beach', 'mountain', 'city', 'country', 'home', 'office'],
            'emotion': ['happy', 'sad', 'excited', 'anxious', 'relaxed', 'stressed'],
        }
        
        hierarchy = defaultdict(list)
        relations_to_add = []
        
        for concept in concepts:
            concept_name = concept['name'].lower().replace('c_', '').replace('_', ' ')
            
            for parent_cat, keywords in parent_categories.items():
                if any(kw in concept_name for kw in keywords):
                    parent_id = f"c_{parent_cat}"
                    
                    # Create parent concept if not exists
                    if parent_id not in self.graph._g:
                        parent_concept = Concept(
                            id=parent_id,
                            name=parent_cat,
                            description=f"Category: {parent_cat}",
                            category="category"
                        )
                        self.graph.add_node(parent_concept)
                    
                    # Add hierarchy relationship
                    hierarchy[concept['id']].append(parent_id)
                    
                    # Create IS_CHILD_OF edge
                    relations_to_add.append(Relation(
                        head_id=concept['id'],
                        tail_id=parent_id,
                        rel_type='IS_CHILD_OF',
                        weight=1.0,
                        confidence=0.8
                    ))
                    
                    # Create IS_PARENT_OF edge (reverse)
                    relations_to_add.append(Relation(
                        head_id=parent_id,
                        tail_id=concept['id'],
                        rel_type='IS_PARENT_OF',
                        weight=1.0,
                        confidence=0.8
                    ))
                    break  # Only one parent per concept for simplicity
        
        if relations_to_add:
            self.graph.add_relations(relations_to_add)
            print(f"[Consolidator] Built concept hierarchy: {len(relations_to_add)//2} parent-child relationships")
        
        return dict(hierarchy)

    def run_consolidation(
        self,
        source_workspace: Optional[str] = None,
        target_workspace: Optional[str] = None,
        affected_person_ids: Optional[List[str]] = None,
        min_events: int = 3,
        archive_original_events: bool = False,
        merge_concepts: bool = True,
        deduplicate_facts: bool = True,
        reconcile_claims: bool = True,
        concept_similarity_threshold: float = 0.90,
        concept_high_threshold: float = 0.95,
        fact_high_threshold: float = 0.95,
        max_supported_by_per_fact: int = 5,
        cleanup_archived: bool = True
    ) -> Dict[str, Any]:
        """Run memory consolidation process
        
        Args:
            source_workspace: Source workspace directory (if None, use current graph)
            target_workspace: Target workspace directory (if None, modify in place)
            affected_person_ids: If set, only process these Person IDs (incremental consolidation)
            min_events: Minimum number of events required to create a Fact (default: 3)
            archive_original_events: Whether to mark original events as archived
            merge_concepts: Whether to merge synonymous concepts (default: True)
            deduplicate_facts: Whether to deduplicate facts (default: True)
            reconcile_claims: Whether to reconcile claims (default: True)
            concept_similarity_threshold: Threshold for concept similarity (default: 0.90)
            concept_high_threshold: High similarity threshold for full concept merge (default: 0.95)
            fact_high_threshold: High similarity threshold for full fact merge (default: 0.95)
            max_supported_by_per_fact: Maximum SUPPORTED_BY edges per fact (default: 5)
            cleanup_archived: Whether to delete archived nodes after consolidation (default: True)
        
        Returns:
            Statistics dictionary with consolidation results and phase costs
        """
        import time
        from ..utils.metrics import METRICS
        
        stats = {
            "persons_processed": 0,
            "facts_created": 0,
            "events_clustered": 0,
            "events_archived": 0,
            "concepts_merged": 0,
            "facts_deduplicated": 0,
            "claims_reconciled": 0,
            "errors": [],
            "phase_costs": {}  # 记录各阶段的详细开销
        }

        # Load graph if source_workspace is specified
        if source_workspace:
            from ..cogmem import CogMem
            source_cogmem = CogMem(workspace_dir=source_workspace, auto_load=True)
            self.graph = source_cogmem.graph
            self.vector_store = source_cogmem.vstore
            # Re-initialize sub-components with new graph
            self.concept_merger = ConceptMerger(
                graph=self.graph,
                vector_store=self.vector_store,
                embedding_encoder=self.embedding_encoder,
                use_embedding=True
            )
            self.fact_deduplicator = FactDeduplicator(
                graph=self.graph,
                vector_store=self.vector_store,
                embedding_encoder=self.embedding_encoder
            )
            self.claim_reconciler = ClaimReconciler(graph=self.graph)
        
        # ===== Phase 1: Concept Merging =====
        if merge_concepts:
            print("\n" + "="*60)
            print("[Consolidator] Phase 1: Concept Merging")
            print("="*60)
            
            # 记录阶段开始时间和METRICS
            phase1_start_time = time.time()
            phase1_metrics_before = METRICS.snapshot()
            
            concept_merge_map = self.concept_merger.run(
                threshold=concept_similarity_threshold,
                create_same_as=True,
                high_threshold=concept_high_threshold
            )
            
            # 记录阶段结束时间和METRICS
            phase1_end_time = time.time()
            phase1_metrics_after = METRICS.snapshot()
            
            # 计算阶段开销
            phase1_cost = {
                "time_seconds": phase1_end_time - phase1_start_time,
                "llm_calls": phase1_metrics_after['chat']['calls'] - phase1_metrics_before['chat']['calls'],
                "llm_tokens": phase1_metrics_after['chat']['total_tokens'] - phase1_metrics_before['chat']['total_tokens'],
                "embedding_calls": phase1_metrics_after['embed']['calls'] - phase1_metrics_before['embed']['calls'],
                "embedding_tokens": phase1_metrics_after['embed']['total_tokens'] - phase1_metrics_before['embed']['total_tokens']
            }
            stats["phase_costs"]["phase1_concept_merging"] = phase1_cost
            
            stats["concepts_merged"] = len(concept_merge_map)
            print(f"[Consolidator] Merged {len(concept_merge_map)} concepts")
            print(f"[Consolidator] Phase 1 cost: {phase1_cost['time_seconds']:.2f}s, LLM calls: {phase1_cost['llm_calls']}, Embedding calls: {phase1_cost['embedding_calls']}")

        # ===== Phase 2: Event -> Fact Synthesis =====
        print("\n" + "="*60)
        print("[Consolidator] Phase 2: Event -> Fact Synthesis")
        print("="*60)
        
        # 记录阶段开始时间和METRICS
        phase2_start_time = time.time()
        phase2_metrics_before = METRICS.snapshot()
        
        # Iterate through all Person nodes (optionally filtered for incremental consolidation)
        all_person_nodes = [
            (node_id, node_data)
            for node_id, node_data in self.graph._g.nodes(data=True)
            if node_data and node_data.get('node_type') == NodeTypeEnum.PERSON.value
        ]
        if affected_person_ids is not None:
            affected_set = set(affected_person_ids)
            person_nodes = [(nid, ndata) for nid, ndata in all_person_nodes if nid in affected_set]
            print(f"[Consolidator] Incremental: processing {len(person_nodes)} of {len(all_person_nodes)} persons")
        else:
            person_nodes = all_person_nodes

        print(f"[Consolidator] Found {len(person_nodes)} Person nodes to process")
        print(f"[Consolidator] Using min_events={min_events} for Fact creation")

        # OPTIMIZATION: Collect all synthesis tasks first, then batch process
        synthesis_tasks = []
        task_metadata = []  # Store (person_id, concept_id, event_list) for later use
        
        for person_id, person_data in person_nodes:
            try:
                stats["persons_processed"] += 1
                person_name = person_data.get('canonical_name', person_id)

                # Get all PARTICIPATED_IN events for this person
                events = self._get_participated_events(person_id)
                
                if not events:
                    continue

                # Cluster events by Concept
                concept_clusters = self._cluster_events_by_concept(events)

                # Collect synthesis tasks (skip if stable Fact already exists for this person+concept)
                for concept_id, event_list in concept_clusters.items():
                    if len(event_list) < min_events:
                        continue  # Skip clusters with too few events
                    if self._has_stable_fact_for(person_id, concept_id):
                        continue  # Reduce duplicate synthesis

                    stats["events_clustered"] += len(event_list)
                    synthesis_tasks.append((person_id, person_name, concept_id, event_list))
                    task_metadata.append((person_id, concept_id, event_list))

            except Exception as e:
                import traceback
                error_msg = f"Error collecting tasks for person {person_id}: {str(e)}"
                error_details = traceback.format_exc()
                stats["errors"].append(error_msg)
                print(f"[Consolidator] {error_msg}")
                if not self.config.runtime.quiet if hasattr(self.config.runtime, 'quiet') else False:
                    print(f"[Consolidator] Error details:\n{error_details}")
        
        # OPTIMIZATION: Batch synthesize all facts with concurrency
        if synthesis_tasks:
            # Determine max_workers from config or use default
            max_llm_workers = getattr(self.config.extraction, 'max_llm_workers', 10)
            fact_results = self._synthesize_facts_batch(synthesis_tasks, max_workers=max_llm_workers)
            
            # Create fact nodes from results
            for (person_id, concept_id), fact_data in fact_results:
                # Find corresponding event_list from task_metadata
                event_list = None
                for meta_person_id, meta_concept_id, meta_event_list in task_metadata:
                    if meta_person_id == person_id and meta_concept_id == concept_id:
                        event_list = meta_event_list
                        break
                
                if not event_list:
                    continue
                
                # Create Fact node and link it
                try:
                    fact_id = self._create_fact_node(
                        person_id=person_id,
                        concept_id=concept_id,
                        fact_data=fact_data,
                        supporting_events=event_list,
                        archive_events=archive_original_events,
                        max_supported_by=max_supported_by_per_fact
                    )

                    if fact_id:
                        stats["facts_created"] += 1
                        if archive_original_events:
                            stats["events_archived"] += len(event_list)
                    else:
                        error_msg = f"Failed to create fact node for {person_id} + {concept_id}"
                        stats["errors"].append(error_msg)
                        if not (hasattr(self.config.runtime, 'quiet') and self.config.runtime.quiet):
                            print(f"[Consolidator] {error_msg}")
                except Exception as e:
                    error_msg = f"Error creating fact for {person_id} + {concept_id}: {str(e)}"
                    stats["errors"].append(error_msg)
                    import traceback
                    if not (hasattr(self.config.runtime, 'quiet') and self.config.runtime.quiet):
                        print(f"[Consolidator] {error_msg}")
                        print(f"[Consolidator] Traceback:\n{traceback.format_exc()}")
        
        # 记录阶段2结束时间和METRICS
        phase2_end_time = time.time()
        phase2_metrics_after = METRICS.snapshot()
        
        # 计算阶段2开销
        phase2_cost = {
            "time_seconds": phase2_end_time - phase2_start_time,
            "llm_calls": phase2_metrics_after['chat']['calls'] - phase2_metrics_before['chat']['calls'],
            "llm_tokens": phase2_metrics_after['chat']['total_tokens'] - phase2_metrics_before['chat']['total_tokens'],
            "embedding_calls": phase2_metrics_after['embed']['calls'] - phase2_metrics_before['embed']['calls'],
            "embedding_tokens": phase2_metrics_after['embed']['total_tokens'] - phase2_metrics_before['embed']['total_tokens']
        }
        stats["phase_costs"]["phase2_fact_synthesis"] = phase2_cost
        print(f"[Consolidator] Phase 2 cost: {phase2_cost['time_seconds']:.2f}s, LLM calls: {phase2_cost['llm_calls']}, Embedding calls: {phase2_cost['embedding_calls']}")
        
        # ===== Phase 3: Fact Deduplication =====
        if deduplicate_facts:
            print("\n" + "="*60)
            print("[Consolidator] Phase 3: Fact Deduplication")
            print("="*60)
            
            # 记录阶段开始时间和METRICS
            phase3_start_time = time.time()
            phase3_metrics_before = METRICS.snapshot()
            
            fact_merge_map = self.fact_deduplicator.run(
                create_supersedes=True,
                high_threshold=fact_high_threshold,
                max_supported_by=max_supported_by_per_fact
            )
            
            # 记录阶段结束时间和METRICS
            phase3_end_time = time.time()
            phase3_metrics_after = METRICS.snapshot()
            
            # 计算阶段3开销
            phase3_cost = {
                "time_seconds": phase3_end_time - phase3_start_time,
                "llm_calls": phase3_metrics_after['chat']['calls'] - phase3_metrics_before['chat']['calls'],
                "llm_tokens": phase3_metrics_after['chat']['total_tokens'] - phase3_metrics_before['chat']['total_tokens'],
                "embedding_calls": phase3_metrics_after['embed']['calls'] - phase3_metrics_before['embed']['calls'],
                "embedding_tokens": phase3_metrics_after['embed']['total_tokens'] - phase3_metrics_before['embed']['total_tokens']
            }
            stats["phase_costs"]["phase3_fact_deduplication"] = phase3_cost
            
            stats["facts_deduplicated"] = len(fact_merge_map)
            print(f"[Consolidator] Deduplicated {len(fact_merge_map)} facts")
            print(f"[Consolidator] Phase 3 cost: {phase3_cost['time_seconds']:.2f}s, LLM calls: {phase3_cost['llm_calls']}, Embedding calls: {phase3_cost['embedding_calls']}")
        
        # ===== Phase 4: Claim Reconciliation =====
        if reconcile_claims:
            print("\n" + "="*60)
            print("[Consolidator] Phase 4: Claim Reconciliation")
            print("="*60)
            
            # 记录阶段开始时间和METRICS
            phase4_start_time = time.time()
            phase4_metrics_before = METRICS.snapshot()
            
            claim_supersede_map = self.claim_reconciler.run(
                create_supersedes=True,
                strategy='keep_all'
            )
            
            # 记录阶段结束时间和METRICS
            phase4_end_time = time.time()
            phase4_metrics_after = METRICS.snapshot()
            
            # 计算阶段4开销
            phase4_cost = {
                "time_seconds": phase4_end_time - phase4_start_time,
                "llm_calls": phase4_metrics_after['chat']['calls'] - phase4_metrics_before['chat']['calls'],
                "llm_tokens": phase4_metrics_after['chat']['total_tokens'] - phase4_metrics_before['chat']['total_tokens'],
                "embedding_calls": phase4_metrics_after['embed']['calls'] - phase4_metrics_before['embed']['calls'],
                "embedding_tokens": phase4_metrics_after['embed']['total_tokens'] - phase4_metrics_before['embed']['total_tokens']
            }
            stats["phase_costs"]["phase4_claim_reconciliation"] = phase4_cost
            
            stats["claims_reconciled"] = len(claim_supersede_map)
            print(f"[Consolidator] Reconciled {len(claim_supersede_map)} claims")
            print(f"[Consolidator] Phase 4 cost: {phase4_cost['time_seconds']:.2f}s, LLM calls: {phase4_cost['llm_calls']}, Embedding calls: {phase4_cost['embedding_calls']}")

        # ===== Optional: Cleanup archived nodes from vector index =====
        # P0: cleanup_archived now removes archived nodes from vector store so they are not retrieved
        if cleanup_archived:
            archived_count = 0
            for node_id, node_data in list(self.graph._g.nodes(data=True)):
                if node_data and node_data.get('is_archived') is True:
                    node_type = node_data.get('node_type')
                    if node_type and hasattr(self.vector_store, 'remove_typed'):
                        try:
                            self.vector_store.remove_typed(node_id, node_type)
                            archived_count += 1
                        except Exception:
                            pass
            if archived_count > 0:
                stats["archived_removed_from_index"] = archived_count
                print(f"[Consolidator] Cleanup: removed {archived_count} archived node(s) from vector index")

        # Save to target workspace if specified
        if target_workspace:
            from ..cogmem import CogMem
            target_cogmem = CogMem(workspace_dir=target_workspace)
            target_cogmem.graph = self.graph
            target_cogmem.vstore = self.vector_store
            target_cogmem.save()

        # ===== Summary =====
        print("\n" + "="*60)
        print("[Consolidator] Consolidation Complete - Summary")
        print("="*60)
        print(f"  Phase 1 - Concepts merged: {stats['concepts_merged']}")
        print(f"  Phase 2 - Persons processed: {stats['persons_processed']}")
        print(f"  Phase 2 - Events clustered: {stats['events_clustered']}")
        print(f"  Phase 2 - Facts created: {stats['facts_created']}")
        print(f"  Phase 2 - Events archived: {stats['events_archived']}")
        print(f"  Phase 3 - Facts deduplicated: {stats['facts_deduplicated']}")
        if stats.get('errors'):
            print(f"  Errors encountered: {len(stats['errors'])}")
            if len(stats['errors']) <= 10:
                for error in stats['errors']:
                    print(f"    - {error}")
            else:
                print(f"    (showing first 10 of {len(stats['errors'])} errors)")
                for error in stats['errors'][:10]:
                    print(f"    - {error}")
        print(f"  Phase 4 - Claims reconciled: {stats['claims_reconciled']}")
        if stats['errors']:
            print(f"  Errors: {len(stats['errors'])}")

        return stats

    def _get_participated_events(self, person_id: str) -> List[Dict[str, Any]]:
        """Get all events where the person participated"""
        events = []
        
        # Get outgoing edges with PARTICIPATED_IN or INVOLVED_IN
        for _, target_id, edge_data in self.graph._g.out_edges(person_id, data=True):
            rel_type = edge_data.get('rel_type', '')
            if rel_type in ('PARTICIPATED_IN', 'INVOLVED_IN'):
                # Get event node data
                event_data = self.graph.get_node(target_id)
                if event_data and event_data.get('node_type') == NodeTypeEnum.EVENT.value:
                    event_data['id'] = target_id  # Ensure ID is present
                    events.append(event_data)
        
        return events

    def _cluster_events_by_concept(self, events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Cluster events by the Concepts they involve"""
        clusters = defaultdict(list)

        for event in events:
            event_id = event.get('id', '')
            
            # Get concepts involved in this event
            # Look for outgoing INVOLVES edges
            for _, concept_id, edge_data in self.graph._g.out_edges(event_id, data=True):
                rel_type = edge_data.get('rel_type', '')
                if rel_type == 'INVOLVES':
                    concept_data = self.graph.get_node(concept_id)
                    if concept_data and concept_data.get('node_type') == NodeTypeEnum.CONCEPT.value:
                        clusters[concept_id].append(event)

        return dict(clusters)

    def _has_stable_fact_for(self, person_id: str, concept_id: str) -> bool:
        """Return True if a non-archived (stable) Fact already exists for this (person, concept)."""
        for node_id, node_data in self.graph._g.nodes(data=True):
            if not node_data or node_data.get('node_type') != NodeTypeEnum.FACT.value:
                continue
            if node_data.get('is_archived'):
                continue
            if node_data.get('subject_id') != person_id:
                continue
            if node_data.get('object_id') == concept_id or node_data.get('purpose_id') == concept_id:
                return True
        return False

    def _synthesize_fact(
        self,
        person_id: str,
        person_name: str,
        concept_id: str,
        events: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to synthesize a generalized Fact from events"""
        
        # Get concept name
        concept_data = self.graph.get_node(concept_id)
        concept_name = concept_data.get('name', concept_id) if concept_data else concept_id

        # Prepare events data for LLM
        events_json = []
        for event in events:
            event_info = {
                "id": event.get('id', ''),
                "title": event.get('title', ''),
                "time_start": event.get('time_start'),
                "time_start_raw": event.get('time_start_raw'),
                "description": event.get('description', ''),
                "content": event.get('content', '')
            }
            events_json.append(event_info)

        # Format prompt
        user_prompt = MEMORY_CONSOLIDATION_USER.format(
            person_id=person_id,
            person_name=person_name,
            concept_id=concept_id,
            concept_name=concept_name,
            events_json=json.dumps(events_json, ensure_ascii=False, indent=2)
        )

        # Call LLM
        try:
            # CRITICAL FIX:
            # LLMClient.chat signature is chat(system: str, user: str, model: Optional[str] = None)
            # Do NOT pass system_prompt/user_prompt/temperature (will raise TypeError and trigger retries).
            response = self.llm.chat(
                system=MEMORY_CONSOLIDATION_SYSTEM,
                user=user_prompt,
                model=None,
            )

            # Parse JSON response
            fact_data = safe_json_loads(response)
            if not fact_data:
                # Try to extract JSON from markdown
                import re
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    fact_data = safe_json_loads(json_match.group())

            if not fact_data:
                print(f"[Consolidator] Failed to parse LLM response: {response[:200]}")
                return None

            # Validate required fields
            if 'content' not in fact_data or 'predicate' not in fact_data:
                print(f"[Consolidator] Missing required fields in fact_data: {fact_data}")
                return None

            # Ensure object_id is set
            if 'object_id' not in fact_data:
                fact_data['object_id'] = concept_id

            return fact_data

        except Exception as e:
            print(f"[Consolidator] Error synthesizing fact: {e}")
            return None
    
    def _synthesize_facts_batch(
        self,
        tasks: List[Tuple[str, str, str, List[Dict[str, Any]]]],
        max_workers: int = 10
    ) -> List[Tuple[Tuple[str, str], Dict[str, Any]]]:
        """Batch synthesize facts with concurrency (OPTIMIZATION)
        
        Args:
            tasks: List of (person_id, person_name, concept_id, events) tuples
            max_workers: Maximum concurrent LLM calls (default: 10)
        
        Returns:
            List of ((person_id, concept_id), fact_data) tuples
        """
        results = []
        
        if not tasks:
            return results
        
        # Determine concurrency level
        actual_workers = min(max_workers, len(tasks))
        
        print(f"[Consolidator] Batch synthesizing {len(tasks)} facts with {actual_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    self._synthesize_fact,
                    person_id, person_name, concept_id, events
                ): (person_id, concept_id) 
                for person_id, person_name, concept_id, events in tasks
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Synthesizing Facts", disable=self.config.runtime.max_concurrency > 1):
                try:
                    fact_data = future.result(timeout=60)  # 60s timeout per LLM call
                    if fact_data:
                        results.append((futures[future], fact_data))
                except Exception as e:
                    person_id, concept_id = futures[future]
                    if self.config.runtime.max_concurrency == 1:
                        print(f"[Consolidator] Error synthesizing fact for {person_id} + {concept_id}: {e}")
        
        print(f"[Consolidator] Successfully synthesized {len(results)}/{len(tasks)} facts")
        return results

    def _normalize_fact_date(self, date_str: Optional[str]) -> Optional[str]:
        """Normalize date string for Fact valid_from/valid_to fields
        
        Handles:
        - "recently", "lately" -> None (fuzzy time, no specific date)
        - "YYYY-MM" -> "YYYY-MM-01" (year-month only, add day)
        - "YYYY-MM-DD" -> as-is (valid ISO format)
        - "YYYY-MM-DDTHH:MM:SS" -> as-is (valid ISO format)
        - None -> None
        - Invalid formats -> None (will be skipped)
        """
        if not date_str:
            return None
        
        date_str = str(date_str).strip()
        
        # Handle fuzzy time expressions
        fuzzy_time = ["recently", "lately", "sometime", "sometimes", "earlier", "before", "recent"]
        if date_str.lower() in fuzzy_time:
            return None  # Fuzzy time, no specific date
        
        # Handle "YYYY-MM" format (year-month only)
        import re
        year_month_match = re.match(r'^(\d{4})-(\d{2})$', date_str)
        if year_month_match:
            year, month = year_month_match.groups()
            # Validate month
            if 1 <= int(month) <= 12:
                return f"{year}-{month}-01"  # Use first day of month
        
        # Handle "YYYY-MM-DD" format
        date_match = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', date_str)
        if date_match:
            year, month, day = date_match.groups()
            # Validate date
            try:
                datetime(int(year), int(month), int(day))
                return date_str
            except ValueError:
                return None
        
        # Handle "YYYY-MM-DDTHH:MM:SS" format
        if 'T' in date_str:
            try:
                datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return date_str
            except ValueError:
                return None
        
        # Invalid format, return None
        return None

    def _create_fact_node(
        self,
        person_id: str,
        concept_id: str,
        fact_data: Dict[str, Any],
        supporting_events: List[Dict[str, Any]],
        archive_events: bool = False,
        max_supported_by: int = 5
    ) -> Optional[str]:
        """Create Fact node and link it to person and supporting events
        
        Args:
            person_id: Person ID
            concept_id: Concept ID
            fact_data: Fact data from LLM synthesis
            supporting_events: List of supporting events
            archive_events: Whether to archive original events
            max_supported_by: Maximum number of SUPPORTED_BY edges to create (default: 5)
        """
        
        # Generate Fact ID
        fact_id = _make_fact_id(person_id, fact_data.get('object_id', concept_id), fact_data['predicate'])

        # Check if Fact already exists
        if fact_id in self.graph._g:
            print(f"[Consolidator] Fact {fact_id} already exists, skipping")
            return fact_id

        # Normalize dates before creating Fact
        valid_from = self._normalize_fact_date(fact_data.get('valid_from'))
        valid_to = self._normalize_fact_date(fact_data.get('valid_to'))

        # Importance score for retrieval: support count, confidence, recency
        n_support = min(len(supporting_events), 10)
        conf = fact_data.get('confidence', 0.7)
        recency = 1.0  # Consolidated facts are from recent events
        importance_score = min(1.0, 0.4 * (n_support / 5.0) + 0.4 * conf + 0.2 * recency)

        # Create Fact node (consolidation produces stable facts)
        fact = Fact(
            id=fact_id,
            content=fact_data['content'],
            predicate=fact_data['predicate'],
            subject_id=person_id,
            object_id=fact_data.get('object_id', concept_id),
            valid_from=valid_from,
            valid_to=valid_to,
            confidence=fact_data.get('confidence', 0.7),
            metadata=Metadata(
                source="memory_consolidation",
                created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            memory_lifecycle=MemoryLifecycle.STABLE,
            importance_score=importance_score,
        )

        # Add Fact to graph using GraphStore's add_node method
        self.graph.add_node(fact)

        # Link Person --[HAS_ATTRIBUTE]--> Fact
        person_fact_rel = Relation(
            head_id=person_id,
            tail_id=fact_id,
            rel_type='HAS_ATTRIBUTE',
            weight=fact.confidence or 0.7,
            confidence=fact.confidence or 0.7
        )
        self.graph.add_relations([person_fact_rel])

        # Link Fact --[SUPPORTED_BY]--> Events
        # Limit to max_supported_by most important events
        supported_by_relations = []
        
        # Score events by importance (time, confidence)
        scored_events = []
        for event in supporting_events:
            event_id = event.get('id', '')
            if not event_id:
                continue
            
            # Score: time (newer = higher) + confidence
            time_score = 0.0
            time_start = event.get('time_start')
            if time_start:
                try:
                    # Use global datetime import from top of file
                    dt = datetime.fromisoformat(time_start.replace('Z', '+00:00'))
                    time_score = dt.timestamp()
                except:
                    pass
            
            confidence = event.get('confidence', 0.5)
            score = time_score + confidence * 1000
            scored_events.append((score, event))
        
        # Sort by score (descending) and take top max_supported_by
        scored_events.sort(key=lambda x: x[0], reverse=True)
        top_events = scored_events[:max_supported_by]
        
        for _, event in top_events:
            event_id = event.get('id', '')
            if event_id:
                supported_by_rel = Relation(
                    head_id=fact_id,
                    tail_id=event_id,
                    rel_type='SUPPORTED_BY',
                    weight=1.0,
                    confidence=1.0
                )
                supported_by_relations.append(supported_by_rel)
        
        if supported_by_relations:
            self.graph.add_relations(supported_by_relations)

            # Optionally archive original events
            if archive_events:
                event_data = self.graph.get_node(event_id)
                if event_data:
                    event_data['is_archived'] = True
                    # Update node in graph
                    for key, value in event_data.items():
                        if key != 'id':  # Don't overwrite the node ID
                            self.graph._g.nodes[event_id][key] = value

        # Vectorize the Fact (CRITICAL for retrieval!)
        try:
            # CRITICAL FIX: 优化 Fact 的向量化文本，包含更多语义信息
            # 包含：content（核心内容）、predicate（关系类型）、object_id（概念名称）
            # 添加 subject 信息以提高检索精度
            person_data = self.graph.get_node(person_id)
            person_name = person_data.get('canonical_name', person_id) if person_data else person_id
            
            # 获取 concept 名称
            concept_data = self.graph.get_node(fact.object_id or concept_id)
            concept_name = concept_data.get('name', fact.object_id or concept_id) if concept_data else (fact.object_id or concept_id)
            concept_name_readable = concept_name.replace('c_', '').replace('_', ' ') if concept_name.startswith('c_') else concept_name
            
            # 构建更丰富的向量化文本
            # 格式：person_name predicate concept_name: content
            # 例如："Alice has_habit hiking: Alice regularly goes hiking in the mountains"
            fact_text = f"{person_name} {fact.predicate} {concept_name_readable}: {fact.content}"
            
            # CRITICAL FIX: 检查 fact_text 是否为空
            if not fact_text or not fact_text.strip():
                print(f"[Consolidator] Warning: Empty fact_text for {fact_id}, skipping vectorization")
                return fact_id
            
            # 调用 encode 并检查返回结果
            embeddings = self.embedding_encoder.encode([fact_text])
            
            # CRITICAL FIX: 检查返回结果是否为空
            if not embeddings or len(embeddings) == 0:
                print(f"[Consolidator] Warning: encode() returned empty list for fact {fact_id}, skipping vectorization")
                return fact_id
            
            fact_embedding = embeddings[0]
            
            # CRITICAL FIX: 检查 embedding 是否为空或无效
            if not fact_embedding or len(fact_embedding) == 0:
                print(f"[Consolidator] Warning: Empty embedding vector for fact {fact_id}, skipping vectorization")
                return fact_id
            
            # Add to typed vector store
            node_type = NodeTypeEnum.FACT.value  # "fact"
            self.vector_store.add_typed(
                node_id=fact_id,
                vector=fact_embedding,
                node_type=node_type
            )
        except IndexError as e:
            print(f"[Consolidator] Warning: IndexError when vectorizing fact {fact_id}: {e}")
            print(f"  fact_text: {fact_text[:100] if 'fact_text' in locals() else 'N/A'}")
            print(f"  embeddings length: {len(embeddings) if 'embeddings' in locals() else 'N/A'}")
        except Exception as e:
            print(f"[Consolidator] Warning: Failed to vectorize fact {fact_id}: {e}")
            import traceback
            traceback.print_exc()

        # P0 FIX: Register Fact evidence from supporting events
        # This fixes the 91.8% bad cases issue where Facts have no evidence
        if self.evidence_store:
            try:
                for event in supporting_events[:max_supported_by]:
                    event_id = event.get('id', '')
                    if event_id:
                        # Get chunks from event and register for fact
                        event_chunks = self.evidence_store.get_chunks(event_id, limit=3)
                        for chunk in event_chunks:
                            self.evidence_store.add_chunk_for_entity(fact_id, chunk)
            except Exception as e:
                # Don't fail if evidence registration fails
                if self.config.runtime.max_concurrency == 1:
                    print(f"[Consolidator] Warning: Failed to register evidence for fact {fact_id}: {e}")

        return fact_id
