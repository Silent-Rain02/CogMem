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
from cogmem.models.execution_errors import raise_if_execution_failure

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

def _make_fact_id(person_id: str, concept_id: str, predicate: str, source_event_ids=()) -> str:
    """Generate a unique Fact ID based on person, concept, and predicate"""
    content = json.dumps([person_id, concept_id, predicate, sorted(set(source_event_ids))], ensure_ascii=False)
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
        evidence_store=None,
        chunk_store=None
    ):
        self.config = config
        self.graph = graph
        self.vector_store = vector_store
        self.embedding_encoder = embedding_encoder
        self.evidence_store = evidence_store  # Source references must follow the same workspace.
        self.chunk_store = chunk_store
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
        from datetime import timezone
        started_at = datetime.now(timezone.utc).isoformat()
        if affected_person_ids is not None:
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
        if not stats.get('errors') and not stats.get('synthesis_failed'):
            # Checkpoint the start, so writes arriving during consolidation are
            # reconsidered next time; a failed pass never advances the cursor.
            self._last_consolidation_timestamp = started_at
        
        return stats

    @staticmethod
    def _change_timestamp(value):
        """Normalize ISO instants; legacy naive wall times use host local zone."""
        from datetime import timezone
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return parsed.astimezone(timezone.utc)

    def _find_affected_persons(self, since_timestamp: str) -> List[str]:
        """Find changed event participants; unknown change times cannot be skipped."""
        since_dt = self._change_timestamp(since_timestamp)
        affected = set()
        for event_id, node in self.graph._g.nodes(data=True):
            if not node or node.get('node_type') != NodeTypeEnum.EVENT.value:
                continue
            metadata = node.get('metadata') or {}
            times = []
            unknown = False
            for field in ('created_at', 'updated_at'):
                value = metadata.get(field) if isinstance(metadata, dict) else None
                if value:
                    try:
                        times.append(self._change_timestamp(value))
                    except (ValueError, TypeError, AttributeError):
                        unknown = True
            if not unknown and times and max(times) < since_dt:
                continue
            affected.update(node.get('participants') or [])
            for source, _, data in self.graph._g.in_edges(event_id, data=True):
                if (data.get('rel_type') or data.get('relation_type')) == 'PARTICIPATED_IN':
                    affected.add(source)
        return sorted(affected)

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
        cleanup_archived: bool = True,
        max_time_window_days: int = 30
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
            max_supported_by_per_fact: Deprecated write cap; all provenance links are retained, read limits apply later
            cleanup_archived: Whether to delete archived nodes after consolidation (default: True)
        
        Returns:
            Statistics dictionary with consolidation results and phase costs
        """
        import time
        from ..utils.metrics import METRICS
        if type(max_time_window_days) is not int or max_time_window_days < 0:
            raise ValueError('Consolidation window must be a nonnegative number of calendar days')
        if type(min_events) is not int or min_events < 1:
            raise ValueError('Consolidation requires a positive count of distinct source events')
        
        stats = {
            "persons_processed": 0,
            "facts_created": 0,
            "events_clustered": 0,
            "events_archived": 0,
            "concepts_merged": 0,
            "facts_deduplicated": 0,
            "claims_reconciled": 0,
            "errors": [],
            "time_window_days": max_time_window_days,
            "event_concept_pairs_without_bounded_time": 0,
            "phase_costs": {}  # 记录各阶段的详细开销
        }

        # Validate export targets before synthesis or any graph mutation.
        from pathlib import Path
        if target_workspace:
            target = Path(target_workspace)
            if target.exists() and (not target.is_dir() or any(target.iterdir())):
                raise FileExistsError('Consolidation export requires a new or empty target workspace')
            if not source_workspace and (self.evidence_store is None or getattr(self, 'chunk_store', None) is None):
                raise ValueError('Export requires complete original chunks and evidence references')
        if source_workspace:
            from ..storage.workspace import read_workspace
            source = Path(source_workspace)
            if not (source / self.config.graph_file).is_file():
                raise FileNotFoundError('Consolidation source workspace is missing its graph')
            view = read_workspace(self.config, source)
            self.graph, self.vector_store = view.graph, view.vectors
            self.evidence_store, self.chunk_store = view.evidence, view.chunks
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
                    stats['event_concept_pairs_without_bounded_time'] += sum(
                        self._event_bounds(e) is None or
                        (self._event_bounds(e)[1] - self._event_bounds(e)[0]).days > max_time_window_days
                        for e in event_list)
                    for window in self._bounded_event_windows(event_list, max_time_window_days):
                        if len(window) < min_events or self._has_synthesis_for(person_id, concept_id, window):
                            continue
                        stats['events_clustered'] += len(window)
                        synthesis_tasks.append((person_id, person_name, concept_id, window))

            except Exception as e:
                raise_if_execution_failure(e)
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
            stats["synthesis_attempted"] = len(synthesis_tasks)
            stats["synthesis_succeeded"] = len(fact_results)
            stats["synthesis_failed"] = len(synthesis_tasks) - len(fact_results)
            if stats["synthesis_failed"]:
                stats["errors"].append(f"{stats['synthesis_failed']} fact synthesis tasks failed")
            
            # Create fact nodes from results
            for (person_id, person_name, concept_id, event_list), fact_data in fact_results:
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
                    raise_if_execution_failure(e)
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
                        except Exception as execution_error:
                            raise_if_execution_failure(execution_error)
                            pass
            if archived_count > 0:
                stats["archived_removed_from_index"] = archived_count
                print(f"[Consolidator] Cleanup: removed {archived_count} archived node(s) from vector index")

        if target_workspace:
            from ..storage.workspace import WorkspaceView, export_workspace
            export_workspace(self.config, target_workspace,
                WorkspaceView(self.graph, self.vector_store, self.chunk_store, self.evidence_store))

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
        seen = set()
        
        # Get outgoing edges with PARTICIPATED_IN or INVOLVED_IN
        for _, target_id, edge_data in self.graph._g.out_edges(person_id, data=True):
            rel_type = edge_data.get('rel_type', '')
            if rel_type in ('PARTICIPATED_IN', 'INVOLVED_IN'):
                # Get event node data
                event_data = self.graph.get_node(target_id)
                if (event_data and event_data.get('node_type') == NodeTypeEnum.EVENT.value
                        and not event_data.get('is_archived') and target_id not in seen):
                    seen.add(target_id)
                    event_data['id'] = target_id  # Ensure ID is present
                    events.append(event_data)
        
        return events

    def _cluster_events_by_concept(self, events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Cluster events by the Concepts they involve"""
        clusters = defaultdict(list)
        seen = set()

        for event in events:
            event_id = event.get('id', '')
            
            # Get concepts involved in this event
            # Look for outgoing INVOLVES edges
            for _, concept_id, edge_data in self.graph._g.out_edges(event_id, data=True):
                rel_type = edge_data.get('rel_type', '')
                if rel_type == 'INVOLVES':
                    concept_data = self.graph.get_node(concept_id)
                    if (concept_data and concept_data.get('node_type') == NodeTypeEnum.CONCEPT.value
                            and (concept_id, event_id) not in seen):
                        seen.add((concept_id, event_id))
                        clusters[concept_id].append(event)

        return dict(clusters)

    @staticmethod
    def _event_bounds(event):
        from ..utils.temporal import date_bounds
        start = date_bounds(event.get('time_start')) or date_bounds(event.get('time_resolved'))
        end = date_bounds(event.get('time_end')) if event.get('time_end') else start
        if not start or not end or start[0] > end[1]:
            return None
        return start[0], end[1]

    @classmethod
    def _bounded_event_windows(cls, events, max_days):
        """Maximal source sets whose entire known occurrence spans fit a window.

        Unknown/coarse intervals are retained as original events, not forced
        onto a recording date. Maximal windows avoid redundant subset summaries;
        distinct overlapping windows retain their exact independent support.
        """
        from datetime import timedelta
        known = {}
        for event in events:
            bounds = cls._event_bounds(event)
            if event.get('id') and bounds and (bounds[1] - bounds[0]).days <= max_days:
                known[event['id']] = (bounds, event)
        ordered = sorted(known.values(), key=lambda row: (row[0][0], row[0][1], row[1]['id']))
        candidates = {}
        for bounds, _ in ordered:
            from datetime import date
            try:
                limit = bounds[0] + timedelta(days=max_days)
            except OverflowError:
                limit = date.max
            rows = [event for interval, event in ordered if bounds[0] <= interval[0] and interval[1] <= limit]
            candidates[frozenset(event['id'] for event in rows)] = rows
        return [rows for keys, rows in candidates.items()
                if keys and not any(keys < other for other in candidates)]

    def _has_synthesis_for(self, person_id, concept_id, events):
        """Skip only a derived fact built from this exact source set."""
        expected = {event['id'] for event in events if event.get('id')}
        if not expected:
            return False
        for node_id, data in self.graph._g.nodes(data=True):
            if (data.get('node_type') != 'fact' or data.get('subject_id') != person_id
                    or data.get('object_id') != concept_id
                    or (data.get('metadata') or {}).get('source') != 'memory_consolidation'):
                continue
            actual = {target for _, target, edge in self.graph._g.out_edges(node_id, data=True)
                      if edge.get('rel_type') == 'SUPPORTED_BY'}
            if actual == expected:
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

            if not isinstance(fact_data, dict) or not fact_data:
                print(f"[Consolidator] Failed to parse LLM response: {response[:200]}")
                return None

            # The task fixes source/subject/concept identities. The model may
            # describe a proposition but cannot invent a different graph address.
            for field in ['content', 'predicate']:
                if not isinstance(fact_data.get(field), str) or not fact_data[field].strip():
                    return None
            if fact_data.get('object_id') not in (None, concept_id):
                return None
            confidence = fact_data.get('confidence', 0.7)
            import math
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                return None
            fact_data = dict(fact_data)
            fact_data['object_id'] = concept_id
            return fact_data

        except Exception as e:
            raise_if_execution_failure(e)
            print(f"[Consolidator] Error synthesizing fact: {e}")
            return None
    
    def _synthesize_facts_batch(
        self,
        tasks: List[Tuple[str, str, str, List[Dict[str, Any]]]],
        max_workers: int = 10
    ) -> List[Tuple[Tuple[str, str, str, List[Dict[str, Any]]], Dict[str, Any]]]:
        """Batch synthesize facts with concurrency (OPTIMIZATION)
        
        Args:
            tasks: List of (person_id, person_name, concept_id, events) tuples
            max_workers: Maximum concurrent LLM calls (default: 10)
        
        Returns:
            List of (original_task_including_exact_support_events, fact_data) tuples
        """
        results = []
        
        if not tasks:
            return results
        
        # Determine concurrency level
        actual_workers = min(max_workers, len(tasks))
        
        print(f"[Consolidator] Batch synthesizing {len(tasks)} facts with {actual_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(self._synthesize_fact, *task): task for task in tasks}
            
            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Synthesizing Facts", disable=self.config.runtime.max_concurrency > 1):
                try:
                    fact_data = future.result(timeout=60)  # 60s timeout per LLM call
                    if fact_data:
                        results.append((futures[future], fact_data))
                except Exception as e:
                    raise_if_execution_failure(e)
                    person_id, _, concept_id, _ = futures[future]
                    print(f"[Consolidator] Error synthesizing fact for {person_id} + {concept_id}: {e}")
        
        print(f"[Consolidator] Successfully synthesized {len(results)}/{len(tasks)} facts")
        return results

    def _normalize_fact_date(self, date_str: Optional[str]) -> Optional[str]:
        """Preserve absolute precision without fabricating validity boundaries."""
        from ..utils.temporal import normalize_event_time
        normalized = normalize_event_time(date_str)
        if normalized is not None:
            return normalized
        if isinstance(date_str, str) and 'T' in date_str:
            try:
                datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return date_str
            except ValueError:
                pass
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
        
        # Validate identities and complete support before any model call/write.
        import math
        from ..storage.vector_store import validate_vectors
        person = self.graph.get_node(person_id)
        concept = self.graph.get_node(concept_id)
        if not person or person.get('node_type') != 'person' or not concept or concept.get('node_type') != 'concept':
            raise ValueError('Consolidation subject and concept must resolve to declared graph nodes')
        if fact_data.get('object_id') not in (None, concept_id):
            raise ValueError('Consolidation cannot redirect the task concept address')
        if any(not isinstance(fact_data.get(field), str) or not fact_data[field].strip() for field in ['content', 'predicate']):
            raise ValueError('Consolidation requires nonempty proposition and predicate')
        confidence = fact_data.get('confidence', 0.7)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Consolidation confidence must be finite and within [0,1]')
        event_ids = list(dict.fromkeys(e.get('id') for e in supporting_events))
        if not event_ids or any(not key or not self.graph.get_node(key) or self.graph.get_node(key).get('node_type') != 'event' for key in event_ids):
            raise ValueError('Every consolidation support must be an existing event occurrence')
        fact_id = _make_fact_id(person_id, concept_id, fact_data['predicate'], event_ids)
        existing = self.graph.get_node(fact_id)
        if existing:
            support = {tail for _, tail, edge in self.graph._g.out_edges(fact_id, data=True) if edge.get('rel_type') == 'SUPPORTED_BY'}
            if (existing.get('node_type') != 'fact' or existing.get('subject_id') != person_id
                    or existing.get('object_id') != concept_id or support != set(event_ids)):
                raise ValueError('Consolidated address collision with another projection')
            return fact_id
        fact = Fact(id=fact_id, content=fact_data['content'], predicate=fact_data['predicate'],
            subject_id=person_id, object_id=concept_id,
            valid_from=self._normalize_fact_date(fact_data.get('valid_from')),
            valid_to=self._normalize_fact_date(fact_data.get('valid_to')), confidence=confidence,
            metadata=Metadata(source='memory_consolidation', created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            memory_lifecycle=MemoryLifecycle.STABLE,
            importance_score=min(1., .4*(min(len(event_ids),10)/5.)+.4*confidence+.2))
        person_name = person.get('canonical_name', person_id)
        concept_name = concept.get('name', concept_id)
        readable = concept_name.replace('c_', '').replace('_', ' ') if concept_name.startswith('c_') else concept_name
        text = f'{person_name} {fact.predicate} {readable}: {fact.content}'
        sources = []
        if self.evidence_store is not None:
            for key in event_ids:
                chunks = self.evidence_store.get_chunks(key, limit=None)
                if not chunks:
                    raise ValueError('A supporting event has no recoverable original source')
                sources.extend(chunks)
        embeddings = self.embedding_encoder.encode([text])
        if len(embeddings) != 1:
            raise ValueError('Consolidation embedding must return exactly one vector')
        # The store validates finite values/dimension before graph mutation.
        if hasattr(self.vector_store, '_dim'):
            validate_vectors([fact_id], embeddings, self.vector_store._dim)
        elif not embeddings[0] or not all(math.isfinite(x) for x in embeddings[0]):
            raise ValueError('Invalid consolidation embedding')
        if hasattr(self.vector_store, 'add_typed'):
            self.vector_store.add_typed(node_id=fact_id, vector=embeddings[0], node_type=NodeTypeEnum.FACT.value)
        else:
            self.vector_store.add([fact_id], embeddings[0])
        self.graph.add_node(fact)
        self.graph.add_relations([Relation(head_id=person_id,tail_id=fact_id,rel_type='HAS_ATTRIBUTE',weight=confidence,confidence=confidence)] +
            [Relation(head_id=fact_id,tail_id=key,rel_type='SUPPORTED_BY') for key in event_ids])
        if self.evidence_store is not None:
            for source in dict.fromkeys(sources):
                self.evidence_store.add_chunk_for_entity(fact_id, source)
        if archive_events:
            for key in event_ids:self.graph._g.nodes[key]['is_archived'] = True
        return fact_id
