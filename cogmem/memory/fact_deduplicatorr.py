"""Fact Deduplicator Module

This module implements Fact deduplication and version management.
It detects semantically equivalent Facts and manages temporal validity
using SUPERSEDES relationships.
"""

from __future__ import annotations

import numpy as np
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

from ..storage.graph_store import GraphStore
from ..storage.typed_vector_store import TypedVectorStore
from ..storage.schema import Fact, Relation, NodeTypeEnum, Metadata
from ..models.embedding import EmbeddingEncoder

def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(v1, v2) / (norm1 * norm2))

class FactDeduplicator:
    """Fact Deduplicator: Detects and manages duplicate Facts
    
    This class implements Fact deduplication with version management:
    1. Detect duplicate Facts based on (subject, predicate, object) triple
    2. Detect semantic duplicates using embedding similarity
    3. Manage temporal validity (valid_from, valid_to)
    4. Create SUPERSEDES edges for version history
    
    Usage:
        deduplicator = FactDeduplicator(graph_store, embedding_encoder)
        
        # Run full deduplication
        deduplicator.run()
        
        # Or deduplicate a specific list of facts
        unique_facts = deduplicator.deduplicate(facts)
    """
    
    def __init__(
        self,
        graph: GraphStore,
        vector_store: Optional[TypedVectorStore] = None,
        embedding_encoder: Optional[EmbeddingEncoder] = None,
        semantic_threshold: float = 0.90
    ):
        """Initialize FactDeduplicator
        
        Args:
            graph: Graph store instance
            vector_store: Vector store instance (optional)
            embedding_encoder: Embedding encoder instance (optional)
            semantic_threshold: Threshold for semantic similarity (default: 0.90)
        """
        self.graph = graph
        self.vector_store = vector_store
        self.embedding_encoder = embedding_encoder
        self.semantic_threshold = semantic_threshold
        
        # Cache for fact embeddings
        self._fact_embeddings: Dict[str, List[float]] = {}
        
        # Statistics
        self._stats = {
            "facts_processed": 0,
            "duplicates_found": 0,
            "facts_merged": 0,
            "supersedes_edges_created": 0,
            "facts_updated": 0
        }
    
    def _get_all_facts(self) -> List[Tuple[str, Dict]]:
        """Get all Fact nodes from the graph"""
        facts = []
        for node_id, node_data in self.graph._g.nodes(data=True):
            if node_data and node_data.get('node_type') == NodeTypeEnum.FACT.value:
                facts.append((node_id, node_data))
        return facts
    
    def _get_fact_embedding(self, fact_id: str, fact_data: Dict) -> Optional[List[float]]:
        """Get or generate embedding for a fact"""
        # Check cache first
        if fact_id in self._fact_embeddings:
            return self._fact_embeddings[fact_id]
        
        # Try to get from vector store
        if self.vector_store:
            try:
                vec = self.vector_store.get_vector(fact_id)
                if vec is not None:
                    self._fact_embeddings[fact_id] = vec
                    return vec
            except:
                pass
        
        # Generate new embedding
        if self.embedding_encoder:
            content = fact_data.get('content', '')
            predicate = fact_data.get('predicate', '')
            object_id = fact_data.get('object_id', '')
            
            text = f"{content} {predicate} {object_id}".strip()
            
            if text:
                try:
                    embeddings = self.embedding_encoder.encode([text])
                    if embeddings and embeddings[0]:
                        self._fact_embeddings[fact_id] = embeddings[0]
                        return embeddings[0]
                except Exception as e:
                    print(f"[FactDeduplicator] Failed to generate embedding for {fact_id}: {e}")
        
        return None
    
    def _get_triple_key(self, fact_data: Dict) -> str:
        """Generate a key from (subject, predicate, object) triple"""
        subject = fact_data.get('subject_id', '')
        predicate = fact_data.get('predicate', '')
        obj = fact_data.get('object_id', '') or fact_data.get('purpose_id', '')
        
        return f"{subject}|{predicate}|{obj}"
    
    def _are_semantically_similar(
        self,
        fact1: Tuple[str, Dict],
        fact2: Tuple[str, Dict]
    ) -> bool:
        """Check if two facts are semantically similar"""
        id1, data1 = fact1
        id2, data2 = fact2
        
        if self.embedding_encoder:
            emb1 = self._get_fact_embedding(id1, data1)
            emb2 = self._get_fact_embedding(id2, data2)
            
            if emb1 is not None and emb2 is not None:
                similarity = _cosine_similarity(emb1, emb2)
                return similarity >= self.semantic_threshold
        
        # Fallback: compare content strings
        content1 = data1.get('content', '').lower().strip()
        content2 = data2.get('content', '').lower().strip()
        
        return content1 == content2
    
    def _parse_timestamp(self, ts: Optional[str]) -> Optional[datetime]:
        """Parse ISO timestamp string to datetime"""
        if not ts:
            return None
        
        try:
            # Handle various formats
            if 'T' in ts:
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            else:
                return datetime.strptime(ts, "%Y-%m-%d")
        except:
            return None
    
    def _select_canonical_fact(self, facts: List[Tuple[str, Dict]]) -> Tuple[str, Dict]:
        """Select the canonical fact from a group of duplicates
        
        Selection criteria:
        1. Most recent valid_from (newer information)
        2. Higher confidence
        3. More supporting evidence (SUPPORTED_BY edges)
        """
        def score(fact: Tuple[str, Dict]) -> Tuple:
            fact_id, data = fact
            
            # Parse valid_from
            valid_from = self._parse_timestamp(data.get('valid_from'))
            valid_from_ts = valid_from.timestamp() if valid_from else 0
            
            # Get confidence
            confidence = data.get('confidence', 0.5)
            
            # Count supporting evidence
            support_count = 0
            if fact_id in self.graph._g:
                for _, _, edge_data in self.graph._g.out_edges(fact_id, data=True):
                    if edge_data.get('rel_type') == 'SUPPORTED_BY':
                        support_count += 1
            
            return (valid_from_ts, confidence, support_count)
        
        return max(facts, key=score)
    
    def find_duplicates(self) -> Tuple[Dict[str, List[Tuple[str, Dict]]], Dict[Tuple[str, str], float]]:
        """Find groups of duplicate Facts with similarity scores
        
        Returns:
            Tuple of:
            - Dictionary mapping triple keys to lists of duplicate facts
            - Dictionary mapping (fact_id1, fact_id2) -> similarity score
        """
        facts = self._get_all_facts()
        self._stats["facts_processed"] = len(facts)
        
        if len(facts) < 2:
            return {}, {}
        
        print(f"[FactDeduplicator] Processing {len(facts)} facts...")
        
        # Group by (subject, predicate, object) triple
        triple_groups: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
        
        for fact_id, fact_data in facts:
            key = self._get_triple_key(fact_data)
            triple_groups[key].append((fact_id, fact_data))
        
        # Filter to only groups with duplicates
        duplicate_groups = {
            key: group for key, group in triple_groups.items()
            if len(group) > 1
        }
        
        # Also check for semantic duplicates within non-duplicate groups
        # (Facts with different triples but same meaning)
        if self.embedding_encoder:
            remaining_facts = [
                (fact_id, fact_data)
                for key, group in triple_groups.items()
                if len(group) == 1
                for fact_id, fact_data in group
            ]
            
            # Check semantic similarity for remaining facts
            semantic_groups = self._find_semantic_duplicates(remaining_facts)
            
            for group in semantic_groups:
                if len(group) > 1:
                    # Create a synthetic key for this semantic group
                    key = f"semantic_{group[0][0]}"
                    duplicate_groups[key] = group
        
        # Calculate similarity scores for duplicate groups
        similarity_scores: Dict[Tuple[str, str], float] = {}
        
        for key, group in duplicate_groups.items():
            n = len(group)
            for i in range(n):
                for j in range(i + 1, n):
                    fact1 = group[i]
                    fact2 = group[j]
                    
                    # Calculate semantic similarity
                    if self.embedding_encoder:
                        emb1 = self._get_fact_embedding(fact1[0], fact1[1])
                        emb2 = self._get_fact_embedding(fact2[0], fact2[1])
                        
                        if emb1 is not None and emb2 is not None:
                            similarity = _cosine_similarity(emb1, emb2)
                            # Use sorted tuple as key for consistency
                            sim_key = tuple(sorted([fact1[0], fact2[0]]))
                            similarity_scores[sim_key] = similarity
                    else:
                        # Fallback: use content similarity
                        content1 = fact1[1].get('content', '').lower().strip()
                        content2 = fact2[1].get('content', '').lower().strip()
                        similarity = 1.0 if content1 == content2 else 0.0
                        sim_key = tuple(sorted([fact1[0], fact2[0]]))
                        similarity_scores[sim_key] = similarity
        
        self._stats["duplicates_found"] = sum(len(g) - 1 for g in duplicate_groups.values())
        print(f"[FactDeduplicator] Found {len(duplicate_groups)} groups with duplicates")
        
        return duplicate_groups, similarity_scores
    
    def _find_semantic_duplicates(
        self,
        facts: List[Tuple[str, Dict]]
    ) -> List[List[Tuple[str, Dict]]]:
        """Find semantically similar facts using embeddings"""
        if len(facts) < 2:
            return []
        
        # Build adjacency for connected components
        adjacency: Dict[int, Set[int]] = defaultdict(set)
        
        n = len(facts)
        for i in range(n):
            for j in range(i + 1, n):
                if self._are_semantically_similar(facts[i], facts[j]):
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        
        # Find connected components
        visited: Set[int] = set()
        groups: List[List[Tuple[str, Dict]]] = []
        
        for i in range(n):
            if i in visited or i not in adjacency:
                continue
            
            # BFS
            group_indices: Set[int] = set()
            queue = [i]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                
                visited.add(current)
                group_indices.add(current)
                
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            if len(group_indices) > 1:
                groups.append([facts[idx] for idx in group_indices])
        
        return groups
    
    def _should_fully_merge(self, similarity: float, high_threshold: float) -> bool:
        """Determine if facts should be fully merged based on similarity
        
        Args:
            similarity: Similarity score between two facts
            high_threshold: High similarity threshold for full merge
        
        Returns:
            True if should fully merge, False if should create SUPERSEDES edge
        """
        return similarity >= high_threshold
    
    def _fully_merge_fact(self, source_id: str, target_id: str) -> None:
        """Fully merge a source fact into target fact
        
        This method:
        1. Merges SUPPORTED_BY edges to target
        2. Merges HAS_ATTRIBUTE edges to target
        3. Updates vector store (removes source, keeps target)
        4. Deletes source node
        
        Args:
            source_id: Source fact ID to merge
            target_id: Target (canonical) fact ID
        """
        if source_id not in self.graph._g:
            return
        
        # Transfer SUPPORTED_BY edges
        self._transfer_support_edges(source_id, target_id)
        
        # Transfer HAS_ATTRIBUTE edges (incoming edges from Person)
        in_edges = list(self.graph._g.in_edges(source_id, data=True))
        for person_id, _, edge_data in in_edges:
            if edge_data.get('rel_type') == 'HAS_ATTRIBUTE':
                # Check if target already has this edge
                existing = False
                for p_id, _, _ in self.graph._g.in_edges(target_id, data=True):
                    if p_id == person_id:
                        existing = True
                        break
                
                if not existing:
                    # Add edge from person to target
                    self.graph._g.add_edge(person_id, target_id, **edge_data)
        
        # Update vector store: remove source, keep target
        if self.vector_store:
            try:
                self.vector_store.remove_typed(source_id, "fact")
            except Exception as e:
                print(f"[FactDeduplicator] Warning: Failed to remove vector for {source_id}: {e}")
        
        # Delete source node
        self.graph.remove_node(source_id)
    
    def _limit_supported_by_edges(self, fact_id: str, max_edges: int = 5) -> None:
        """Limit the number of SUPPORTED_BY edges for a fact
        
        Keeps the most important events based on:
        1. Time (newer events preferred)
        2. Confidence (higher confidence preferred)
        
        Args:
            fact_id: Fact ID
            max_edges: Maximum number of SUPPORTED_BY edges to keep
        """
        if fact_id not in self.graph._g:
            return
        
        # Get all SUPPORTED_BY edges
        out_edges = list(self.graph._g.out_edges(fact_id, data=True))
        supported_by_edges = [
            (event_id, edge_data) 
            for _, event_id, edge_data in out_edges
            if edge_data.get('rel_type') == 'SUPPORTED_BY'
        ]
        
        if len(supported_by_edges) <= max_edges:
            return  # No need to limit
        
        # Score edges by importance
        scored_edges = []
        for event_id, edge_data in supported_by_edges:
            event_data = self.graph.get_node(event_id) or {}
            
            # Score: time (newer = higher) + confidence
            time_score = 0.0
            time_start = event_data.get('time_start')
            if time_start:
                try:
                    # Parse time_start to datetime for comparison
                    # Use global datetime import from top of file
                    dt = datetime.fromisoformat(time_start.replace('Z', '+00:00'))
                    # Use timestamp as score (newer = higher)
                    time_score = dt.timestamp()
                except:
                    pass
            
            confidence = event_data.get('confidence', 0.5)
            edge_confidence = edge_data.get('confidence', 1.0)
            
            # Combined score: time (normalized) + confidence
            score = time_score + confidence * 1000 + edge_confidence * 100
            scored_edges.append((score, event_id, edge_data))
        
        # Sort by score (descending) and keep top max_edges
        scored_edges.sort(key=lambda x: x[0], reverse=True)
        edges_to_keep = scored_edges[:max_edges]
        edges_to_remove = scored_edges[max_edges:]
        
        # Remove excess edges
        for _, event_id, _ in edges_to_remove:
            # Remove edge from graph
            if self.graph._g.has_edge(fact_id, event_id):
                # Find and remove the SUPPORTED_BY edge
                edges_to_delete = []
                for u, v, k, data in self.graph._g.out_edges(fact_id, keys=True, data=True):
                    if v == event_id and data.get('rel_type') == 'SUPPORTED_BY':
                        edges_to_delete.append((u, v, k))
                
                for u, v, k in edges_to_delete:
                    self.graph._g.remove_edge(u, v, k)
    
    def merge_duplicates(
        self,
        duplicate_groups: Dict[str, List[Tuple[str, Dict]]],
        similarity_scores: Dict[Tuple[str, str], float],
        create_supersedes: bool = True,
        high_threshold: float = 0.95,
        max_supported_by: int = 5
    ) -> Dict[str, str]:
        """Merge duplicate Facts with tiered strategy
        
        Strategy:
        - High similarity (>high_threshold): Fully merge (delete source node)
        - Medium similarity (threshold <= sim < high_threshold): Create SUPERSEDES edge
        - Low similarity (<threshold): Skip
        
        Args:
            duplicate_groups: Groups of duplicate facts from find_duplicates()
            similarity_scores: Dict mapping (fact_id1, fact_id2) -> similarity score
            create_supersedes: Whether to create SUPERSEDES edges for medium similarity
            high_threshold: High similarity threshold for full merge (default: 0.95)
            max_supported_by: Maximum SUPPORTED_BY edges per fact (default: 5)
        
        Returns:
            Mapping from old fact IDs to canonical fact IDs
        """
        merge_map: Dict[str, str] = {}
        fully_merged_count = 0
        supersedes_count = 0
        
        for key, group in duplicate_groups.items():
            if len(group) < 2:
                continue
            
            # Select canonical fact
            canonical_id, canonical_data = self._select_canonical_fact(group)
            
            for fact_id, fact_data in group:
                if fact_id == canonical_id:
                    continue
                
                # Calculate similarity to canonical
                canonical_key = tuple(sorted([fact_id, canonical_id]))
                similarity = similarity_scores.get(canonical_key, 0.0)
                
                # Determine merge strategy
                if self._should_fully_merge(similarity, high_threshold):
                    # High similarity: fully merge
                    self._fully_merge_fact(fact_id, canonical_id)
                    merge_map[fact_id] = canonical_id
                    self._stats["facts_merged"] += 1
                    fully_merged_count += 1
                else:
                    # Medium similarity: create SUPERSEDES edge
                    merge_map[fact_id] = canonical_id
                    self._stats["facts_merged"] += 1
                    
                # Create SUPERSEDES edge (canonical supersedes old)
                if create_supersedes:
                    supersedes_rel = Relation(
                        head_id=canonical_id,
                        tail_id=fact_id,
                        rel_type='SUPERSEDES',
                        weight=1.0,
                        confidence=1.0
                    )
                    self.graph.add_relations([supersedes_rel])
                    self._stats["supersedes_edges_created"] += 1
                    supersedes_count += 1
                
                # Mark old fact: valid_to, is_archived, is_latest=False, memory_lifecycle=archived
                if fact_id in self.graph._g:
                    self.graph._g.nodes[fact_id]['valid_to'] = datetime.now().strftime("%Y-%m-%d")
                    self.graph._g.nodes[fact_id]['is_archived'] = True
                    self.graph._g.nodes[fact_id]['is_latest'] = False
                    self.graph._g.nodes[fact_id]['memory_lifecycle'] = 'archived'
                    self._stats["facts_updated"] += 1
                
                # Transfer SUPPORTED_BY edges to canonical
                self._transfer_support_edges(fact_id, canonical_id)
            
            # Limit SUPPORTED_BY edges for canonical fact
            self._limit_supported_by_edges(canonical_id, max_supported_by)
            # Ensure canonical (head) has is_latest=True and memory_lifecycle=stable
            if canonical_id in self.graph._g:
                self.graph._g.nodes[canonical_id]['is_latest'] = True
                self.graph._g.nodes[canonical_id]['memory_lifecycle'] = 'stable'
        
        print(f"[FactDeduplicator] Merged {self._stats['facts_merged']} facts")
        print(f"[FactDeduplicator]   - Fully merged: {fully_merged_count}")
        print(f"[FactDeduplicator]   - SUPERSEDES edges created: {supersedes_count}")
        
        return merge_map
    
    def _transfer_support_edges(self, source_id: str, target_id: str):
        """Transfer SUPPORTED_BY edges from source fact to target fact"""
        if source_id not in self.graph._g:
            return
        
        # Get all SUPPORTED_BY edges from source
        out_edges = list(self.graph._g.out_edges(source_id, data=True))
        
        for _, event_id, edge_data in out_edges:
            if edge_data.get('rel_type') == 'SUPPORTED_BY':
                # Check if target already has this edge
                existing = False
                for _, e_id, _ in self.graph._g.out_edges(target_id, data=True):
                    if e_id == event_id:
                        existing = True
                        break
                
                if not existing:
                    # Add edge from target to event
                    self.graph._g.add_edge(target_id, event_id, **edge_data)
    
    def deduplicate(self, facts: List[Fact]) -> List[Fact]:
        """Deduplicate a list of facts (for new facts before insertion)
        
        Args:
            facts: List of Fact objects to deduplicate
        
        Returns:
            Deduplicated list of Facts
        """
        if len(facts) < 2:
            return facts
        
        # Group by triple
        triple_groups: Dict[str, List[Fact]] = defaultdict(list)
        
        for fact in facts:
            key = f"{fact.subject_id}|{fact.predicate}|{fact.object_id or fact.purpose_id}"
            triple_groups[key].append(fact)
        
        # Select canonical from each group
        unique_facts: List[Fact] = []
        
        for key, group in triple_groups.items():
            if len(group) == 1:
                unique_facts.append(group[0])
            else:
                # Select the one with highest confidence
                canonical = max(group, key=lambda f: f.confidence or 0.5)
                unique_facts.append(canonical)
        
        return unique_facts
    
    def update_fact(self, old_fact_id: str, new_fact: Fact) -> str:
        """Update an existing fact with a new version
        
        Args:
            old_fact_id: ID of the fact to update
            new_fact: New fact data
        
        Returns:
            ID of the new fact
        """
        # Mark old fact as no longer valid
        if old_fact_id in self.graph._g:
            self.graph._g.nodes[old_fact_id]['valid_to'] = datetime.now().strftime("%Y-%m-%d")
        
        # Add new fact
        self.graph.add_node(new_fact)
        
        # Create SUPERSEDES edge
        supersedes_rel = Relation(
            head_id=new_fact.id,
            tail_id=old_fact_id,
            rel_type='SUPERSEDES',
            weight=1.0,
            confidence=1.0
        )
        self.graph.add_relations([supersedes_rel])
        
        return new_fact.id
    
    def run(
        self, 
        create_supersedes: bool = True,
        high_threshold: float = 0.95,
        max_supported_by: int = 5
    ) -> Dict[str, str]:
        """Run the full fact deduplication process
        
        Args:
            create_supersedes: Whether to create SUPERSEDES edges for medium similarity
            high_threshold: High similarity threshold for full merge (default: 0.95)
            max_supported_by: Maximum SUPPORTED_BY edges per fact (default: 5)
        
        Returns:
            Mapping from old fact IDs to canonical fact IDs
        """
        print(f"[FactDeduplicator] Starting fact deduplication (high_threshold={high_threshold}, max_supported_by={max_supported_by})")
        
        # Find duplicates with similarity scores
        duplicate_groups, similarity_scores = self.find_duplicates()
        
        # Merge duplicates with tiered strategy
        merge_map = self.merge_duplicates(
            duplicate_groups, 
            similarity_scores,
            create_supersedes=create_supersedes,
            high_threshold=high_threshold,
            max_supported_by=max_supported_by
        )
        
        print(f"[FactDeduplicator] Deduplication complete:")
        print(f"  - Facts processed: {self._stats['facts_processed']}")
        print(f"  - Duplicates found: {self._stats['duplicates_found']}")
        print(f"  - Facts merged: {self._stats['facts_merged']}")
        print(f"  - SUPERSEDES edges created: {self._stats['supersedes_edges_created']}")
        
        return merge_map
    
    def get_stats(self) -> Dict[str, int]:
        """Get deduplication statistics"""
        return self._stats.copy()
