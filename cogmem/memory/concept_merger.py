"""Concept Merger Module

This module implements semantic-based concept deduplication and merging.
It detects synonymous concepts using embedding similarity and merges them
into canonical forms while maintaining SAME_AS relationships.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

from ..storage.graph_store import GraphStore
from ..storage.typed_vector_store import TypedVectorStore
from ..storage.schema import Concept, Relation, NodeTypeEnum, Metadata
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

def _edit_distance_similarity(s1: str, s2: str) -> float:
    """Calculate string similarity based on Levenshtein distance (fallback when no embedding)"""
    if not s1 or not s2:
        return 0.0
    
    # Simple edit distance calculation
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    
    # Create distance matrix
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    
    max_len = max(len1, len2)
    return 1.0 - (dp[len1][len2] / max_len)

class ConceptMerger:
    """Concept Merger: Detects and merges synonymous concepts
    
    This class implements semantic-based concept deduplication:
    1. Generate embeddings for all concepts (if embedding available)
    2. Calculate pairwise similarity
    3. Group synonymous concepts (similarity > threshold)
    4. Merge into canonical forms
    5. Create SAME_AS edges to maintain provenance
    
    Usage:
        merger = ConceptMerger(graph_store, vector_store, embedding_encoder)
        
        # Find synonym groups
        synonym_groups = merger.find_synonyms(threshold=0.85)
        
        # Merge concepts
        merge_map = merger.merge_concepts(synonym_groups)
    """
    
    def __init__(
        self,
        graph: GraphStore,
        vector_store: Optional[TypedVectorStore] = None,
        embedding_encoder: Optional[EmbeddingEncoder] = None,
        use_embedding: bool = True
    ):
        """Initialize ConceptMerger
        
        Args:
            graph: Graph store instance
            vector_store: Vector store instance (optional)
            embedding_encoder: Embedding encoder instance (optional)
            use_embedding: Whether to use embedding for similarity (if False, use edit distance)
        """
        self.graph = graph
        self.vector_store = vector_store
        self.embedding_encoder = embedding_encoder
        self.use_embedding = use_embedding and embedding_encoder is not None
        
        # Cache for concept embeddings
        self._concept_embeddings: Dict[str, List[float]] = {}
        
        # Statistics
        self._stats = {
            "concepts_processed": 0,
            "synonym_groups_found": 0,
            "concepts_merged": 0,
            "same_as_edges_created": 0
        }
    
    def _get_all_concepts(self) -> List[Tuple[str, Dict]]:
        """Get all concept nodes from the graph"""
        concepts = []
        for node_id, node_data in self.graph._g.nodes(data=True):
            if node_data and node_data.get('node_type') == NodeTypeEnum.CONCEPT.value:
                concepts.append((node_id, node_data))
        return concepts
    
    def _get_concept_embedding(self, concept_id: str, concept_data: Dict) -> Optional[List[float]]:
        """Get or generate embedding for a concept"""
        # Check cache first
        if concept_id in self._concept_embeddings:
            return self._concept_embeddings[concept_id]
        
        # Try to get from vector store
        if self.vector_store:
            try:
                vec = self.vector_store.get_vector(concept_id)
                if vec is not None:
                    self._concept_embeddings[concept_id] = vec
                    return vec
            except:
                pass
        
        # Generate new embedding
        if self.embedding_encoder:
            name = concept_data.get('name', '')
            description = concept_data.get('description', '')
            text = f"{name} {description}".strip()
            
            if text:
                try:
                    embeddings = self.embedding_encoder.encode([text])
                    if embeddings and embeddings[0]:
                        self._concept_embeddings[concept_id] = embeddings[0]
                        return embeddings[0]
                except Exception as e:
                    print(f"[ConceptMerger] Failed to generate embedding for {concept_id}: {e}")
        
        return None
    
    def _calculate_similarity(
        self,
        concept1: Tuple[str, Dict],
        concept2: Tuple[str, Dict]
    ) -> float:
        """Calculate similarity between two concepts"""
        id1, data1 = concept1
        id2, data2 = concept2
        
        if self.use_embedding:
            # Use embedding similarity
            emb1 = self._get_concept_embedding(id1, data1)
            emb2 = self._get_concept_embedding(id2, data2)
            
            if emb1 is not None and emb2 is not None:
                return _cosine_similarity(emb1, emb2)
        
        # Fallback to edit distance similarity
        name1 = data1.get('name', id1).lower()
        name2 = data2.get('name', id2).lower()
        
        return _edit_distance_similarity(name1, name2)
    
    def find_synonyms(
        self, 
        threshold: float = 0.85,
        high_threshold: float = 0.95
    ) -> Tuple[List[Set[str]], Dict[Tuple[str, str], float]]:
        """Find groups of synonymous concepts with similarity scores
        
        Args:
            threshold: Similarity threshold for considering concepts as synonyms
            high_threshold: High similarity threshold for full merge (default: 0.95)
        
        Returns:
            Tuple of:
            - List of sets, each set contains IDs of synonymous concepts
            - Dict mapping (concept_id1, concept_id2) -> similarity score
        """
        concepts = self._get_all_concepts()
        self._stats["concepts_processed"] = len(concepts)
        
        if len(concepts) < 2:
            return [], {}
        
        print(f"[ConceptMerger] Processing {len(concepts)} concepts...")
        
        # Build adjacency graph for connected components
        # concept_id -> set of similar concept_ids
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        similarity_scores: Dict[Tuple[str, str], float] = {}
        
        # Calculate pairwise similarities
        n = len(concepts)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._calculate_similarity(concepts[i], concepts[j])
                
                if sim >= threshold:
                    id1, id2 = concepts[i][0], concepts[j][0]
                    adjacency[id1].add(id2)
                    adjacency[id2].add(id1)
                    # Store similarity score (use sorted tuple as key for consistency)
                    key = tuple(sorted([id1, id2]))
                    similarity_scores[key] = sim
        
        # Find connected components (synonym groups)
        visited: Set[str] = set()
        groups: List[Set[str]] = []
        
        for concept_id, _ in concepts:
            if concept_id in visited:
                continue
            
            if concept_id not in adjacency:
                continue  # No synonyms
            
            # BFS to find all connected concepts
            group: Set[str] = set()
            queue = [concept_id]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                
                visited.add(current)
                group.add(current)
                
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            if len(group) > 1:
                groups.append(group)
        
        self._stats["synonym_groups_found"] = len(groups)
        print(f"[ConceptMerger] Found {len(groups)} synonym groups")
        
        return groups, similarity_scores
    
    def _select_canonical(self, group: Set[str]) -> str:
        """Select the canonical concept from a synonym group
        
        Selection criteria (in order):
        1. Most connections (edges)
        2. Shortest name (more concise)
        3. Alphabetically first
        """
        candidates = list(group)
        
        def score(concept_id: str) -> Tuple[int, int, str]:
            # Count edges
            in_edges = self.graph._g.in_degree(concept_id) if concept_id in self.graph._g else 0
            out_edges = self.graph._g.out_degree(concept_id) if concept_id in self.graph._g else 0
            edge_count = in_edges + out_edges
            
            # Get name length
            data = self.graph.get_node(concept_id) or {}
            name = data.get('name', concept_id)
            name_len = len(name)
            
            # Return tuple for sorting (negative edge_count for descending)
            return (-edge_count, name_len, concept_id)
        
        candidates.sort(key=score)
        return candidates[0]
    
    def _should_fully_merge(self, similarity: float, high_threshold: float) -> bool:
        """Determine if concepts should be fully merged based on similarity
        
        Args:
            similarity: Similarity score between two concepts
            high_threshold: High similarity threshold for full merge
        
        Returns:
            True if should fully merge, False if should create SAME_AS edge
        """
        return similarity >= high_threshold
    
    def _fully_merge_concept(self, source_id: str, target_id: str) -> None:
        """Fully merge a source concept into target concept
        
        This method:
        1. Redirects all edges from source to target
        2. Merges aliases into target
        3. Updates vector store (removes source, keeps target)
        4. Deletes source node
        
        Args:
            source_id: Source concept ID to merge
            target_id: Target (canonical) concept ID
        """
        if source_id not in self.graph._g:
            return
        
        source_data = self.graph.get_node(source_id) or {}
        target_data = self.graph.get_node(target_id) or {}
        
        # Merge aliases
        source_name = source_data.get('name', '')
        source_aliases = source_data.get('aliases', [])
        target_aliases = target_data.get('aliases', [])
        
        merged_aliases = set(target_aliases)
        if source_name:
            merged_aliases.add(source_name)
        merged_aliases.update(source_aliases)
        
        # Update target node with merged aliases
        if target_id in self.graph._g:
            self.graph._g.nodes[target_id]['aliases'] = list(merged_aliases)
        
        # Redirect all edges from source to target
        self._redirect_edges(source_id, target_id)
        
        # Update vector store: remove source, keep target
        if self.vector_store:
            try:
                self.vector_store.remove_typed(source_id, "concept")
            except Exception as e:
                print(f"[ConceptMerger] Warning: Failed to remove vector for {source_id}: {e}")
        
        # Delete source node
        self.graph.remove_node(source_id)
    
    def merge_concepts(
        self, 
        synonym_groups: List[Set[str]], 
        similarity_scores: Dict[Tuple[str, str], float],
        create_same_as: bool = True,
        high_threshold: float = 0.95
    ) -> Dict[str, str]:
        """Merge synonymous concepts with tiered strategy
        
        Strategy:
        - High similarity (>high_threshold): Fully merge (delete source node)
        - Medium similarity (threshold <= sim < high_threshold): Create SAME_AS edge
        - Low similarity (<threshold): Skip
        
        Args:
            synonym_groups: List of synonym groups from find_synonyms()
            similarity_scores: Dict mapping (id1, id2) -> similarity score
            create_same_as: Whether to create SAME_AS edges for medium similarity
            high_threshold: High similarity threshold for full merge (default: 0.95)
        
        Returns:
            Mapping from old concept IDs to new canonical IDs
        """
        merge_map: Dict[str, str] = {}  # old_id -> canonical_id
        fully_merged_count = 0
        same_as_count = 0
        
        for group in synonym_groups:
            if len(group) < 2:
                continue
            
            # Select canonical concept
            canonical_id = self._select_canonical(group)
            
            # Get canonical concept data
            canonical_data = self.graph.get_node(canonical_id) or {}
            
            # Collect aliases from all concepts in the group
            all_aliases: Set[str] = set()
            
            for concept_id in group:
                if concept_id == canonical_id:
                    continue
                
                concept_data = self.graph.get_node(concept_id) or {}
                
                # Calculate similarity to canonical
                canonical_key = tuple(sorted([concept_id, canonical_id]))
                similarity = similarity_scores.get(canonical_key, 0.0)
                
                # Determine merge strategy
                if self._should_fully_merge(similarity, high_threshold):
                    # High similarity: fully merge
                    self._fully_merge_concept(concept_id, canonical_id)
                    merge_map[concept_id] = canonical_id
                    self._stats["concepts_merged"] += 1
                    fully_merged_count += 1
                else:
                    # Medium similarity: create SAME_AS edge
                    # Add the concept's name as an alias
                    name = concept_data.get('name', '')
                    if name:
                        all_aliases.add(name)
                
                # Add existing aliases
                existing_aliases = concept_data.get('aliases', [])
                if existing_aliases:
                    all_aliases.update(existing_aliases)
                
                # Record mapping
                merge_map[concept_id] = canonical_id
                self._stats["concepts_merged"] += 1
                
                # Create SAME_AS edge
                if create_same_as:
                    same_as_rel = Relation(
                        head_id=concept_id,
                        tail_id=canonical_id,
                        rel_type='SAME_AS',
                        weight=1.0,
                        confidence=1.0
                    )
                    self.graph.add_relations([same_as_rel])
                    self._stats["same_as_edges_created"] += 1
                    same_as_count += 1
                
                # Redirect edges from merged concept to canonical
                self._redirect_edges(concept_id, canonical_id)
            
            # Update canonical concept with collected aliases (for medium similarity merges)
            if all_aliases:
                existing_aliases = canonical_data.get('aliases', [])
                merged_aliases = list(set(existing_aliases) | all_aliases)
                
                # Update the node
                if canonical_id in self.graph._g:
                    self.graph._g.nodes[canonical_id]['aliases'] = merged_aliases
        
        print(f"[ConceptMerger] Merged {self._stats['concepts_merged']} concepts")
        print(f"[ConceptMerger]   - Fully merged: {fully_merged_count}")
        print(f"[ConceptMerger]   - SAME_AS edges created: {same_as_count}")
        
        return merge_map
    
    def _redirect_edges(self, source_id: str, target_id: str):
        """Redirect all edges from source concept to target concept"""
        if source_id not in self.graph._g:
            return
        
        # Redirect incoming edges
        in_edges = list(self.graph._g.in_edges(source_id, data=True))
        for src, _, data in in_edges:
            if src != target_id:  # Avoid self-loops
                self.graph._g.add_edge(src, target_id, **data)
        
        # Redirect outgoing edges
        out_edges = list(self.graph._g.out_edges(source_id, data=True))
        for _, dst, data in out_edges:
            if dst != target_id:  # Avoid self-loops
                self.graph._g.add_edge(target_id, dst, **data)
    
    def run(
        self, 
        threshold: float = 0.90, 
        create_same_as: bool = True,
        high_threshold: float = 0.95
    ) -> Dict[str, str]:
        """Run the full concept merge process
        
        Args:
            threshold: Similarity threshold for synonyms (default: 0.90, increased from 0.85)
            create_same_as: Whether to create SAME_AS edges for medium similarity
            high_threshold: High similarity threshold for full merge (default: 0.95)
        
        Returns:
            Mapping from old concept IDs to canonical IDs
        """
        print(f"[ConceptMerger] Starting concept merge (threshold={threshold}, high_threshold={high_threshold})")
        
        # Find synonym groups with similarity scores
        synonym_groups, similarity_scores = self.find_synonyms(
            threshold=threshold,
            high_threshold=high_threshold
        )
        
        # Merge concepts with tiered strategy
        merge_map = self.merge_concepts(
            synonym_groups, 
            similarity_scores,
            create_same_as=create_same_as,
            high_threshold=high_threshold
        )
        
        print(f"[ConceptMerger] Merge complete:")
        print(f"  - Concepts processed: {self._stats['concepts_processed']}")
        print(f"  - Synonym groups found: {self._stats['synonym_groups_found']}")
        print(f"  - Concepts merged: {self._stats['concepts_merged']}")
        print(f"  - SAME_AS edges created: {self._stats['same_as_edges_created']}")
        
        return merge_map
    
    def get_stats(self) -> Dict[str, int]:
        """Get merge statistics"""
        return self._stats.copy()
