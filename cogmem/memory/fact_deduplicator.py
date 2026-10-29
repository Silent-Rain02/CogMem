"""Read-only similarity diagnostics for facts; source occurrences are never collapsed."""

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
    """Similarity candidates are diagnostic only; keep every source observation."""
    
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

        if self.embedding_encoder and len(facts) > 1:
            pending = []
            for fact_id, data in facts:
                if fact_id in self._fact_embeddings:
                    continue
                vector = self.vector_store.get_vector(fact_id) if self.vector_store else None
                if vector is not None:
                    self._fact_embeddings[fact_id] = vector
                    continue
                text = f"{data.get('content', '')} {data.get('predicate', '')} {data.get('object_id', '')}".strip()
                if text:
                    pending.append((fact_id, text))
            for start in range(0, len(pending), 10):
                batch = pending[start:start + 10]
                vectors = self.embedding_encoder.encode([text for _, text in batch])
                if len(vectors) != len(batch):
                    raise ValueError("Fact embedding batch cardinality mismatch")
                self._fact_embeddings.update((item[0], vector) for item, vector in zip(batch, vectors))
        
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
    
    def merge_duplicates(self, duplicate_groups, similarity_scores, create_supersedes=True,
                         high_threshold=0.95, max_supported_by=5):
        """Keep source occurrences independent; similarity does not prove an update."""
        return {}
    
    def deduplicate(self, facts: List[Fact]) -> List[Fact]:
        """Preserve independent observations, even when their triples match."""
        return list(facts)

    def update_fact(self, old_fact_id: str, new_fact: Fact) -> str:
        """Refuse unvalidated replacement; callers must append observations."""
        raise ValueError("Fact supersession requires source-backed context and validity")

    def run(self, high_threshold=0.95, max_supported_by=5, create_supersedes=True):
        """Do not collapse or expire facts without an explicit source-backed update.

        Equal triples may coexist across time and context; similar sentences may
        describe different subjects. find_duplicates is diagnostic only.
        """
        return {}
    
    def get_stats(self) -> Dict[str, int]:
        """Get deduplication statistics"""
        return self._stats.copy()
