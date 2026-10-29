"""Read-only concept similarity diagnostics; identity needs source validation."""

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
    """Similarity does not prove identity or authorize destructive updates."""
    
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

        # Warm the same per-concept vectors in bounded batches. The legacy
        # typed store's get_vector() is a placeholder, so lazy pairwise lookup
        # otherwise makes one remote embedding request for every concept.
        if self.use_embedding:
            pending = []
            for concept_id, data in concepts:
                if concept_id in self._concept_embeddings:
                    continue
                vector = self.vector_store.get_vector(concept_id) if self.vector_store else None
                if vector is not None:
                    self._concept_embeddings[concept_id] = vector
                    continue
                text = f"{data.get('name', '')} {data.get('description', '')}".strip()
                if text:
                    pending.append((concept_id, text))
            for start in range(0, len(pending), 10):
                batch = pending[start:start + 10]
                vectors = self.embedding_encoder.encode([text for _, text in batch])
                if len(vectors) != len(batch):
                    raise ValueError("Concept embedding batch cardinality mismatch")
                self._concept_embeddings.update((item[0], vector) for item, vector in zip(batch, vectors))
        
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
    
    def merge_concepts(self, synonym_groups, similarity_scores, create_same_as=True, high_threshold=0.95):
        """Similarity candidates do not authorize identity or destructive graph updates."""
        return {}
    
    def run(self, threshold=0.85, create_same_as=True, high_threshold=0.95):
        """Keep concepts separate until identity is established by source evidence.

        find_synonyms remains a read-only diagnostic. Cosine scores alone do not
        authorize alias transfer, SAME_AS edges, vector deletion or node deletion.
        """
        return {}
    
    def get_stats(self) -> Dict[str, int]:
        """Get merge statistics"""
        return self._stats.copy()
