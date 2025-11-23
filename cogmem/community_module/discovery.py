from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from ..config import CogMemConfig
from ..storage.schema import Community, Topic, Relation, Metadata, NodeType
from ..models.llm_client import LLMClient


def _try_import_leiden():
    try:
        import igraph as ig  # type: ignore
        import leidenalg as la  # type: ignore
        return ig, la
    except Exception:
        return None, None


class CommunityDiscovery:
    def __init__(self, cfg: CogMemConfig, llm: Optional[LLMClient] = None) -> None:
        """Community discovery with semantic and structure methods

        Args:
            cfg: Configuration object
            llm: Optional LLM client for generating topic labels and descriptions
        """
        self._cfg = cfg
        self._llm = llm
        self._initialized = llm is not None

    def initialize_llm(self, llm: LLMClient) -> None:
        """Initialize LLM client for topic generation"""
        self._llm = llm
        self._initialized = True

    def _generate_topic_label(
        self,
        member_ids: List[str],
        member_texts: List[str],
        graph_store: Any
    ) -> Tuple[str, str, List[str]]:
        """Generate topic label, description, and keywords using LLM

        Args:
            member_ids: List of node IDs in this cluster
            member_texts: List of text representations of members
            graph_store: Graph store to query member details

        Returns:
            Tuple of (label, description, keywords)
        """
        if not self._initialized or not self._llm:
            # Fallback: Generate simple label from most common words
            return self._fallback_topic_label(member_texts)

        try:
            # Prepare context for LLM
            # Limit to top 20 members to avoid context overflow
            sample_texts = member_texts[:20]
            combined_text = "\n".join(f"- {text}" for text in sample_texts)

            # Get member types for context
            member_types = set()
            for member_id in member_ids[:10]:  # Sample first 10
                node_data = graph_store.get_node(member_id)
                if node_data:
                    node_type = node_data.get("node_type", "unknown")
                    member_types.add(node_type)

            member_types_str = ", ".join(sorted(member_types))

            prompt = f"""You are a knowledge graph analyst. Analyze the following {len(member_ids)} nodes ({member_types_str}) and generate:

1. A concise, human-readable topic label (2-5 words)
2. A brief description (1-2 sentences)
3. 5-10 relevant keywords

Nodes in this cluster:
{combined_text}

Provide your response in JSON format:
{{
  "label": "Topic Label",
  "description": "Brief description of what this topic covers.",
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
}}"""

            response = self._llm.complete(prompt)
            result = self._parse_llm_response(response)

            if result:
                return result["label"], result["description"], result["keywords"]
            else:
                # Fallback if parsing fails
                return self._fallback_topic_label(member_texts)

        except Exception as e:
            print(f"[CommunityDiscovery] Error generating topic label: {e}")
            return self._fallback_topic_label(member_texts)

    def _fallback_topic_label(self, member_texts: List[str]) -> Tuple[str, str, List[str]]:
        """Fallback method when LLM is not available or fails"""
        if not member_texts:
            return "Unknown Topic", "No description available.", []

        # Simple keyword extraction: most frequent meaningful words
        from collections import Counter
        import re

        all_text = " ".join(member_texts).lower()
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text)  # Words with 4+ letters

        # Filter out common stop words
        stop_words = {
            'this', 'that', 'these', 'those', 'with', 'have', 'been', 'will',
            'shall', 'should', 'could', 'would', 'might', 'must', 'than', 'then',
            'also', 'like', 'over', 'under', 'more', 'most', 'such', 'only',
            'just', 'well', 'even', 'ever', 'once', 'twice', 'time', 'very',
            'here', 'there', 'when', 'where', 'what', 'which', 'who', 'whom',
            'whose', 'how', 'why', 'much', 'many', 'some', 'any', 'each'
        }

        filtered_words = [w for w in words if w not in stop_words]

        if not filtered_words:
            return "Miscellaneous", "A collection of various items.", []

        # Get most common words
        counter = Counter(filtered_words)
        keywords = [word for word, _ in counter.most_common(10)]

        # Generate label from top 2-3 keywords
        label_words = keywords[:3]
        label = " ".join(word.capitalize() for word in label_words)

        # Generate simple description
        description = f"A collection of {len(member_texts)} items related to {', '.join(keywords[:5])}."

        return label, description, keywords

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response for topic generation"""
        import json
        import re

        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)

                # Validate required fields
                if "label" in parsed and "description" in parsed and "keywords" in parsed:
                    return {
                        "label": str(parsed["label"]),
                        "description": str(parsed["description"]),
                        "keywords": list(parsed["keywords"])[:10]  # Limit to 10 keywords
                    }
        except Exception as e:
            print(f"[CommunityDiscovery] Error parsing LLM response: {e}")

        return None

    def semantic_clusters(self, entity_ids: List[str], vectors: List[List[float]]) -> List[Community]:
        if not entity_ids:
            return []

        print(f"[CommunityDiscovery] Preparing semantic clustering for {len(entity_ids)} entities...")

        # Prepare data
        k = min(self._cfg.community.semantic_k, max(1, len(entity_ids)))
        X = np.asarray(vectors, dtype=np.float32)
        if len(entity_ids) < k:
            k = len(entity_ids)

        print(f"[CommunityDiscovery] Using {k} clusters for {len(entity_ids)} entities")

        # K-means clustering with progress indication
        print(f"[CommunityDiscovery] Running K-means clustering (k={k})...")
        km = KMeans(n_clusters=k, n_init=10, random_state=42, verbose=0)

        # Use tqdm to show progress for large datasets
        if len(entity_ids) > 1000:
            print(f"[CommunityDiscovery] Processing {len(entity_ids)} entities, this may take a while...")

        labels = km.fit_predict(X)

        # Group entities by cluster
        print(f"[CommunityDiscovery] Organizing {len(entity_ids)} entities into clusters...")
        coms: Dict[int, List[str]] = {}
        for eid, lb in tqdm(zip(entity_ids, labels), total=len(entity_ids), desc="Grouping entities"):
            coms.setdefault(int(lb), []).append(eid)

        # Create community objects
        print(f"[CommunityDiscovery] Creating {len(coms)} semantic communities...")
        results: List[Community] = []
        for cid, members in tqdm(coms.items(), total=len(coms), desc="Building communities"):
            results.append(
                Community(
                    id=f"sem-{cid}",
                    kind="semantic",
                    label=f"sem-{cid}",
                    score=None,
                    member_ids=members,
                )
            )

        # Summary statistics
        total_members = sum(len(members) for members in coms.values())
        avg_cluster_size = total_members / len(coms) if coms else 0

        print(f"[CommunityDiscovery] Completed: {len(results)} communities, "
              f"avg size: {avg_cluster_size:.1f} entities")

        return results

    def structure_leiden(self, edges: List[Tuple[str, str]]) -> List[Community]:
        ig, la = _try_import_leiden()
        if ig is None or la is None or not edges:
            return []
        # 构建 igraph 图
        nodes = {}  # id -> idx
        idx = 0
        for u, v in edges:
            if u not in nodes:
                nodes[u] = idx; idx += 1
            if v not in nodes:
                nodes[v] = idx; idx += 1
        g = ig.Graph(n=len(nodes), directed=True)
        g.add_edges([(nodes[u], nodes[v]) for u, v in edges])
        # 使用配置的分辨率参数
        part = la.find_partition(g, la.RBConfigurationVertexPartition, resolution_parameter=self._cfg.community.leiden_resolution)
        communities: Dict[int, List[str]] = {}
        for node_idx, com_idx in enumerate(part.membership):
            # 反查节点 id
            nid = list(nodes.keys())[list(nodes.values()).index(node_idx)]
            communities.setdefault(int(com_idx), []).append(nid)
        results: List[Community] = []
        for cid, members in communities.items():
            results.append(
                Community(
                    id=f"str-{cid}",
                    kind="structure",
                    label=f"str-{cid}",
                    member_ids=members,
                )
            )
        return results
    
    def semantic_clusters_multi_community(
        self,
        entity_ids: List[str],
        vectors: List[List[float]],
        threshold: float = 0.3
    ) -> Tuple[List[Community], Dict[str, List[Tuple[str, float]]]]:
        """Semantic clustering with multi-community membership
        
        Args:
            entity_ids: List of entity IDs
            vectors: List of embedding vectors
            threshold: Minimum membership weight for multi-community assignment
            
        Returns:
            Tuple of (communities, node_community_memberships)
            where node_community_memberships maps node_id -> [(community_id, weight), ...]
        """
        if not entity_ids:
            return [], {}
        
        # First, do standard clustering
        communities = self.semantic_clusters(entity_ids, vectors)
        
        # Calculate centroids
        X = np.asarray(vectors, dtype=np.float32)
        id_to_index = {eid: i for i, eid in enumerate(entity_ids)}
        community_centroids = {}
        
        for comm in communities:
            member_indices = [id_to_index[eid] for eid in comm.member_ids if eid in id_to_index]
            if member_indices:
                centroid = X[member_indices].mean(axis=0)
                community_centroids[comm.id] = centroid
        
        # Calculate membership weights for each node
        node_memberships: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        
        for eid, vec in zip(entity_ids, vectors):
            vec_array = np.array(vec, dtype=np.float32)
            weights = []
            
            # Calculate similarity to each community centroid
            for comm_id, centroid in community_centroids.items():
                # Cosine similarity
                similarity = np.dot(vec_array, centroid) / (np.linalg.norm(vec_array) * np.linalg.norm(centroid) + 1e-8)
                weights.append((comm_id, float(similarity)))
            
            # Sort by weight and filter by threshold
            weights.sort(key=lambda x: x[1], reverse=True)
            filtered_weights = [(cid, w) for cid, w in weights if w >= threshold]
            
            if filtered_weights:
                node_memberships[eid] = filtered_weights
            else:
                # Assign to top community if below threshold
                if weights:
                    node_memberships[eid] = [weights[0]]
        
        return communities, dict(node_memberships)
    
    def facet_based_communities(
        self,
        person_ids: List[str],
        facet_type: str,
        graph_store
    ) -> List[Community]:
        """Create communities based on PersonFacet nodes
        
        Args:
            person_ids: List of person IDs
            facet_type: Type of facet (work, health, finance, etc.)
            graph_store: GraphStore instance to query facets
        """
        # Group persons by facet
        facet_to_persons: Dict[str, List[str]] = defaultdict(list)
        
        for person_id in person_ids:
            # Query facets for this person
            # This is a simplified version - in practice, you'd query the graph
            facet_id = f"facet_{person_id}_{facet_type}"
            facet_to_persons[facet_id].append(person_id)
        
        communities = []
        for facet_id, members in facet_to_persons.items():
            if len(members) > 1:  # Only create community if multiple members
                communities.append(
                    Community(
                        id=f"facet-{facet_type}-{facet_id}",
                        kind="semantic",
                        label=f"facet-{facet_type}",
                        member_ids=members
                    )
                )
        
        return communities
    
    def weighted_heterogeneous_communities(
        self,
        nodes: List[Tuple[str, List[float]]],  # (node_id, embedding)
        edges: List[Tuple[str, str, float]],  # (head_id, tail_id, weight)
        semantic_weight: float = 0.6,
        structural_weight: float = 0.4
    ) -> List[Community]:
        """Create communities using weighted combination of semantic and structural similarity"""
        if not nodes:
            return []
        
        node_ids = [n[0] for n in nodes]
        vectors = [n[1] for n in nodes]
        
        # Semantic communities
        semantic_comms, _ = self.semantic_clusters_multi_community(node_ids, vectors)
        
        # Structural communities (from edges)
        edge_list = [(u, v) for u, v, _ in edges]
        structural_comms = self.structure_leiden(edge_list)
        
        # Combine communities (simplified: merge overlapping communities)
        # In practice, you'd use more sophisticated merging
        all_communities = semantic_comms + structural_comms
        
        return all_communities

    def update_topics(
        self,
        communities: List[Community],
        entity_ids: List[str],
        vectors: List[List[float]],
        graph_store: Any,
        cluster_type: str = "semantic"
    ) -> Tuple[List[Topic], List[Relation]]:
        """Generate TopicNodes from communities and create Topic -> Member edges

        Args:
            communities: List of communities from semantic_clusters()
            entity_ids: List of node IDs that were clustered
            vectors: List of embeddings corresponding to entity_ids
            graph_store: GraphStore to query node details and check existing nodes
            cluster_type: "semantic" or "structure" for ID prefix

        Returns:
            Tuple of (topics_to_add, relations_to_add)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        print(f"[CommunityDiscovery] Generating {len(communities)} topics from {cluster_type} communities...")

        topics_to_add = []
        relations_to_add = []

        # Prepare data
        X = np.asarray(vectors, dtype=np.float32)
        id_to_index = {eid: i for i, eid in enumerate(entity_ids)}

        for community in tqdm(communities, desc=f"Generating {cluster_type} topics"):
            member_ids = community.member_ids

            if not member_ids:
                continue

            # Get member indices
            member_indices = [id_to_index[eid] for eid in member_ids if eid in id_to_index]

            if not member_indices:
                continue

            # Compute centroid
            centroid = X[member_indices].mean(axis=0).tolist()

            # Get member texts for labeling
            member_texts = []
            member_types = set()

            for member_id in member_ids[:20]:  # Limit for efficiency
                node_data = graph_store.get_node(member_id)
                if not node_data:
                    continue

                node_type = node_data.get("node_type", "Entity")
                member_types.add(node_type)

                # Generate text representation based on node type
                text_rep = self._get_node_text_representation(node_data, node_type)
                if text_rep:
                    member_texts.append(text_rep)

            if not member_texts:
                continue

            # Generate label, description, keywords using LLM
            label, description, keywords = self._generate_topic_label(
                member_ids,
                member_texts,
                graph_store
            )

            # Create Topic node
            topic_id = f"topic_{cluster_type}_{community.id}"

            # Check if topic already exists
            existing_topic = None
            try:
                if graph_store.get_node(topic_id):
                    existing_data = graph_store._g.nodes[topic_id]
                    existing_topic = Topic(**existing_data)
            except Exception:
                existing_topic = None

            if existing_topic:
                # Update existing topic
                topic = existing_topic
                # Update metadata
                topic.keywords = keywords
                topic.description = description
                topic.centroid = centroid
                topic.size = len(member_ids)
            else:
                # Create new topic
                topic = Topic(
                    id=topic_id,
                    label=label,
                    description=description,
                    keywords=keywords,
                    member_ids=member_ids,
                    centroid=centroid,
                    source_community_id=community.id,
                    confidence=1.0,  # Could be adjusted based on cluster cohesion
                    member_types=list(member_types),
                    size=len(member_ids),
                    metadata=Metadata(
                        source=f"community_{cluster_type}_clustering",
                        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                )
                topics_to_add.append(topic)

            # Calculate confidence for each member based on similarity to centroid
            member_vectors = X[member_indices]
            centroid_array = np.array(centroid).reshape(1, -1)
            similarities = cosine_similarity(member_vectors, centroid_array).flatten()

            # Create COVERS edges: Topic -> Member
            for member_id, similarity in zip(member_ids, similarities):
                confidence = float(similarity)  # Use similarity as confidence

                rel = Relation(
                    head_id=topic.id,
                    tail_id=member_id,
                    rel_type="COVERS",
                    weight=confidence,
                    confidence=confidence,
                    metadata=Metadata(
                        source="topic_clustering",
                        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                )
                relations_to_add.append(rel)

        print(f"[CommunityDiscovery] Generated {len(topics_to_add)} topics and {len(relations_to_add)} COVERS edges")
        return topics_to_add, relations_to_add

    def _get_node_text_representation(self, node_data: Dict[str, Any], node_type: str) -> str:
        """Get text representation for a node based on its type"""
        if node_type == "Person":
            canonical_name = node_data.get("canonical_name", "")
            aliases = node_data.get("aliases", [])
            description = node_data.get("description", "")
            return f"{canonical_name} {' '.join(aliases)} {description}".strip()

        elif node_type == "Event":
            title = node_data.get("title", "")
            location = node_data.get("location", "")
            description = node_data.get("description", "")
            time_start = node_data.get("time_start", "")
            return f"{title} at {location} on {time_start} {description}".strip()

        elif node_type == "Fact":
            subject_id = node_data.get("subject", "")
            predicate = node_data.get("predicate", "")
            object_val = node_data.get("object", "")
            value = node_data.get("value", "")
            return f"{subject_id} {predicate} {object_val or value}".strip()

        elif node_type == "Episode":
            summary = node_data.get("summary", "")
            week_start = node_data.get("week_start", "")
            participant_ids = node_data.get("participant_ids", [])
            return f"Episode {week_start}: {summary} with {len(participant_ids)} participants".strip()

        else:  # Entity or other types
            name = node_data.get("name") or node_data.get("canonical_name") or node_data.get("title", "")
            description = node_data.get("description", "")
            return f"{name} {description}".strip()

    def get_community_members_text(
        self,
        community: Community,
        graph_store: Any
    ) -> List[str]:
        """Get text representations for all members in a community

        Useful for debugging and analysis of what each community contains.
        """
        member_texts = []

        for member_id in community.member_ids:
            node_data = graph_store.get_node(member_id)
            if not node_data:
                continue

            node_type = node_data.get("node_type", "Entity")
            text_rep = self._get_node_text_representation(node_data, node_type)

            if text_rep:
                member_texts.append(text_rep)

        return member_texts
