from __future__ import annotations

import os
import pickle
import json
from typing import Dict, Iterable, List, Tuple, Optional, Set, Union, Any
from datetime import datetime

import networkx as nx

from .schema import (
    BaseNode, NodeTypeEnum, Concept, Relation, Person, Event, Fact, Claim
)

class GraphStore:
    def __init__(self) -> None:
        self._g = nx.MultiDiGraph()
        self._changelog: List[dict] = []
        # Performance indexes for O(1) lookups
        self._name_to_id_index: Dict[str, List[str]] = {}  # name -> [entity_ids] (multiple entities can have same name)
        self._id_to_node_index: Dict[str, Any] = {}  # node_id -> node_data (cached node data)
        self._index_dirty = True  # Flag to rebuild indexes when needed

    @property
    def node_count(self) -> int:
        """Return the number of nodes in the graph"""
        return self._g.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Return the number of edges in the graph"""
        return self._g.number_of_edges()

    def add_concepts(self, concepts: Iterable[Concept]) -> None:
        """Add concepts to the graph
        
        Note: This method replaces add_entities(). Consider migrating to add_nodes().
        """
        for c in concepts:
            concept_data = c.model_dump()
            concept_data["node_type"] = NodeTypeEnum.CONCEPT.value
            self._g.add_node(c.id, **concept_data)
            self._update_indexes_on_add(c.id, c.name, concept_data)
            self._changelog.append({"op": "upsert_concept", "id": c.id})

    def add_relations(self, relations: Iterable[Relation]) -> None:
        for r in relations:
            payload = r.model_dump()
            existing = self._g.get_edge_data(r.head_id, r.tail_id, default={})
            if any(attrs == payload for attrs in existing.values()):
                continue
            self._g.add_edge(r.head_id, r.tail_id, **payload)
            self._changelog.append({
                "op": "upsert_relation",
                "head": r.head_id,
                "tail": r.tail_id,
                "rel_type": r.rel_type,
                "version": getattr(r, "version", 1),
            })

    def get_concept(self, concept_id: str) -> Concept | None:
        """Get a Concept node by ID"""
        if concept_id in self._g:
            data = self._g.nodes[concept_id]
            try:
                # Add id field (node_id is the key, not in data)
                data_with_id = {**data, "id": concept_id}
                node_type = data_with_id.get("node_type", "Concept")
                
                # Only handle Concept nodes
                if node_type in ("Concept", "concept"):
                    # Ensure required Concept fields
                    if "name" not in data_with_id:
                        data_with_id["name"] = data_with_id.get("id", "").replace("c_", "").replace("concept_", "")
                    return Concept(**data_with_id)
                
                # For backward compatibility, try to convert other node types to Concept
                # This is a temporary bridge during migration
                if node_type in ("Person", "person"):
                    # Convert Person to Concept representation
                    name = data_with_id.get("canonical_name", data_with_id.get("name", concept_id))
                    return Concept(
                        id=concept_id,
                        name=name.lower().replace(" ", "_"),
                        aliases=data_with_id.get("aliases", []),
                        description=f"Person: {name}",
                        content=data_with_id.get("content", "")
                    )
                
                elif node_type == "Event":
                    # Convert Event to Concept representation
                    name = data_with_id.get("title", concept_id)
                    return Concept(
                        id=concept_id,
                        name=name.lower().replace(" ", "_"),
                        description=f"Event: {name}",
                        content=data_with_id.get("content", "")
                    )
                
                # Default: try to create Concept from any node
                name = data_with_id.get("name", data_with_id.get("title", concept_id))
                return Concept(
                    id=concept_id,
                    name=str(name).lower().replace(" ", "_"),
                    description=data_with_id.get("description", ""),
                    content=data_with_id.get("content", "")
                )
            except Exception as e:
                # Debug: print error in development
                import os
                if os.getenv("DEBUG", "").lower() in ("1", "true", "yes"):
                    print(f"[GraphStore] Error parsing Concept {concept_id}: {e}")
                return None

        # Fallback: Try to get from index cache if not in graph.
        if concept_id in self._id_to_node_index:
            try:
                cached_data = self._id_to_node_index[concept_id]
                if isinstance(cached_data, dict):
                    data_with_id = {**cached_data, "id": concept_id}
                    node_type = data_with_id.get("node_type", "Entity")

                    # Debug: print processing info
                    if os.getenv("DEBUG", "").lower() in ("1", "true", "yes"):
                        print(f"[GraphStore] Processing cached {node_type} for {concept_id}")

                    # Handle different node types appropriately
                    if node_type == "Person":
                        # Handle missing 'name' field for Person entities
                        if "name" not in data_with_id and "canonical_name" in data_with_id:
                            data_with_id["name"] = data_with_id["canonical_name"]
                        # Set required Entity fields
                        data_with_id["type"] = "world"
                        data_with_id["subtype"] = "person"

                        # Generate description from persona traits and aliases
                        description_parts = []
                        if "persona_traits" in data_with_id and data_with_id["persona_traits"]:
                            traits_str = ", ".join(data_with_id["persona_traits"])
                            description_parts.append(f"Traits: {traits_str}")
                        if "aliases" in data_with_id and data_with_id["aliases"]:
                            aliases_str = ", ".join(data_with_id["aliases"])
                            description_parts.append(f"Also known as: {aliases_str}")

                        data_with_id["description"] = "; ".join(description_parts) if description_parts else "A person in the knowledge graph"

                        # Remove Person-specific fields that Entity doesn't have
                        for key in ["canonical_name", "aliases", "persona_traits"]:
                            data_with_id.pop(key, None)
                        return Concept(**data_with_id)

                    elif node_type == "Event":
                        # For events, create a pseudo-entity representation
                        event_title = data_with_id.get("title", concept_id)
                        data_with_id["name"] = event_title
                        data_with_id["type"] = "world"
                        data_with_id["subtype"] = "event"

                        # Generate comprehensive description from event details
                        description_parts = []
                        
                        # Always include event title as primary description
                        description_parts.append(f"Event: {event_title}")

                        # Add time information
                        if "time_start" in data_with_id and data_with_id["time_start"]:
                            description_parts.append(f"Time: {data_with_id['time_start']}")

                        # Add location information
                        if "location" in data_with_id and data_with_id["location"]:
                            description_parts.append(f"Location: {data_with_id['location']}")

                        # Add participants information
                        if "participants" in data_with_id and data_with_id["participants"]:
                            participants_str = ", ".join(data_with_id["participants"][:3])  # Limit to first 3
                            if len(data_with_id["participants"]) > 3:
                                participants_str += f" and {len(data_with_id['participants']) - 3} others"
                            description_parts.append(f"Participants: {participants_str}")

                        data_with_id["description"] = "; ".join(description_parts)

                        # Remove Event-specific fields
                        for key in ["title", "participants", "location", "time_start", "time_end", "granularity", "time_confidence", "temporal_conflict"]:
                            data_with_id.pop(key, None)
                        return Concept(**data_with_id)

                    elif node_type == "Fact":
                        # For facts, create a pseudo-entity representation
                        data_with_id["name"] = f"{data_with_id.get('predicate', 'fact')} {data_with_id.get('object', '')}"
                        data_with_id["type"] = "world"
                        data_with_id["subtype"] = "fact"

                        # Generate description from fact details
                        subject = data_with_id.get('subject', 'unknown')
                        predicate = data_with_id.get('predicate', 'is')
                        obj = data_with_id.get('object', 'unknown')
                        confidence = data_with_id.get('confidence', 1.0)
                        data_with_id["description"] = f"A fact stating that {subject} {predicate} {obj} (confidence: {confidence})"

                        # Remove Fact-specific fields
                        for key in ["subject", "predicate", "object", "value", "valid_from", "valid_to", "confidence", "supersedes"]:
                            data_with_id.pop(key, None)
                        return Concept(**data_with_id)

                    elif node_type == "Preference":
                        # For preferences, create a pseudo-entity representation
                        data_with_id["name"] = f"{data_with_id.get('entity', 'someone')} {data_with_id.get('preference_type', 'prefers')} {data_with_id.get('aspect', '')}"
                        data_with_id["type"] = "world"
                        data_with_id["subtype"] = "preference"

                        # Generate description from preference details
                        entity = data_with_id.get('entity', 'someone')
                        preference_type = data_with_id.get('preference_type', 'prefers')
                        aspect = data_with_id.get('aspect', '')
                        value = data_with_id.get('value', '')
                        data_with_id["description"] = f"A preference indicating that {entity} {preference_type} {value} regarding {aspect}"

                        # Remove Preference-specific fields
                        for key in ["entity", "preference_type", "aspect", "value"]:
                            data_with_id.pop(key, None)
                        return Concept(**data_with_id)

                    else:
                        # Default Entity handling
                        if "type" not in data_with_id:
                            data_with_id["type"] = "world"
                        if "subtype" not in data_with_id:
                            data_with_id["subtype"] = "entity"
                        return Concept(**data_with_id)
            except Exception as e:
                if os.getenv("DEBUG", "").lower() in ("1", "true", "yes"):
                    print(f"[GraphStore] Error parsing cached Concept {concept_id}: {e}")

        return None
    
    def _update_indexes_on_add(self, node_id: str, name: str, node_data: Dict[str, Any]) -> None:
        """Update indexes when a node is added"""
        # Update name_to_id index
        if name not in self._name_to_id_index:
            self._name_to_id_index[name] = []
        if node_id not in self._name_to_id_index[name]:
            self._name_to_id_index[name].append(node_id)
        
        # Update id_to_node index
        self._id_to_node_index[node_id] = node_data
    
    def _update_indexes_on_remove(self, node_id: str) -> None:
        """Update indexes when a node is removed"""
        # Remove from id_to_node index
        if node_id in self._id_to_node_index:
            node_data = self._id_to_node_index.pop(node_id)
            
            # Remove from name_to_id index
            # Need to find the name(s) associated with this node
            node_type = node_data.get("node_type", "unknown") if node_data else "unknown"
            name = None
            
            if node_type == "Person":
                name = node_data.get("canonical_name") if node_data else None
            elif node_type in ("Entity", "Concept", "concept"):
                name = node_data.get("name") if node_data else None
            elif node_type == "Event":
                name = node_data.get("title") if node_data else None
            elif node_type == "Preference":
                name = node_data.get("subject") if node_data else None
            else:
                name = node_data.get("name") or node_data.get("canonical_name") if node_data else None
            
            if name and name in self._name_to_id_index:
                if node_id in self._name_to_id_index[name]:
                    self._name_to_id_index[name].remove(node_id)
                # Clean up empty lists
                if not self._name_to_id_index[name]:
                    del self._name_to_id_index[name]
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges from the graph
        
        Args:
            node_id: ID of the node to remove
        """
        if node_id not in self._g:
            return
        
        # Remove all edges connected to this node
        # NetworkX will automatically remove edges when node is removed,
        # but we need to update changelog first
        in_edges = list(self._g.in_edges(node_id, data=True))
        out_edges = list(self._g.out_edges(node_id, data=True))
        
        # Log edge removals
        for src, _, data in in_edges:
            self._changelog.append({
                "op": "remove_relation",
                "head": src,
                "tail": node_id,
                "rel_type": data.get("rel_type", "unknown")
            })
        
        for _, dst, data in out_edges:
            self._changelog.append({
                "op": "remove_relation",
                "head": node_id,
                "tail": dst,
                "rel_type": data.get("rel_type", "unknown")
            })
        
        # Remove node from graph
        self._g.remove_node(node_id)
        
        # Update indexes
        self._update_indexes_on_remove(node_id)
        
        # Log node removal
        self._changelog.append({
            "op": "remove_node",
            "id": node_id
        })
    
    def _rebuild_indexes(self) -> None:
        """Rebuild all indexes from graph (call when loading from disk)"""
        self._name_to_id_index.clear()
        self._id_to_node_index.clear()
        
        for node_id, data in self._g.nodes(data=True):
            if not data:
                continue
            
            # Cache node data
            self._id_to_node_index[node_id] = data
            
            # Index by name based on node type
            node_type = data.get("node_type", "unknown")
            name = None
            
            if node_type == "Person":
                name = data.get("canonical_name")
            elif node_type == "Entity":
                name = data.get("name")
            elif node_type == "Event":
                name = data.get("title")  # Event nodes use title as name
            elif node_type == "Preference":
                name = data.get("subject")  # Preference nodes may use subject as name
            elif node_type == "Fact":
                # Facts don't have a single name, skip indexing by name
                pass
            else:
                # Fallback: try name or canonical_name
                name = data.get("name") or data.get("canonical_name")
            
            if name:
                if name not in self._name_to_id_index:
                    self._name_to_id_index[name] = []
                if node_id not in self._name_to_id_index[name]:
                    self._name_to_id_index[name].append(node_id)
        
        self._index_dirty = False
    
    def get_entity_by_name(self, name: str) -> Concept | None:
        """通过实体名称查找实体（返回第一个匹配的）- 使用索引优化"""
        if self._index_dirty:
            self._rebuild_indexes()
        
        # Use index for O(1) lookup
        if name in self._name_to_id_index:
            entity_ids = self._name_to_id_index[name]
            if entity_ids:
                # Return first matching entity
                entity_id = entity_ids[0]
                if entity_id in self._id_to_node_index:
                    data = self._id_to_node_index[entity_id]
                    try:
                        data_with_id = {**data, "id": entity_id}
                        return Concept(**data_with_id)
                    except Exception:
                        pass
        
        # Fallback to old method (shouldn't happen if indexes are correct)
        return None
    
    def get_entities_by_name(self, name: str) -> List[Concept]:
        """通过实体名称查找所有匹配的实体 - 使用索引优化"""
        if self._index_dirty:
            self._rebuild_indexes()

        results: List[Concept] = []
        # Use index for O(1) lookup
        if name in self._name_to_id_index:
            for entity_id in self._name_to_id_index[name]:
                if entity_id in self._id_to_node_index:
                    data = self._id_to_node_index[entity_id]
                    # 使用与iter_entities相同的逻辑创建Entity对象
                    node_type = data.get("node_type", "unknown")
                    try:
                        if node_type == "Person":
                            # Person节点使用canonical_name作为name
                            entity_data = data.copy()
                            entity_data["id"] = entity_id
                            entity_data["name"] = data.get("canonical_name", entity_id)
                            entity_data["type"] = "world"
                            entity_data["subtype"] = "person"

                            # Generate description from persona traits and aliases
                            description_parts = []
                            if "persona_traits" in entity_data and entity_data["persona_traits"]:
                                traits_str = ", ".join(entity_data["persona_traits"])
                                description_parts.append(f"Traits: {traits_str}")
                            if "aliases" in entity_data and entity_data["aliases"]:
                                aliases_str = ", ".join(entity_data["aliases"])
                                description_parts.append(f"Also known as: {aliases_str}")

                            entity_data["description"] = "; ".join(description_parts) if description_parts else "A person in the knowledge graph"

                            # 移除Person特有的字段
                            for key in ["canonical_name", "aliases", "persona_traits"]:
                                entity_data.pop(key, None)
                            results.append(Concept(**entity_data))
                        elif node_type in ("Concept", "concept", "Entity", "entity"):
                            # Entity节点直接使用
                            entity_data = data.copy()
                            entity_data["id"] = entity_id
                            results.append(Concept(**entity_data))
                        elif node_type == "Event":
                            # Event节点使用title作为name
                            entity_data = data.copy()
                            entity_data["id"] = entity_id
                            event_title = data.get("title", entity_id)
                            entity_data["name"] = event_title
                            entity_data["type"] = "world"
                            entity_data["subtype"] = "event"

                            # Generate comprehensive description from event details
                            description_parts = []
                            
                            # Always include event title as primary description
                            description_parts.append(f"Event: {event_title}")

                            # Add time information
                            if "time_start" in entity_data and entity_data["time_start"]:
                                description_parts.append(f"Time: {entity_data['time_start']}")

                            # Add location information
                            if "location" in entity_data and entity_data["location"]:
                                description_parts.append(f"Location: {entity_data['location']}")

                            # Add participants information
                            if "participants" in entity_data and entity_data["participants"]:
                                participants_str = ", ".join(entity_data["participants"][:3])  # Limit to first 3
                                if len(entity_data["participants"]) > 3:
                                    participants_str += f" and {len(entity_data['participants']) - 3} others"
                                description_parts.append(f"Participants: {participants_str}")

                            entity_data["description"] = "; ".join(description_parts)

                            # 移除Event特有的字段
                            for key in ["title", "participants", "location", "time_start", "time_end", "granularity", "time_confidence", "temporal_conflict"]:
                                entity_data.pop(key, None)
                            results.append(Concept(**entity_data))
                        elif node_type == "Preference":
                            # Preference节点使用subject作为name（如果有的话）
                            if "subject" in data:
                                entity_data = data.copy()
                                entity_data["id"] = entity_id
                                entity_data["name"] = str(data.get("subject", entity_id))
                                entity_data["type"] = "world"
                                entity_data["subtype"] = "preference"

                                # Generate description from preference details
                                entity = data.get('entity', 'someone')
                                preference_type = data.get('preference_type', 'prefers')
                                aspect = data.get('aspect', '')
                                value = data.get('value', '')
                                entity_data["description"] = f"A preference indicating that {entity} {preference_type} {value} regarding {aspect}"

                                # 移除Preference特有的字段
                                for key in ["entity", "preference_type", "aspect", "value", "subject", "predicate", "object", "confidence", "evidence_span"]:
                                    entity_data.pop(key, None)
                                results.append(Concept(**entity_data))
                        # Fact和其他类型跳过
                    except Exception:
                        pass
        return results

    def iter_entities(self) -> Iterable[Concept]:
        """迭代所有实体，跳过无效节点"""
        for node_id, data in self._g.nodes(data=True):
            # 节点ID就是实体ID，检查数据是否存在
            if not data:
                continue

            node_type = data.get("node_type", "unknown")

            # 根据节点类型创建合适的Entity对象
            try:
                if node_type == "Person":
                    # Person节点使用canonical_name作为name
                    entity_data = data.copy()
                    entity_data["id"] = node_id  # 确保id字段存在
                    entity_data["name"] = data.get("canonical_name", node_id)
                    entity_data["type"] = "world"  # Person作为world类型的实体
                    entity_data["subtype"] = "person"

                    # Generate comprehensive description from all available info
                    description_parts = []

                    # Add persona traits
                    if "persona_traits" in entity_data and entity_data["persona_traits"]:
                        traits_str = ", ".join(entity_data["persona_traits"])
                        description_parts.append(f"Traits: {traits_str}")

                    # Add aliases
                    if "aliases" in entity_data and entity_data["aliases"]:
                        aliases_str = ", ".join(entity_data["aliases"])
                        description_parts.append(f"Also known as: {aliases_str}")

                    # Add role, profession, or other descriptive fields
                    descriptive_fields = ["role", "profession", "occupation", "background", "identity"]
                    for field in descriptive_fields:
                        if field in entity_data and entity_data[field]:
                            description_parts.append(f"{field.title()}: {entity_data[field]}")

                    # If no specific description available, create a meaningful generic description
                    if not description_parts:
                        person_name = entity_data.get("name", "This person")
                        description_parts.append(f"{person_name} is a person mentioned in conversations")

                    entity_data["description"] = "; ".join(description_parts)

                    # 移除Person特有的字段，避免Entity验证失败
                    for key in ["canonical_name", "aliases", "persona_traits", "role", "profession", "occupation", "background", "identity"]:
                        entity_data.pop(key, None)
                    yield Concept(**entity_data)
                elif node_type == "Entity":
                    # Entity节点直接使用，确保id字段存在
                    entity_data = data.copy()
                    entity_data["id"] = node_id
                    yield Concept(**entity_data)
                elif node_type == "Event":
                    # Event节点使用title作为name
                    entity_data = data.copy()
                    entity_data["id"] = node_id
                    event_title = data.get("title", node_id)
                    entity_data["name"] = event_title
                    entity_data["type"] = "world"
                    entity_data["subtype"] = "event"

                    # Generate comprehensive description from event details
                    description_parts = []
                    
                    # Always include event title as primary description
                    description_parts.append(f"Event: {event_title}")

                    # Add time information
                    if "time_start" in entity_data and entity_data["time_start"]:
                        description_parts.append(f"Time: {entity_data['time_start']}")

                    # Add location information
                    if "location" in entity_data and entity_data["location"]:
                        description_parts.append(f"Location: {entity_data['location']}")

                    # Add participants information
                    if "participants" in entity_data and entity_data["participants"]:
                        participants_str = ", ".join(entity_data["participants"][:3])  # Limit to first 3
                        if len(entity_data["participants"]) > 3:
                            participants_str += f" and {len(entity_data['participants']) - 3} others"
                        description_parts.append(f"Participants: {participants_str}")

                    entity_data["description"] = "; ".join(description_parts)

                    # 移除Event特有的字段
                    for key in ["title", "participants", "location", "time_start", "time_end", "granularity", "time_confidence", "temporal_conflict"]:
                        entity_data.pop(key, None)
                    yield Concept(**entity_data)
                elif node_type == "Fact":
                    # Fact节点跳过（不是实体）
                    continue
                elif node_type == "Preference":
                    # Preference节点使用subject作为name（如果有的话）
                    if "subject" in data:
                        entity_data = data.copy()
                        entity_data["id"] = node_id
                        entity_data["name"] = str(data.get("subject", node_id))
                        entity_data["type"] = "world"
                        entity_data["subtype"] = "preference"
                        # 移除Preference特有的字段
                        for key in ["subject", "predicate", "object", "confidence", "evidence_span"]:
                            entity_data.pop(key, None)
                        yield Concept(**entity_data)
                # 其他类型跳过
            except Exception as e:
                # 跳过无法解析的节点
                continue

    # --- 查询工具 ---
    def neighbors(self, entity_id: str, rel_type: Optional[str] = None, direction: str = "out", min_weight: Optional[float] = None, min_confidence: Optional[float] = None) -> List[Concept]:
        """获取邻居实体，支持按关系类型、权重、置信度过滤"""
        results: List[Concept] = []
        if direction == "out":
            edges = self._g.out_edges(entity_id, keys=True, data=True)
        elif direction == "in":
            edges = self._g.in_edges(entity_id, keys=True, data=True)
        else:
            edges = list(self._g.out_edges(entity_id, keys=True, data=True)) + list(self._g.in_edges(entity_id, keys=True, data=True))
        for u, v, k, data in edges:
            if rel_type and data.get("rel_type") != rel_type:
                continue
            if min_weight is not None and data.get("weight", 0.0) < min_weight:
                continue
            if min_confidence is not None and data.get("confidence", 0.0) < min_confidence:
                continue
            nid = v if u == entity_id else u
            if nid in self._g:
                results.append(Concept(**self._g.nodes[nid]))
        return results

    def k_hop_subgraph(self, start_ids: List[str], k: int = 2) -> nx.MultiDiGraph:
        visited: Set[str] = set(start_ids)
        frontier: Set[str] = set(start_ids)
        for _ in range(k):
            next_frontier: Set[str] = set()
            for nid in frontier:
                next_frontier.update([v for _, v in self._g.out_edges(nid)])
                next_frontier.update([u for u, _ in self._g.in_edges(nid)])
            frontier = next_frontier - visited
            visited |= frontier
        return self._g.subgraph(visited).copy()

    # --- 导出 ---
    def export_graphml(self, path: str) -> None:
        """导出为 GraphML 格式（用于可视化工具如 Gephi）
        
        注意：GraphML 不支持列表类型，会自动转换为 JSON 字符串
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        # 创建一个副本用于导出，过滤掉 GraphML 不支持的数据类型
        import json
        export_g = nx.MultiDiGraph()
        
        # 复制节点，将列表类型转换为 JSON 字符串
        for node_id, attrs in self._g.nodes(data=True):
            export_attrs = {}
            for key, value in attrs.items():
                if isinstance(value, (list, dict)):
                    export_attrs[key] = json.dumps(value, ensure_ascii=False)
                elif value is not None:
                    export_attrs[key] = value
            export_g.add_node(node_id, **export_attrs)
        
        # 复制边，将列表类型转换为 JSON 字符串
        for u, v, key, attrs in self._g.edges(keys=True, data=True):
            export_attrs = {}
            for attr_key, attr_value in attrs.items():
                if isinstance(attr_value, (list, dict)):
                    export_attrs[attr_key] = json.dumps(attr_value, ensure_ascii=False)
                elif attr_value is not None:
                    export_attrs[attr_key] = attr_value
            export_g.add_edge(u, v, key=key, **export_attrs)
        
        # 导出为无向图（GraphML 通常用于可视化）
        nx.write_graphml(export_g.to_undirected(), path)

    def export_json(self, path: str, indent: int = 2) -> None:
        """Export graph to readable JSON format with statistics
        
        Args:
            path: Output file path
            indent: JSON indentation (default 2 for readability)
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        # Collect entity statistics
        entity_types = {}
        relations_by_type = {}
        
        nodes = []
        for nid, attrs in self._g.nodes(data=True):
            if attrs:
                node_data = {"id": nid, **attrs}
                nodes.append(node_data)
                
                # Count node types: prioritize node_type field, fallback to type field
                # node_type is set for Person, Event, Fact, Preference, PersonFacet nodes
                # type is used for Entity nodes (preference/world)
                node_type = attrs.get("node_type")
                if node_type:
                    # Use node_type for Person, Event, Fact, Preference, PersonFacet
                    entity_types[node_type] = entity_types.get(node_type, 0) + 1
                else:
                    # Fallback to type for Entity nodes
                    entity_type = attrs.get("type", "unknown")
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        edges = []
        for u, v, attrs in self._g.edges(data=True):
            edge_data = {"source": u, "target": v, **attrs}
            edges.append(edge_data)
            
            # Count relation types
            rel_type = attrs.get("rel_type", "unknown")
            relations_by_type[rel_type] = relations_by_type.get(rel_type, 0) + 1
        
        data = {
            "metadata": {
                "total_entities": len(nodes),
                "total_relations": len(edges),
                "entity_types": entity_types,  # For backward compatibility (includes all node types)
                "node_types": entity_types,     # More accurate name (Person, Event, Fact, Preference, Entity, etc.)
                "relation_types": relations_by_type,
            },
            "nodes": nodes,
            "edges": edges,
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

    def save(self, path: str, changelog_path: Optional[str] = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._g, f)
        if changelog_path:
            with open(changelog_path, "wb") as f:
                pickle.dump(self._changelog, f)

    def load(self, path: str, changelog_path: Optional[str] = None) -> None:
        candidate = GraphStore()
        if os.path.exists(path):
            with open(path, "rb") as f:
                graph = pickle.load(f)
            if not isinstance(graph, nx.MultiDiGraph):
                raise ValueError('Invalid graph snapshot')
            candidate._g = graph
            if changelog_path and os.path.exists(changelog_path):
                with open(changelog_path, "rb") as f:
                    log = pickle.load(f)
                if not isinstance(log, list):
                    raise ValueError('Invalid graph changelog')
                candidate._changelog = log
        elif changelog_path and os.path.exists(changelog_path):
            raise ValueError('Graph changelog cannot be loaded without its graph')
        candidate._rebuild_indexes()
        self.__dict__.update(candidate.__dict__)

    def num_entities(self) -> int:
        return self._g.number_of_nodes()

    def num_relations(self) -> int:
        return self._g.number_of_edges()
    
    # --- Multi-type node support ---
    
    def add_node(self, node: BaseNode) -> None:
        """Add any type of node to the graph with type-safe NodeType metadata

        Args:
            node: Node object that inherits from BaseNode or is a recognized node type
        """
        node_data = node.model_dump()
        node_id = node_data.pop("id")

        # Determine node type from the actual class or from BaseNode inheritance
        node_type_class = type(node).__name__

        # Use NodeTypeEnum for standardization (fallback to class name if not found)
        try:
            # For Concept nodes, we need to check the actual subtype
            if isinstance(node, Concept):
                node_type = NodeTypeEnum.CONCEPT.value
            elif isinstance(node, Person):
                node_type = NodeTypeEnum.PERSON.value
            elif isinstance(node, Event):
                node_type = NodeTypeEnum.EVENT.value
            elif isinstance(node, Fact):
                node_type = NodeTypeEnum.FACT.value
            else:
                # Fallback: try to map class name to NodeTypeEnum
                node_type = getattr(NodeTypeEnum, node_type_class.upper(), NodeTypeEnum.CONCEPT).value
        except AttributeError:
            node_type = node_type_class.lower()

        node_data["node_type"] = node_type
        self._g.add_node(node_id, **node_data)

        # Update indexes based on node type
        if isinstance(node, Person) and hasattr(node, "canonical_name"):
            self._update_indexes_on_add(node_id, node.canonical_name, node_data)
        elif isinstance(node, Concept) and hasattr(node, "name"):
            self._update_indexes_on_add(node_id, node.name, node_data)
        elif isinstance(node, Event) and hasattr(node, "title"):
            # Event nodes use title as name for indexing
            self._update_indexes_on_add(node_id, node.title, node_data)
        else:
            # For other node types (Fact, etc.), just cache the node data
            self._id_to_node_index[node_id] = node_data

        self._changelog.append({"op": f"upsert_{node_type}", "id": node_id})
    
    def add_nodes(self, nodes: Iterable[BaseNode]) -> None:
        """Add multiple nodes of any type"""
        for node in nodes:
            self.add_node(node)
    
    def get_node(
        self, node_id: str, node_type: Optional[Union[NodeTypeEnum, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get node by ID, optionally filtering by NodeType enum
        
        ROOT FIX: 确保返回的数据总是包含 node_type 字段，如果没有则从节点ID推断
        这保证了数据模型的一致性，避免了字段命名不一致的问题
        """
        if node_id not in self._g:
            return None
        data = self._g.nodes[node_id]
        
        # ROOT FIX: 确保 node_type 字段存在，如果没有则从ID推断
        node_data = dict(data)
        if "node_type" not in node_data or not node_data.get("node_type"):
            # 从节点ID前缀推断类型（优雅降级）
            if node_id.startswith("p_") or node_id.startswith("person_"):
                node_data["node_type"] = NodeTypeEnum.PERSON.value
            elif node_id.startswith("c_") or node_id.startswith("concept_"):
                node_data["node_type"] = NodeTypeEnum.CONCEPT.value
            elif node_id.startswith("event_"):
                node_data["node_type"] = NodeTypeEnum.EVENT.value
            elif node_id.startswith("fact_"):
                node_data["node_type"] = NodeTypeEnum.FACT.value
            elif node_id.startswith("claim_"):
                node_data["node_type"] = NodeTypeEnum.CLAIM.value
            else:
                # 如果无法推断，使用已有的值或默认值
                node_data["node_type"] = node_data.get("node_type", "unknown")
        
        # 确保 id 字段存在（节点ID是图的key，不在数据中）
        if "id" not in node_data:
            node_data["id"] = node_id
        
        if node_type:
            expected_type = node_type.value if isinstance(node_type, NodeTypeEnum) else node_type
            if node_data.get("node_type") != expected_type:
                return None
        return node_data
    
    def get_person(self, person_id: str) -> Optional[Person]:
        """Get Person node"""
        data = self.get_node(person_id)
        if data:
            try:
                # Add id field (node_id is the key, not in data)
                data_with_id = {**data, "id": person_id}
                return Person(**data_with_id)
            except Exception as e:
                # Debug: print error in development
                import os
                if os.getenv("DEBUG", "").lower() in ("1", "true", "yes"):
                    print(f"[GraphStore] Error parsing Person {person_id}: {e}")
                pass
        return None
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """Get Event node"""
        data = self.get_node(event_id)
        if data:
            try:
                # Add id field (node_id is the key, not in data)
                data_with_id = {**data, "id": event_id}
                return Event(**data_with_id)
            except Exception:
                pass
        return None
    
    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Get Fact node"""
        data = self.get_node(fact_id)
        if data:
            try:
                # Add id field (node_id is the key, not in data)
                data_with_id = {**data, "id": fact_id}
                return Fact(**data_with_id)
            except Exception:
                pass
        return None
    
    def get_edges_by_type(self, rel_type: str) -> List[Dict[str, Any]]:
        """Get all edges of a specific relation type
        
        Args:
            rel_type: Relation type to filter (e.g., 'CLAIMS', 'INVOLVED_IN')
            
        Returns:
            List of edge dictionaries with 'head_id', 'tail_id', and edge data
        """
        edges = []
        for u, v, key, data in self._g.edges(keys=True, data=True):
            if data.get('rel_type') == rel_type:
                edges.append({
                    'head_id': u,
                    'tail_id': v,
                    'key': key,
                    **data
                })
        return edges
    
    # --- Time-aware queries ---
    
    def get_facts_by_validity(
        self, query_time: Optional[str] = None, subject: Optional[str] = None,
        predicate: Optional[str] = None
    ) -> List[Fact]:
        """Facts whose declared validity overlaps the requested calendar period.

        Unknown validity is not proof of validity. A single bound certifies its
        stated precision only, consistently with the fact lookup operator.
        """
        from ..utils.temporal import date_bounds
        requested = date_bounds(query_time if query_time is not None else datetime.now().isoformat())
        if not requested:
            raise ValueError('query_time must be a valid calendar period or ISO timestamp')
        facts = []
        for node_id, data in self._g.nodes(data=True):
            if data.get('node_type') != NodeTypeEnum.FACT.value or data.get('is_archived'):
                continue
            fact = Fact(**{**data, 'id': node_id})
            if subject and fact.subject_id != subject:
                continue
            if predicate and fact.predicate != predicate:
                continue
            start, end = date_bounds(fact.valid_from), date_bounds(fact.valid_to)
            start, end = start or end, end or start
            if (start and end and start[0] <= end[1]
                    and start[0] <= requested[1] and requested[0] <= end[1]):
                facts.append(fact)
        return facts

    def get_events_by_time_range(
        self, start_time: Optional[str] = None, end_time: Optional[str] = None,
        entity_id: Optional[str] = None
    ) -> List[Event]:
        """Calendar-overlap query over occurrence time; unknown time fails closed."""
        from ..utils.temporal import date_bounds
        lower, upper = date_bounds(start_time), date_bounds(end_time)
        if ((start_time is not None and lower is None)
                or (end_time is not None and upper is None)):
            raise ValueError('Time bounds must be valid calendar periods or ISO timestamps')
        if lower and upper and lower[0] > upper[1]:
            raise ValueError('Time range start exceeds its end')
        events = []
        for node_id, data in self._g.nodes(data=True):
            if data.get('node_type') != NodeTypeEnum.EVENT.value or data.get('is_archived'):
                continue
            event = Event(**{**data, 'id': node_id})
            if entity_id and entity_id not in event.participants:
                continue
            if lower or upper:
                start = date_bounds(event.time_start) or date_bounds(event.time_resolved)
                end = date_bounds(event.time_end) if event.time_end else start
                if not start or not end or start[0] > end[1]:
                    continue
                if (lower and end[1] < lower[0]) or (upper and start[0] > upper[1]):
                    continue
            events.append(event)
        events.sort(key=lambda event: (event.time_start or event.time_resolved or '', event.id))
        return events

    # --- SUPERSEDES chain traversal ---
    
    def get_supersedes_chain(self, fact_id: str) -> List[str]:
        """This fact and all explicit superseding predecessors, without dropping forks."""
        if fact_id not in self._g:
            return []
        from collections import deque
        pending, visited, chain = deque([fact_id]), set(), []
        while pending:
            current = pending.popleft()
            if current in visited:
                continue
            visited.add(current)
            chain.append(current)
            pending.extend(sorted({source for source, _, data in self._g.in_edges(current, data=True)
                if data.get('rel_type') == 'SUPERSEDES' and source not in visited}))
        return chain

    def get_superseded_by(self, fact_id: str) -> List[str]:
        """Get facts that are superseded by this fact (follow SUPERSEDES edges forwards)"""
        superseded = []
        for _, target, _, data in self._g.out_edges(fact_id, keys=True, data=True):
            if data.get("rel_type") == "SUPERSEDES":
                superseded.append(target)
        return superseded
    
    def get_latest_fact(self, subject: str, predicate: str) -> Optional[Fact]:
        """Return the unique active graph head; unresolved forks have no latest fact.

        This is version topology, not current-time validity or a recency guess.
        SUPERSEDES points from the replacement to the replaced node.
        """
        candidates = {node_id: data for node_id, data in self._g.nodes(data=True)
            if data.get('node_type') == NodeTypeEnum.FACT.value
            and data.get('subject_id') == subject and data.get('predicate') == predicate}
        versions = nx.DiGraph()
        versions.add_nodes_from(candidates)
        for node_id in candidates:
            for source, _, edge in self._g.in_edges(node_id, data=True):
                if edge.get('rel_type') != 'SUPERSEDES':
                    continue
                if source not in candidates:
                    return None  # Cross-proposition edges cannot certify a version head.
                versions.add_edge(source, node_id)
        if not nx.is_directed_acyclic_graph(versions):
            return None
        heads = [node_id for node_id in candidates
            if versions.in_degree(node_id) == 0 and not candidates[node_id].get('is_archived')]
        return Fact(**{**candidates[heads[0]], 'id': heads[0]}) if len(heads) == 1 else None

    # --- Hub penalty calculation ---
    
    def get_node_degree(self, node_id: str, direction: str = "total") -> int:
        """Get node degree (out, in, or total)"""
        if direction == "out":
            return self._g.out_degree(node_id)
        elif direction == "in":
            return self._g.in_degree(node_id)
        else:
            return self._g.degree(node_id)
    
    def get_hub_penalty(self, node_id: str, base_penalty: float = 0.1) -> float:
        """Calculate hub penalty: log(degree) * base_penalty"""
        import math
        degree = self.get_node_degree(node_id)
        if degree <= 1:
            return 0.0
        return math.log(degree) * base_penalty

    def get_facts_by_entity_and_time(
        self,
        subject_id: str,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None
    ) -> List[Fact]:
        """Get facts for an entity filtered by temporal validity (Phase 3)

        Args:
            subject_id: Entity ID (person_ or entity_ prefix)
            time_start: Optional start time in ISO-8601 format
            time_end: Optional end time in ISO-8601 format

        Returns:
            List of Fact objects that are valid within the time range
        """
        facts = []

        for node_id, data in self._g.nodes(data=True):
            if data.get("node_type") == NodeTypeEnum.FACT.value:
                try:
                    fact = Fact(**data)

                    # Check if fact is for the requested subject
                    if fact.subject != subject_id:
                        continue

                    # Check temporal validity if time range provided
                    if time_start or time_end:
                        if not fact.is_temporal:
                            continue

                        # Check if fact is valid during the time range
                        fact_start = fact.valid_from or "0000-01-01"
                        fact_end = fact.valid_to or "9999-12-31"

                        # Fact overlaps with query range if:
                        # fact_start <= time_end AND fact_end >= time_start
                        if time_start and fact_end < time_start:
                            continue
                        if time_end and fact_start > time_end:
                            continue

                    facts.append(fact)

                except Exception as e:
                    print(f"[GraphStore] Error parsing fact {node_id}: {e}")
                    continue

        return facts
