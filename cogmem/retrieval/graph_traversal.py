"""图遍历检索增强模块：通过多跳图遍历扩展检索实体

第二阶段优化：实现激活扩散（Spreading Activation）机制，通过能量传播
实现智能的多跳推理，而不是固定的max_depth遍历。
"""

from __future__ import annotations

import heapq
from typing import List, Optional, TYPE_CHECKING, Dict, Tuple, Set
from collections import defaultdict

if TYPE_CHECKING:
    from ..storage.graph_store import GraphStore
    from ..storage.schema import Entity
    from ..models.embedding import EmbeddingEncoder


class GraphTraversal:
    """图遍历检索增强"""

    def __init__(self, graph: GraphStore):
        """
        Args:
            graph: 知识图谱存储
        """
        self.graph = graph

    def expand_entities(
        self, 
        seed_entities: List[Entity], 
        k_hop: int = 2,
        relation_types: Optional[List[str]] = None,
        min_confidence: float = 0.5
    ) -> List[Entity]:
        """从种子实体出发，遍历 k-hop 邻居
        
        Args:
            seed_entities: 初始实体列表
            k_hop: 跳数（默认2）
            relation_types: 优先的关系类型（如 temporal, located_at），None表示不过滤
            min_confidence: 最小关系置信度（默认0.5）
            
        Returns:
            扩展后的实体列表（包括种子实体和邻居实体）
        """
        visited = {e.id for e in seed_entities}
        current_layer = seed_entities
        all_entities = list(seed_entities)

        for _ in range(k_hop):
            next_layer = []
            for entity in current_layer:
                # 获取邻居实体
                neighbors = self.graph.neighbors(
                    entity.id,
                    min_confidence=min_confidence
                )
                
                for nb in neighbors:
                    if nb.id not in visited:
                        # 如果指定了关系类型，检查关系类型
                        if relation_types:
                            # 获取连接的关系类型（通过检查边的属性）
                            # 这里简化处理：如果邻居被找到，说明有边连接
                            # 更精确的实现需要检查 edge 的 rel_type
                            pass  # TODO: 更精确的关系类型过滤
                        
                        next_layer.append(nb)
                        visited.add(nb.id)
                        all_entities.append(nb)
            
            current_layer = next_layer
            if not current_layer:
                break

        return all_entities

    def expand_with_relation_filter(
        self,
        seed_entities: List[Entity],
        relation_types: List[str],
        k_hop: int = 2,
        min_confidence: float = 0.5
    ) -> List[Entity]:
        """按关系类型过滤的图遍历
        
        例如：只扩展 temporal 和 located_at 关系
        
        Args:
            seed_entities: 初始实体列表
            relation_types: 允许的关系类型列表
            k_hop: 跳数
            min_confidence: 最小置信度
            
        Returns:
            扩展后的实体列表
        """
        visited = {e.id for e in seed_entities}
        current_layer = seed_entities
        all_entities = list(seed_entities)

        for _ in range(k_hop):
            next_layer = []
            for entity in current_layer:
                # 对每个关系类型分别查询
                for rel_type in relation_types:
                    neighbors = self.graph.neighbors(
                        entity.id,
                        rel_type=rel_type,
                        min_confidence=min_confidence
                    )
                    for nb in neighbors:
                        if nb.id not in visited:
                            next_layer.append(nb)
                            visited.add(nb.id)
                            all_entities.append(nb)
            
            current_layer = next_layer
            if not current_layer:
                break

        return all_entities

    def expand_with_importance_scoring(
        self,
        seed_entities: List[Entity],
        k_hop: int = 2,
        max_entities_per_hop: int = 20,
        min_confidence: float = 0.5
    ) -> List[Entity]:
        """基于重要性评分的智能图遍历"""
        if not seed_entities:
            return []

        visited = {e.id for e in seed_entities}
        current_layer = seed_entities
        all_entities = list(seed_entities)
        entity_distances = {e.id: 0 for e in seed_entities}

        for hop in range(k_hop):
            hop_candidates = []
            
            for entity in current_layer:
                neighbors = self.graph.neighbors(entity.id, min_confidence=min_confidence)
                
                for neighbor in neighbors:
                    if neighbor.id not in visited:
                        importance_score = self._compute_neighbor_importance(
                            neighbor, seed_entities, hop + 1
                        )
                        hop_candidates.append((neighbor, importance_score))
            
            hop_candidates.sort(key=lambda x: x[1], reverse=True)
            selected_neighbors = hop_candidates[:max_entities_per_hop]
            
            next_layer = []
            for neighbor, _ in selected_neighbors:
                if neighbor.id not in visited:
                    next_layer.append(neighbor)
                    visited.add(neighbor.id)
                    all_entities.append(neighbor)
                    entity_distances[neighbor.id] = hop + 1
            
            current_layer = next_layer
            if not current_layer:
                break
        
        all_entities.sort(key=lambda e: (
            entity_distances.get(e.id, 999),
            -len(e.description)
        ))
        
        return all_entities

    def _compute_neighbor_importance(self, neighbor: Entity, seed_entities: List[Entity], distance: int) -> float:
        """计算邻居实体的重要性分数"""
        score = 0.0
        distance_penalty = 1.0 / (distance + 1)
        score += distance_penalty * 0.3
        
        desc_richness = min(len(neighbor.description) / 100.0, 1.0)
        score += desc_richness * 0.2
        
        type_bonus = self._get_entity_type_bonus(neighbor.type)
        score += type_bonus * 0.2
        
        name_similarity = self._compute_name_similarity(neighbor, seed_entities)
        score += name_similarity * 0.3
        
        return score

    def _get_entity_type_bonus(self, entity_type: str) -> float:
        """根据实体类型给予奖励分数"""
        type_scores = {
            "person": 0.9, "temporal": 0.8, "location": 0.7, "event": 0.8,
            "organization": 0.6, "concept": 0.5, "world": 0.6, "preference": 0.4,
        }
        return type_scores.get(entity_type, 0.3)

    def _compute_name_similarity(self, entity: Entity, seed_entities: List[Entity]) -> float:
        """计算实体名称与种子实体的相似度"""
        max_similarity = 0.0
        entity_name_lower = entity.name.lower()
        
        for seed in seed_entities:
            seed_name_lower = seed.name.lower()
            
            if entity_name_lower in seed_name_lower or seed_name_lower in entity_name_lower:
                similarity = 0.8
            else:
                entity_words = set(entity_name_lower.split())
                seed_words = set(seed_name_lower.split())
                if entity_words and seed_words:
                    overlap = len(entity_words.intersection(seed_words))
                    similarity = overlap / max(len(entity_words), len(seed_words))
                else:
                    similarity = 0.0
            
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity


class SpreadingActivationTraversal:
    """激活扩散遍历：通过能量传播实现智能多跳推理

    核心思想：模拟认知科学中的激活扩散模型，节点间传播"能量"，
    能量值反映节点与查询的语义相关性。

    算法优势：
    - 不必预先设定固定的max_depth
    - 能量自然衰减，避免过度扩散
    - 语义相似度高的边传导更多能量
    - 边类型权重让不同关系有不同的传导率
    - 最终按能量排序，取Top-N节点作为证据
    """

    # 边类型权重配置（可调参数）
    EDGE_TYPE_WEIGHTS = {
        "INVOLVED_IN": 1.0,      # 事件参与（强关系）
        "ASSERTS": 0.9,           # 事实断言（强关系）
        "AT_TIME": 0.8,           # 时间关联（中等）
        "SUPPORTS_ANSWER": 0.7,   # 答案支持（中等）
        "SAME_AS": 0.9,           # 同指（强关系）
        "ALIAS_OF": 0.9,          # 别名（强关系）
        "BELONGS_TO": 0.6,        # 归属（较弱）
        "SPOKEN_BY": 0.5,         # 说话者（弱）
        "MENTIONS": 0.4,          # 提及（很弱）
        "related": 0.7,           # 通用相关（默认）
        "EVIDENCE_IN": 0.6,       # 证据位于（中等）
        "BEFORE": 0.7,            # 时间先后（中等）
        "AFTER": 0.7,             # 时间先后（中等）
        "CONTRADICTS": 0.8,       # 矛盾（强，因为矛盾也是重要关系）
        "SUPERSEDES": 0.8,        # 取代（强）
        "default": 0.5            # 默认权重
    }

    def __init__(
        self,
        graph: GraphStore,
        embed_encoder: Optional[EmbeddingEncoder] = None,
        init_energy: float = 1.0,
        decay_factor: float = 0.8,
        energy_threshold: float = 0.1,
        max_iterations: int = 10
    ):
        """初始化激活扩散遍历器

        Args:
            graph: 知识图谱存储
            embed_encoder: 编码器（用于计算节点间语义相似度）
            init_energy: 种子节点初始能量（默认1.0）
            decay_factor: 每跳能量衰减因子（默认0.8）
            energy_threshold: 能量传播停止阈值（默认0.1）
            max_iterations: 最大迭代次数，防止无限循环（默认10）
        """
        self.graph = graph
        self.embed = embed_encoder
        self.init_energy = init_energy
        self.decay_factor = decay_factor
        self.energy_threshold = energy_threshold
        self.max_iterations = max_iterations

        # 缓存节点嵌入，避免重复计算
        self._embedding_cache = {}

    def traverse(
        self,
        seed_node_ids: List[str],
        top_k: int = 50,
        include_context: bool = True
    ) -> List[Tuple[str, float, Dict[str, any]]]:
        """执行激活扩散遍历

        Args:
            seed_node_ids: 种子节点ID列表
            top_k: 返回Top-K个节点（按能量排序）
            include_context: 是否包含节点的上下文信息（时间、地点等）

        Returns:
            List of (node_id, energy_score, node_info_dict)
            node_info包含：node_type, node_obj, activation_path, context_nodes等
        """
        print(f"[SpreadingActivation] Starting traversal from {len(seed_node_ids)} seed nodes")

        # 初始化：给种子节点赋予初始能量
        node_energy = {node_id: self.init_energy for node_id in seed_node_ids}
        node_visited = set(seed_node_ids)
        activation_paths = {node_id: [node_id] for node_id in seed_node_ids}

        # 激活扩散主循环
        for iteration in range(self.max_iterations):
            print(f"[SpreadingActivation] Iteration {iteration + 1}/{self.max_iterations}")

            # 当前迭代的能量传播
            new_energy = {}
            total_energy_transferred = 0.0

            for node_id, current_energy in node_energy.items():
                # 如果能量低于阈值，停止传播
                if current_energy < self.energy_threshold:
                    continue

                # 获取节点的邻居（通过边连接）
                try:
                    neighbors = self._get_neighbors_with_edges(node_id)
                except Exception as e:
                    print(f"[SpreadingActivation] Error getting neighbors for {node_id}: {e}")
                    continue

                # 向每个邻居传播能量
                for neighbor_id, edge_data in neighbors:
                    # 计算这条边的传导率（conductance）
                    conductance = self._compute_conductance(node_id, neighbor_id, edge_data)

                    # 传播的能量 = 当前能量 * 传导率 * 衰减因子
                    transferred_energy = current_energy * conductance * self.decay_factor

                    if transferred_energy >= self.energy_threshold:
                        # 累加能量（如果是第一次访问）或更新
                        if neighbor_id not in new_energy:
                            new_energy[neighbor_id] = transferred_energy
                        else:
                            new_energy[neighbor_id] += transferred_energy

                        # 记录激活路径
                        if neighbor_id not in activation_paths:
                            activation_paths[neighbor_id] = activation_paths[node_id] + [neighbor_id]

                        total_energy_transferred += transferred_energy

            # 如果没有能量传播，提前终止
            if not new_energy:
                print(f"[SpreadingActivation] No energy propagation at iteration {iteration + 1}, terminating")
                break

            # 更新能量分布
            for node_id, energy in new_energy.items():
                node_energy[node_id] = node_energy.get(node_id, 0.0) + energy
                node_visited.add(node_id)

            print(f"[SpreadingActivation] Iteration {iteration + 1}: "
                  f"{len(new_energy)} new nodes activated, "
                  f"total energy transferred: {total_energy_transferred:.3f}, "
                  f"total active nodes: {len(node_energy)}")

            # 检查收敛条件（能量分布稳定）
            if total_energy_transferred < self.energy_threshold * len(node_energy):
                print(f"[SpreadingActivation] Converged at iteration {iteration + 1}")
                break

        print(f"[SpreadingActivation] Traversal complete. {len(node_energy)} nodes activated.")

        # 收集节点信息（用于后续证据生成）
        node_results = []
        for node_id, energy in node_energy.items():
            # 获取节点对象
            node_obj = self._get_node_object(node_id)
            if not node_obj:
                continue

            node_info = {
                "node_type": self._get_node_type(node_id),
                "node_obj": node_obj,
                "activation_path": activation_paths.get(node_id, [node_id]),
                "energy": energy,
            }

            # 如果是Event节点，自动带出其关联的Time和Location节点
            if include_context:
                node_info["context_nodes"] = self._get_context_nodes(node_id, node_info["node_type"])

            node_results.append((node_id, energy, node_info))

        # 按能量排序，返回Top-K
        node_results.sort(key=lambda x: x[1], reverse=True)
        top_results = node_results[:top_k]

        print(f"[SpreadingActivation] Top {len(top_results)} nodes by energy:")
        for i, (node_id, energy, node_info) in enumerate(top_results[:5], 1):
            node_type = node_info["node_type"]
            activation_path_length = len(node_info["activation_path"]) - 1
            print(f"  [{i}] {node_type} {node_id}: energy={energy:.3f}, hops={activation_path_length}")

        return top_results

    def _get_neighbors_with_edges(self, node_id: str) -> List[Tuple[str, Dict[str, any]]]:
        """获取节点的邻居及边信息

        Args:
            node_id: 节点ID

        Returns:
            List of (neighbor_id, edge_data_dict)
        """
        if self.graph._g is None:
            return []

        neighbors = []
        nx_graph = self.graph._g

        # 遍历所有出边
        for _, neighbor_id, edge_data in nx_graph.out_edges(node_id, data=True):
            neighbor_data = nx_graph.nodes.get(neighbor_id, {})

            # 确保邻居节点存在且有数据
            if not neighbor_data:
                continue

            # 获取边的关系类型和权重
            edge_info = {
                "rel_type": edge_data.get("rel_type", "related"),
                "weight": edge_data.get("weight", 1.0),
                "confidence": edge_data.get("confidence", 1.0),
                "metadata": edge_data
            }

            neighbors.append((neighbor_id, edge_info))

        return neighbors

    def _compute_conductance(self, source_id: str, target_id: str, edge_data: Dict[str, any]) -> float:
        """计算边的传导率（conductance），决定能量传播效率

        传导率 = 边类型权重 * 边置信度 * 节点语义相似度(可选)

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            edge_data: 边数据字典

        Returns:
            传导率值 [0, 1]
        """
        # 1. 边类型权重
        rel_type = edge_data.get("rel_type", "related")
        edge_type_weight = self.EDGE_TYPE_WEIGHTS.get(rel_type, self.EDGE_TYPE_WEIGHTS["default"])

        # 2. 边置信度
        confidence = edge_data.get("confidence", 1.0)

        # 3. 节点间语义相似度（如果提供了嵌入编码器）
        sem_similarity = 1.0
        if self.embed:
            sem_similarity = self._compute_semantic_similarity(source_id, target_id)

        # 综合传导率（各因子相乘）
        conductance = edge_type_weight * confidence * sem_similarity

        return min(conductance, 1.0)  # 限制在[0, 1]范围内

    def _compute_semantic_similarity(self, node_id1: str, node_id2: str) -> float:
        """计算两个节点的语义相似度（基于向量嵌入）

        Args:
            node_id1: 节点1ID
            node_id2: 节点2ID

        Returns:
            相似度值 [0, 1]
        """
        if not self.embed:
            return 1.0

        # 获取节点嵌入（带缓存）
        emb1 = self._get_node_embedding(node_id1)
        emb2 = self._get_node_embedding(node_id2)

        if emb1 is None or emb2 is None:
            return 1.0

        # 计算余弦相似度
        from numpy import dot
        from numpy.linalg import norm

        similarity = dot(emb1, emb2) / (norm(emb1) * norm(emb2) + 1e-12)

        # 转换为概率形式（sigmoid），并对负相似度做截断
        if similarity < 0:
            return 0.5  # 中性值

        return similarity

    def _get_node_embedding(self, node_id: str) -> Optional[List[float]]:
        """获取节点的嵌入向量（带缓存）

        Args:
            node_id: 节点ID

        Returns:
            嵌入向量，如果不存在则返回None
        """
        if node_id in self._embedding_cache:
            return self._embedding_cache[node_id]

        # 尝试从节点数据中获取嵌入
        try:
            node_data = self.graph._g.nodes.get(node_id, {})

            # 检查不同字段名
            embedding = None
            if hasattr(node_data, 'embedding'):
                embedding = node_data.embedding
            elif isinstance(node_data, dict) and 'embedding' in node_data:
                embedding = node_data['embedding']

            if embedding is not None:
                self._embedding_cache[node_id] = embedding
                return embedding
        except Exception as e:
            print(f"[SpreadingActivation] Error getting embedding for {node_id}: {e}")

        # 如果没有预计算的嵌入，尝试用节点描述生成（如果需要）
        return None

    def _get_node_object(self, node_id: str):
        """获取节点对象（兼容不同节点类型）

        Args:
            node_id: 节点ID

        Returns:
            节点对象，如果不存在则返回None
        """
        from ..storage.schema import Person, Event, Fact, Preference

        # 按优先级尝试不同类型
        check_methods = [
            lambda: self.graph.get_person(node_id),
            lambda: self.graph.get_event(node_id),
            lambda: self.graph.get_fact(node_id),
            lambda: self.graph.get_preference(node_id),
            lambda: self.graph.get_entity(node_id)
        ]

        for method in check_methods:
            try:
                obj = method()
                if obj:
                    return obj
            except:
                continue

        return None

    def _get_node_type(self, node_id: str) -> str:
        """获取节点类型字符串

        Args:
            node_id: 节点ID

        Returns:
            节点类型名称（Person, Event, Fact, Preference, Entity）
        """
        # 检查节点数据中的node_type字段
        try:
            node_data = self.graph._g.nodes.get(node_id, {})

            if hasattr(node_data, 'node_type'):
                return node_data.node_type

            if isinstance(node_data, dict) and 'node_type' in node_data:
                return node_data['node_type']
        except:
            pass

        # 回退到类型检查
        if self.graph.get_person(node_id):
            return "Person"
        elif self.graph.get_event(node_id):
            return "Event"
        elif self.graph.get_fact(node_id):
            return "Fact"
        elif self.graph.get_preference(node_id):
            return "Preference"
        else:
            return "Entity"

    def _get_context_nodes(self, node_id: str, node_type: str) -> List[Dict[str, any]]:
        """获取节点的上下文节点（如Event的时间/地点）

        Args:
            node_id: 节点ID
            node_type: 节点类型

        Returns:
            上下文节点信息列表
        """
        context_nodes = []

        # 只对Event节点获取上下文
        if node_type != "Event":
            return context_nodes

        try:
            nx_graph = self.graph._g

            # 查找与时间和位置相关的边
            for _, neighbor_id, edge_data in nx_graph.out_edges(node_id, data=True):
                rel_type = edge_data.get("rel_type", "")

                # 时间关系
                if rel_type in ["AT_TIME", "BEFORE", "AFTER"]:
                    neighbor_data = nx_graph.nodes.get(neighbor_id, {})
                    if neighbor_data:
                        context_nodes.append({
                            "node_id": neighbor_id,
                            "node_type": "TimeNormalization",
                            "rel_type": rel_type,
                            "data": neighbor_data
                        })

                # 位置关系
                elif rel_type in ["LOCATED_AT", "HAPPENED_AT"]:
                    neighbor_data = nx_graph.nodes.get(neighbor_id, {})
                    if neighbor_data:
                        context_nodes.append({
                            "node_id": neighbor_id,
                            "node_type": "Location",
                            "rel_type": rel_type,
                            "data": neighbor_data
                        })

        except Exception as e:
            print(f"[SpreadingActivation] Error getting context for {node_id}: {e}")

        return context_nodes

