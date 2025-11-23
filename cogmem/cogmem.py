from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .config import CogMemConfig, DEFAULT_CONFIG
from .storage import GraphStore, VectorStore, ChunkStore, CommunityStore, Doc, Entity, RetrievalResult, SemanticCommunity, StructureCommunity
from .memory.indexer import Indexer
from .retrieval import Retriever
from .community_module import CommunityDiscovery
from .generation import generate_answer
from .agents.qa_orchestrator import QAOrchestrator
from .utils.io import build_paths
from .utils.metrics import METRICS
from .utils.system_monitor import monitor


class CogMem:
    def __init__(self, config: Optional[CogMemConfig] = None, workspace_dir: Optional[str] = None, auto_load: bool = False) -> None:
        self.cfg = config or DEFAULT_CONFIG
        # 工作空间目录隔离：若指定则覆盖配置中的persist_dir
        if workspace_dir:
            self.cfg.persist_dir = workspace_dir
        self.workspace_dir = self.cfg.persist_dir
        
        self.graph = GraphStore()
        # 在构建向量库前，探测实际嵌入维度，避免与配置不一致
        actual_dim = None
        try:
            from .models.embedding import EmbeddingEncoder

            probe_vecs = EmbeddingEncoder(self.cfg).encode(["__cogmem_dim_probe__"])
            if probe_vecs and isinstance(probe_vecs[0], list):
                actual_dim = len(probe_vecs[0])
                if actual_dim and actual_dim != self.cfg.model.embedding_dim:
                    print(f"[CogMem] Warning: Embedding model returns {actual_dim} dimensions, "
                          f"but config specifies {self.cfg.model.embedding_dim}. "
                          f"Updating config to match actual dimension.")
                    self.cfg.model.embedding_dim = actual_dim
        except Exception as e:
            # 若探测失败，回退到配置中的 embedding_dim
            print(f"[CogMem] Warning: Failed to probe embedding dimension: {e}. "
                  f"Using config value: {self.cfg.model.embedding_dim}")

        # 在创建VectorStore之前，检查已存在的索引维度
        index_dim = None
        try:
            base = self.workspace_dir
            from .utils.io import build_paths
            gpath, ipath, eidspath = build_paths(
                base, self.cfg.graph_file, self.cfg.faiss_index_file, self.cfg.entity_store_file
            )
            import os
            if os.path.exists(ipath):
                import faiss
                loaded_index = faiss.read_index(ipath)
                index_dim = loaded_index.d
                print(f"[CogMem] Found existing index with {index_dim} dimensions")
        except Exception:
            # 如果加载失败，忽略（可能是首次运行）
            pass
        
        # 决定使用哪个维度
        # 关键：必须使用实际模型返回的维度，而不是索引维度
        # 如果索引维度与模型维度不匹配，应该报错而不是静默使用错误的维度
        
        # 首先确定实际模型维度
        final_dim = actual_dim if actual_dim is not None else self.cfg.model.embedding_dim
        
        # 检查维度不匹配
        if index_dim is not None and actual_dim is not None and index_dim != actual_dim:
            print(f"[CogMem] ❌ CRITICAL DIMENSION MISMATCH:")
            print(f"  - Existing index dimension: {index_dim}")
            print(f"  - Current embedding model returns: {actual_dim} dimensions")
            print(f"  - This mismatch will cause search failures!")
            print(f"  - SOLUTION OPTIONS:")
            print(f"    1. Rebuild index: Delete the existing index and rebuild with current model")
            print(f"    2. Use matching model: Switch to the embedding model that built the index ({index_dim}D)")
            print(f"  - Current action: Using model dimension ({actual_dim}D) - index will need to be rebuilt")
            # 不强制使用索引维度，而是使用模型实际维度
            # 这样至少可以正常工作，只是需要重建索引
        
        # 使用实际模型维度（不是索引维度）
        if actual_dim is not None:
            self.cfg.model.embedding_dim = actual_dim
            print(f"[CogMem] Using embedding dimension: {self.cfg.model.embedding_dim} (from actual model)")
        else:
            print(f"[CogMem] Using embedding dimension: {self.cfg.model.embedding_dim} (from config)")
        
        # 如果索引存在但维度不匹配，给出额外提示
        if index_dim is not None and index_dim != self.cfg.model.embedding_dim:
            print(f"[CogMem] ⚠️  Index dimension ({index_dim}) differs from model dimension ({self.cfg.model.embedding_dim})")
            print(f"[CogMem] ⚠️  The index will be incompatible. Consider rebuilding it.")
        
        self.vstore = VectorStore(dim=self.cfg.model.embedding_dim, metric=self.cfg.faiss.metric)
        self.chunk_store = ChunkStore(dim=self.cfg.model.embedding_dim, metric=self.cfg.faiss.metric)
        self.indexer = Indexer(self.cfg, self.graph, self.vstore, chunk_store=self.chunk_store, cogmem=self)
        self.community = CommunityDiscovery(self.cfg)
        self.community_store = CommunityStore()
        self.retriever = Retriever(self.cfg, self.graph, self.vstore, self.community_store, self.chunk_store)
        
        # 初始化嵌入编码器（供agents使用）
        from .models.embedding import EmbeddingEncoder
        self._embed = EmbeddingEncoder(self.cfg)
        
        # 初始化Agentic问答协调器
        self.qa_orchestrator = QAOrchestrator(self)
        
        # 自动加载已有索引
        if auto_load:
            self.load()

    # --- 指标 ---
    def get_metrics(self) -> Dict:
        return METRICS.snapshot()

    def reset_metrics(self) -> None:
        METRICS.reset()

    def get_performance_summary(self) -> Dict:
        """Get system performance summary including retrieval metrics."""
        return monitor.get_performance_summary()

    def get_recent_operations(self, limit: int = 10) -> List[Dict]:
        """Get details of recent operations for debugging."""
        return monitor.get_recent_operations(limit)

    def clear_performance_history(self) -> None:
        """Clear performance monitoring history."""
        monitor.clear_history()

    # --- 数据管线 ---
    def add_documents(self, docs: Iterable[Doc], incremental: bool = False) -> List[Entity]:
        """Add documents to the knowledge graph

        Uses person-centered, event-mediated extraction (now the default and only method).

        Args:
            docs: Documents to add
            incremental: Whether to skip already processed documents
        """
        self.indexer.add_documents(docs, incremental=incremental)
        # Return empty list for backward compatibility
        return []

    def summarize_dialog(self, turns: List[str]) -> str:
        # MVP: 简单拼接与截断
        text = "\n".join(turns)
        return text[-1000:]

    def build_indices(self, rebuild: bool = False) -> None:
        # MVP: 向量与图已在 add_documents 即时构建，无需额外操作
        return None

    def discover_communities(self, structure: bool = False, semantic: bool = True) -> None:
        """发现社区：支持所有节点类型（Person/Event/Fact/Entity）

        Args:
            structure: 是否发现结构社区（基于图连接）
            semantic: 是否发现语义社区（基于embedding相似度）
        """
        from tqdm import tqdm
        import time

        start_time = time.time()
        print(f"[CommunityDiscovery] Starting community discovery at {time.strftime('%H:%M:%S')}")

        # Semantic communities
        if semantic:
            print("[CommunityDiscovery] [1/6] Collecting nodes for semantic community discovery...")
            ids = []
            texts = []  # 用于embedding的文本表示

            # 收集所有节点类型（Person/Event/Fact/Entity）
            graph = self.graph._g
            total_nodes = len(graph.nodes())

            with tqdm(total=total_nodes, desc="Processing nodes", unit="node") as pbar:
                for node_id, node_data in graph.nodes(data=True):
                    if not node_data:
                        pbar.update(1)
                        continue

                    node_type = node_data.get("node_type", "Entity")
                    text_representation = None

                    # 根据节点类型生成文本表示
                    if node_type == "Person":
                        canonical_name = node_data.get("canonical_name", "")
                        aliases = node_data.get("aliases", [])
                        description = node_data.get("description", "")
                        text_representation = f"{canonical_name} {' '.join(aliases)} {description}".strip()
                    elif node_type == "Event":
                        title = node_data.get("title", "")
                        location = node_data.get("location", "")
                        description = node_data.get("description", "")
                        text_representation = f"{title} {location} {description}".strip()
                    elif node_type == "Fact":
                        subject_id = node_data.get("subject", "")
                        predicate = node_data.get("predicate", "")
                        object_val = node_data.get("object", "")
                        value = node_data.get("value", "")
                        # 尝试解析subject名称
                        subject_name = subject_id
                        subject_node = graph.nodes.get(subject_id, {})
                        if subject_node:
                            subject_name = subject_node.get("canonical_name") or subject_node.get("name", subject_id)
                        text_representation = f"{subject_name} {predicate} {object_val or value}".strip()
                    elif node_type == "Entity":
                        name = node_data.get("name", "")
                        description = node_data.get("description", "")
                        text_representation = f"{name} {description}".strip()
                    else:
                        # 对于其他类型，尝试提取name或title
                        name = node_data.get("name") or node_data.get("title") or node_data.get("canonical_name", "")
                        description = node_data.get("description", "")
                        text_representation = f"{name} {description}".strip()

                    if text_representation:
                        ids.append(node_id)
                        texts.append(text_representation)

                    pbar.update(1)

            if not ids:
                print("[CommunityDiscovery] No nodes found for community discovery")
                return

            print(f"[CommunityDiscovery] [2/6] Prepared {len(ids)} nodes with text representations")

            print(f"[CommunityDiscovery] [3/6] Generating embeddings for {len(ids)} nodes...")
            embed_start = time.time()

            # 生成embeddings
            from .models.embedding import EmbeddingEncoder
            enc = EmbeddingEncoder(self.cfg)

            # 分批处理以显示进度
            batch_size = 100
            vecs = []
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
                batch_texts = texts[i:i+batch_size]
                batch_vecs = enc.encode(batch_texts)
                vecs.extend(batch_vecs)

            embed_time = time.time() - embed_start
            print(f"[CommunityDiscovery] Embedding completed in {embed_time:.2f}s")

            print(f"[CommunityDiscovery] [4/6] Running semantic clustering algorithm...")
            cluster_start = time.time()

            # 执行社区发现
            sem = self.community.semantic_clusters(ids, vecs)

            cluster_time = time.time() - cluster_start
            print(f"[CommunityDiscovery] Clustering completed in {cluster_time:.2f}s")

            print(f"[CommunityDiscovery] [5/6] Computing centroids and saving communities...")
            save_start = time.time()

            # 计算质心并持久化
            id_to_index = {eid: i for i, eid in enumerate(ids)}
            communities: List[SemanticCommunity] = []
            import numpy as np

            X = np.asarray(vecs, dtype=float) if vecs else None
            for c in sem:
                idxs = [id_to_index[eid] for eid in c.member_ids if eid in id_to_index]
                if not idxs or X is None:
                    continue
                centroid = X[idxs].mean(axis=0).tolist()
                communities.append(SemanticCommunity(id=c.id, member_ids=c.member_ids, centroid=centroid))

            self.community_store.set_semantic(communities)

            save_time = time.time() - save_start

            # 统计信息
            total_members = sum(len(c.member_ids) for c in communities)
            avg_size = total_members / len(communities) if communities else 0

            print(f"[CommunityDiscovery] Semantic communities saved: {len(communities)} communities")
            print(f"[CommunityDiscovery] Total members: {total_members}, Avg size: {avg_size:.1f}")
            print(f"[CommunityDiscovery] Save time: {save_time:.2f}s")

            # 生成 TopicNodes 和边关系
            print(f"[CommunityDiscovery] [6/6] Generating topics and edges...")
            topic_start = time.time()

            # Initialize LLM for community discovery
            if hasattr(self, '_llm') and self._llm:
                self.community.initialize_llm(self._llm)

            topics, topic_relations = self.community.update_topics(
                communities=sem,  # Use the raw communities (before adding centroids)
                entity_ids=ids,
                vectors=vecs,
                graph_store=self.graph,
                cluster_type="semantic"
            )

            # Add topics and relations to graph
            if topics:
                from .storage.schema import Topic, Relation
                self.graph.add_nodes(topics)

            if topic_relations:
                self.graph.add_relations(topic_relations)

            topic_time = time.time() - topic_start
            print(f"[CommunityDiscovery] Topics generated: {len(topics)}, Relations: {len(topic_relations)}")
            print(f"[CommunityDiscovery] Topic generation time: {topic_time:.2f}s")
        # 结构社区（可选）
        if structure and self.cfg.community.enable_structure:
            print("[CommunityDiscovery] [6/6] Discovering structure communities...")

            struct_start = time.time()

            # 从图中导出关系边
            edges = []
            # MultiDiGraph edges with data
            total_edges = len(self.graph._g.edges())  # type: ignore[attr-defined]

            with tqdm(total=total_edges, desc="Extracting edges", unit="edge") as pbar:
                for u, v, data in self.graph._g.edges(data=True):  # type: ignore[attr-defined]
                    edges.append((u, v))
                    pbar.update(1)

            print(f"[CommunityDiscovery] Extracted {len(edges)} edges for structure analysis")

            str_coms = self.community.structure_leiden(edges)

            if str_coms:
                str_store = [StructureCommunity(id=c.id, member_ids=c.member_ids) for c in str_coms]
                self.community_store.set_structure(str_store)

                struct_time = time.time() - struct_start
                total_struct_members = sum(len(c.member_ids) for c in str_store)
                avg_struct_size = total_struct_members / len(str_store) if str_store else 0

                print(f"[CommunityDiscovery] Structure communities saved: {len(str_store)} communities")
                print(f"[CommunityDiscovery] Total members: {total_struct_members}, Avg size: {avg_struct_size:.1f}")
                print(f"[CommunityDiscovery] Structure time: {struct_time:.2f}s")
            else:
                print("[CommunityDiscovery] No structure communities found")

        total_time = time.time() - start_time
        print(f"[CommunityDiscovery] Total community discovery completed in {total_time:.2f}s")

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        """使用重构的简化检索系统进行检索"""
        if not hasattr(self, '_retriever'):
            self._retriever = Retriever(self.cfg, self.graph, self.vstore, self.community_store, self.chunk_store)
        return self._retriever.retrieve(query, top_k=top_k)

    def answer(
        self, 
        query: str, 
        top_k: int = 10, 
        use_enhanced_retrieval: bool = True,
        enable_post_processing: bool = True
    ) -> str:
        """生成答案（使用Agentic方式）
        
        Args:
            query: 用户查询
            top_k: 返回的实体数量
            use_enhanced_retrieval: 是否使用增强版检索（保留参数以兼容旧代码，但实际使用Agentic）
            enable_post_processing: 是否启用答案后处理（保留参数以兼容旧代码）
            
        Returns:
            生成的答案
        """
        # 使用Agentic问答协调器
        return self.qa_orchestrator.answer(query, top_k)

    def update_entity(self, entity_id: str, patch: Dict) -> None:
        e = self.graph.get_entity(entity_id)
        if not e:
            return
        for k, v in patch.items():
            if hasattr(e, k):
                setattr(e, k, v)
        # 更新更新时间戳（可读本地时间）
        try:
            import time
            e.metadata.updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            e.version = (e.version or 1) + 1
        except Exception:
            pass
        self.graph.add_entities([e])

    # --- 持久化 ---
    def save(self, base_dir: Optional[str] = None) -> None:
        base = base_dir or self.workspace_dir
        gpath, ipath, eidspath = build_paths(
            base, self.cfg.graph_file, self.cfg.faiss_index_file, self.cfg.entity_store_file
        )
        # 变更日志路径
        import os
        changelog_path = os.path.join(base, self.cfg.graph_changelog_file)
        self.graph.save(gpath, changelog_path=changelog_path)
        self.vstore.save(ipath, eidspath)
        # 片段索引
        if self.chunk_store:
            self.chunk_store.save(
                os.path.join(base, self.cfg.chunk_index_file),
                os.path.join(base, self.cfg.chunk_meta_file),
            )
        # 社区
        from .utils.io import ensure_dir

        ensure_dir(base)
        self.community_store.save(
            os.path.join(base, self.cfg.community_meta_file),
            os.path.join(base, self.cfg.community_centroid_file),
        )
        # EvidenceStore（已处理的文档ID）
        evidence_store_path = os.path.join(base, self.cfg.evidence_store_file)
        self.indexer.evidence.save(evidence_store_path)

    def load(self, base_dir: Optional[str] = None) -> None:
        base = base_dir or self.workspace_dir
        gpath, ipath, eidspath = build_paths(
            base, self.cfg.graph_file, self.cfg.faiss_index_file, self.cfg.entity_store_file
        )
        # 变更日志路径
        import os
        changelog_path = os.path.join(base, self.cfg.graph_changelog_file)
        self.graph.load(gpath, changelog_path=changelog_path)
        try:
            self.vstore.load(ipath, eidspath)
        except ValueError as e:
            # 维度不匹配错误
            print(f"[CogMem] ❌ Failed to load index: {e}")
            print(f"[CogMem] The index dimension doesn't match the current embedding model.")
            print(f"[CogMem] Please rebuild the index or switch to the correct embedding model.")
            raise
        # 片段索引
        if self.chunk_store:
            self.chunk_store.load(
                os.path.join(base, self.cfg.chunk_index_file),
                os.path.join(base, self.cfg.chunk_meta_file),
            )
        # 社区
        self.community_store.load(
            os.path.join(base, self.cfg.community_meta_file),
            os.path.join(base, self.cfg.community_centroid_file),
        )
        # EvidenceStore（已处理的文档ID）
        evidence_store_path = os.path.join(base, self.cfg.evidence_store_file)
        self.indexer.evidence.load(evidence_store_path)

    def is_indexed(self) -> bool:
        """检查工作空间是否存在完整索引"""
        import os
        base = self.workspace_dir
        gpath, ipath, eidspath = build_paths(
            base, self.cfg.graph_file, self.cfg.faiss_index_file, self.cfg.entity_store_file
        )
        return os.path.exists(gpath) and os.path.exists(ipath) and os.path.exists(eidspath)

    def add_documents_incremental(self, docs: Iterable[Doc], force_rebuild: bool = False) -> List[Entity]:
        """增量索引：若已有索引且force_rebuild=False，只处理新文档；否则全量重建"""
        if not force_rebuild and self.is_indexed():
            # 加载已有索引
            self.load()
            # 获取已有实体ID集合，用于去重
            existing_ids = {e.id for e in self.graph.iter_entities()}
            # 只处理新文档（增量逻辑由indexer内部处理）
            self.indexer.add_documents(docs, incremental=True)
            # Note: The new add_documents doesn't return entities, but we can extract them if needed
            new_entities = []  # For backward compatibility, return empty list
            return new_entities
        else:
            # 全量重建
            return self.add_documents(docs)
