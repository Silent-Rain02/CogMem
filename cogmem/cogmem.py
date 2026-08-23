"""CogMem 主类：长期记忆系统的核心接口

提供文档索引、知识图谱构建、智能检索和问答等功能。
"""

from __future__ import annotations

import threading
from copy import deepcopy
from typing import Dict, Iterable, List, Optional

from .config import CogMemConfig, DEFAULT_CONFIG
from .storage import GraphStore, VectorStore, ChunkStore, Doc, Concept
from .storage.typed_vector_store import TypedVectorStore
from .memory.indexer import Indexer
from .agents.qa_orchestrator import QAOrchestrator
from .utils.io import build_paths
from .utils.metrics import METRICS
from .utils.system_monitor import monitor

class CogMem:
    """CogMem 主类：长期记忆系统的核心接口
    
    功能包括：
    - 文档索引：将文档转换为知识图谱
    - 智能检索：基于图谱和向量的混合检索
    - 问答系统：基于Agent的智能问答
    - 记忆巩固：将事件转换为语义事实
    
    使用示例：
        ```python
        from cogmem import CogMem
        
        # 初始化
        cogmem = CogMem(workspace_dir="./workspace")
        
        # 添加文档
        docs = [Doc(id="d1", text="Alice likes Bob.")]
        cogmem.add_documents(docs)
        
        # 问答
        answer = cogmem.answer("Who does Alice like?")
        ```
    """
    def __init__(self, config: Optional[CogMemConfig] = None, workspace_dir: Optional[str] = None, auto_load: bool = False) -> None:
        """初始化CogMem实例
        
        Args:
            config: 配置对象，如果为None则使用默认配置
            workspace_dir: 工作空间目录，如果指定则覆盖配置中的persist_dir
            auto_load: 是否自动加载已有索引
        """
        # Each instance owns its configuration. The evaluation runners apply
        # per-run overrides, so sharing DEFAULT_CONFIG would leak settings and
        # workspace paths across instances in the same Python process.
        self.cfg = deepcopy(config) if config is not None else deepcopy(DEFAULT_CONFIG)
        
        # 工作空间目录隔离：若指定则覆盖配置中的persist_dir
        if workspace_dir:
            self.cfg.persist_dir = workspace_dir
        self.workspace_dir = self.cfg.persist_dir
        
        # 保存锁：防止并发保存操作
        self._save_lock = threading.Lock()
        
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

        # 在创建VectorStore之前，检查已存在的索引维度（用于维度兼容性检查）
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
        # 必须使用实际模型返回的维度，而不是索引维度
        # 如果索引维度与模型维度不匹配，应该报错而不是静默使用错误的维度
        
        # 首先确定实际模型维度
        final_dim = actual_dim if actual_dim is not None else self.cfg.model.embedding_dim
        
        # 检查维度不匹配：如果索引存在但维度不匹配，抛出异常
        if index_dim is not None and actual_dim is not None and index_dim != actual_dim:
            error_msg = (
                f"CRITICAL: Index dimension mismatch!\n"
                f"  - Existing index: {index_dim} dimensions\n"
                f"  - Current embedding model: {actual_dim} dimensions\n"
                f"  - This mismatch will cause search failures!\n"
                f"\n"
                f"SOLUTION:\n"
                f"  1. Rebuild index: Delete {self.workspace_dir} and rebuild with current model\n"
                f"  2. Switch to matching model: Use the embedding model that built the index ({index_dim}D)\n"
            )
            raise ValueError(error_msg)
        
        # 使用实际模型维度（不是索引维度）
        if actual_dim is not None:
            self.cfg.model.embedding_dim = actual_dim
            print(f"[CogMem] Using embedding dimension: {self.cfg.model.embedding_dim} (from actual model)")
        else:
            print(f"[CogMem] Using embedding dimension: {self.cfg.model.embedding_dim} (from config)")
        
        # 如果索引存在但维度不匹配，给出额外提示
        if index_dim is not None and index_dim != self.cfg.model.embedding_dim:
            print(f"[CogMem] [WARN] Index dimension ({index_dim}) differs from model dimension ({self.cfg.model.embedding_dim})")
            print(f"[CogMem] [WARN] The index will be incompatible. Consider rebuilding it.")
        
        # 使用TypedVectorStore（按节点类型分片，优化内存使用）
        self.vstore = TypedVectorStore(
            dim=self.cfg.model.embedding_dim,
            metric=self.cfg.faiss.metric,
            enable_global_index=self.cfg.faiss.enable_global_index
        )
        self.chunk_store = ChunkStore(dim=self.cfg.model.embedding_dim, metric=self.cfg.faiss.metric)
        self.indexer = Indexer(self.cfg, self.graph, self.vstore, chunk_store=self.chunk_store, cogmem=self)

        # 初始化嵌入编码器（供agents使用）
        from .models.embedding import EmbeddingEncoder
        self._embed = EmbeddingEncoder(self.cfg)

        # 初始化LLM客户端（供agents使用）
        from .models.llm_client import create_llm_client
        self.llm = create_llm_client(self.cfg)

        # 初始化Agentic问答协调器
        self.qa_orchestrator = QAOrchestrator(self)
        
        # 自动加载已有索引
        if auto_load:
            self.load()

    # ==================== 指标和监控 ====================
    
    def get_metrics(self) -> Dict:
        """获取系统指标快照"""
        return METRICS.snapshot()

    def reset_metrics(self) -> None:
        """重置所有指标"""
        METRICS.reset()

    def print_metrics(self) -> None:
        """打印当前系统指标摘要"""
        snapshot = METRICS.snapshot()
        chat = snapshot.get("chat", {})
        embed = snapshot.get("embed", {})
        
        print("\n" + "="*40)
        print("📊 系统性能指标 (LLM Usage)")
        print("="*40)
        
        if chat.get("calls", 0) > 0:
            print(f"【Chat】")
            print(f"  - 调用次数: {chat.get('calls')}")
            print(f"  - 缓存命中: {chat.get('cache_hits')} ({chat.get('cache_hits')/chat.get('calls'):.1%})")
            print(f"  - 总 Token: {chat.get('total_tokens')}")
            print(f"  - 提示/补全: {chat.get('prompt_tokens')} / {chat.get('completion_tokens')}")
            print(f"  - 平均延迟: {chat.get('total_latency_ms')/chat.get('calls'):.0f} ms")
        
        if embed.get("calls", 0) > 0:
            print(f"\n【Embedding】")
            print(f"  - 调用次数: {embed.get('calls')}")
            print(f"  - 缓存命中: {embed.get('cache_hits')} ({embed.get('cache_hits')/embed.get('calls'):.1%})")
            print(f"  - 总 Token: {embed.get('total_tokens')}")
            print(f"  - 平均延迟: {embed.get('total_latency_ms')/embed.get('calls'):.0f} ms")
        
        print("="*40 + "\n")

    def get_performance_summary(self) -> Dict:
        """获取系统性能摘要（包括检索指标）"""
        return monitor.get_performance_summary()

    def get_recent_operations(self, limit: int = 10) -> List[Dict]:
        """获取最近的操作详情（用于调试）"""
        return monitor.get_recent_operations(limit)

    def clear_performance_history(self) -> None:
        """清除性能监控历史"""
        monitor.clear_history()

    # ==================== 数据管线 ====================
    def add_documents(self, docs: Iterable[Doc], incremental: bool = False) -> List[Concept]:
        """添加文档到知识图谱
        
        使用人物中心、事件中介的抽取方法（默认且唯一的方法）。
        
        Args:
            docs: 要添加的文档列表
            incremental: 是否跳过已处理的文档（增量索引）
            
        Returns:
            空列表（为了向后兼容）
        """
        self.indexer.add_documents(docs, incremental=incremental)
        return []

    def summarize_dialog(self, turns: List[str]) -> str:
        """总结对话（简单实现：拼接并截断）
        
        Args:
            turns: 对话轮次列表
            
        Returns:
            总结后的文本（最多1000字符）
        """
        text = "\n".join(turns)
        return text[-1000:]

    def _extract_retrieval_info_from_agent(self):
        """从search_agent的history中提取检索实体和路径信息
        
        Returns:
            包含entities、evidence、extracted_entity_names、retrieval_path等字段的对象
        """
        import re
        from types import SimpleNamespace

        entities = set()
        retrieval_path = []
        search_agent = getattr(self.qa_orchestrator, 'search_agent', None)

        if not search_agent or not hasattr(search_agent, 'history'):
            return SimpleNamespace(
                entities=[],
                evidence=[],
                extracted_entity_names=[],
                retrieval_path=[],
                candidates=[]
            )

        # 从history中提取信息
        for step in search_agent.history:
            # CRITICAL FIX: 添加防御性编程，确保所有数据都是可哈希的
            try:
                # 构建检索路径
                step_dict = {
                    'step': getattr(step, 'step', len(retrieval_path) + 1),
                    'thought': getattr(step, 'thought', '') or '',
                    'action': getattr(step, 'action', 'unknown') or 'unknown',
                    'action_input': getattr(step, 'action_input', {}) or {},
                    'observation': str(getattr(step, 'observation', ''))[:500] if not isinstance(getattr(step, 'observation', ''), (dict, list)) else str(getattr(step, 'observation', ''))[:500]
                }
                retrieval_path.append(step_dict)

                # 从action_input提取节点ID
                action_input = getattr(step, 'action_input', {})
                if isinstance(action_input, dict):
                    # 遍历action_input的所有值，寻找字符串类型的节点ID
                    for key, value in action_input.items():
                        if isinstance(value, str):
                            # 检查是否符合节点ID格式
                            if re.match(r'^(p_\w+|event_[0-9a-f]{40}|fact_[0-9a-f]{40}|claim_[0-9a-f]{40})$', value):
                                entities.add(value)
                        elif isinstance(value, dict):
                            # 如果值是字典，尝试从中提取node_id
                            # 修复：确保只添加字符串类型的ID，避免 unhashable type: 'dict' 错误
                            try:
                                node_id = value.get('node_id')
                                if isinstance(node_id, str):
                                    entities.add(node_id)
                                # 也尝试其他可能的ID字段
                                elif 'id' in value and isinstance(value['id'], str):
                                    entities.add(value['id'])
                                elif 'name' in value and isinstance(value['name'], str):
                                    # name 可能不是节点ID，但如果是符合格式的字符串，也可以添加
                                    name = value['name']
                                    # 确保 name 是字符串而不是字典或其他类型
                                    if isinstance(name, str) and re.match(r'^(p_\w+|event_[0-9a-f]{40}|fact_[0-9a-f]{40}|claim_[0-9a-f]{40})$', name):
                                        entities.add(name)
                            except (AttributeError, TypeError):
                                # 如果value不是字典或访问出错，跳过
                                pass

                    # 特别处理traverse_neighbors操作
                    if getattr(step, 'action', '') == 'traverse_neighbors' and 'node_id' in action_input:
                        node_id = action_input['node_id']
                        if isinstance(node_id, str) and node_id in self.graph._g:
                            try:
                                neighbors = list(self.graph._g.neighbors(node_id))
                                for neighbor in neighbors:
                                    if isinstance(neighbor, str):
                                        entities.add(neighbor)
                            except Exception:
                                pass

                # 从observation中提取节点ID
                observation = getattr(step, 'observation', '')
                if isinstance(observation, str):
                    # 匹配各种节点ID格式
                    node_matches = re.findall(
                        r'\b(p_\w+|event_[0-9a-f]{40}|fact_[0-9a-f]{40}|claim_[0-9a-f]{40})\b',
                        observation
                    )
                    for match in node_matches:
                        if isinstance(match, str):
                            entities.add(match)
                # 如果observation是列表或字典，转换为字符串再匹配
                elif isinstance(observation, (list, dict)):
                    try:
                        obs_str = str(observation)
                        node_matches = re.findall(
                            r'\b(p_\w+|event_[0-9a-f]{40}|fact_[0-9a-f]{40}|claim_[0-9a-f]{40})\b',
                            obs_str
                        )
                        for match in node_matches:
                            if isinstance(match, str):
                                entities.add(match)
                    except Exception:
                        # 如果转换失败，跳过
                        pass

            except Exception as e:
                # 如果处理某个步骤失败，记录错误但继续处理其他步骤
                continue

        # 获取实体的详细信息
        retrieved_entities = []
        for entity_id in list(entities):
            node_data = self.graph._g.nodes.get(entity_id)
            if node_data and isinstance(node_data, dict):
                node_type = node_data.get('node_type', 'unknown')
                
                # 根据节点类型提取正确的description字段
                # 严格按照schema设计进行字段提取
                description = ''
                if node_type == 'claim':
                    # Claim节点：使用content字段作为description
                    description = node_data.get('content', '')
                elif node_type == 'event':
                    # Event节点：优先使用description，如果没有则从title构建
                    description = node_data.get('description', '')
                    if not description:
                        title = node_data.get('title', '')
                        if title:
                            description = f"Event: {title}"
                elif node_type == 'person':
                    # Person节点：优先使用description，如果没有则从canonical_name构建
                    description = node_data.get('description', '')
                    if not description:
                        canonical_name = node_data.get('canonical_name', '')
                        if canonical_name:
                            description = f"Person: {canonical_name}"
                elif node_type == 'fact':
                    # Fact节点：优先使用description，如果没有则从subject/predicate/object构建
                    description = node_data.get('description', '')
                    if not description:
                        subject = node_data.get('subject', '')
                        predicate = node_data.get('predicate', '')
                        obj = node_data.get('object', '')
                        if subject and predicate:
                            if obj:
                                description = f"Fact: {subject} {predicate} {obj}"
                            else:
                                description = f"Fact: {subject} {predicate}"
                elif node_type == 'concept':
                    # Concept节点：使用description字段
                    description = node_data.get('description', '')
                else:
                    # 其他类型：尝试使用description字段
                    description = node_data.get('description', '')
                
                # 提取canonical_name或name
                canonical_name = node_data.get('canonical_name', '')
                if not canonical_name:
                    canonical_name = node_data.get('name', entity_id)
                
                entity = SimpleNamespace(
                    node_id=entity_id,
                    canonical_name=canonical_name,
                    description=description,
                    type=node_type
                )
                retrieved_entities.append(entity)

        # 从 search_agent history 中提取 evidence 信息
        # P0 FIX: 改进证据提取，尝试从节点和evidence_store中获取Turn ID信息
        evidence_list = []
        search_agent = getattr(self.qa_orchestrator, 'search_agent', None)
        if search_agent and hasattr(search_agent, 'history'):
            import re
            for step in search_agent.history:
                observation = getattr(step, 'observation', '')
                if isinstance(observation, str) and observation:
                    # 尝试从节点中提取 Turn ID
                    # 从 observation 中提取节点 ID
                    node_ids = re.findall(
                        r'\b(p_\w+|event_[0-9a-f]{40}|fact_[0-9a-f]{40}|claim_[0-9a-f]{40})\b',
                        observation
                    )
                    
                    # P0 FIX: 尝试从节点的evidence_span或metadata中提取turn_id
                    # 或者从evidence_store的chunk文本中提取turn_id（如果chunk包含Turn ID格式）
                    turn_ids_found = set()
                    
                    # 方法1: 从节点数据中提取（如果节点有metadata或evidence_span）
                    for node_id in node_ids[:5]:  # 限制检查的节点数量
                        try:
                            node_data = self.graph._g.nodes.get(node_id)
                            if node_data and isinstance(node_data, dict):
                                # 检查是否有evidence_span字段（可能包含turn_id信息）
                                evidence_span = node_data.get('evidence_span')
                                if evidence_span and isinstance(evidence_span, dict):
                                    # evidence_span可能包含turn_id或doc_id信息
                                    span_text = evidence_span.get('text', '')
                                    # 尝试从文本中提取Turn ID格式（如"D2:8"）
                                    turn_id_matches = re.findall(r'\b(D\d+:\d+)\b', span_text)
                                    turn_ids_found.update(turn_id_matches)
                        except Exception:
                            pass
                    
                    # 方法2: 从evidence_store的chunk文本中提取Turn ID
                    if hasattr(self, 'indexer') and hasattr(self.indexer, 'evidence'):
                        for node_id in node_ids[:3]:  # 限制检查的节点数量
                            try:
                                chunks = self.indexer.evidence.get_chunks(node_id, limit=3)
                                for chunk_text in chunks:
                                    # 尝试从chunk文本中提取Turn ID格式
                                    turn_id_matches = re.findall(r'\b(D\d+:\d+)\b', chunk_text)
                                    turn_ids_found.update(turn_id_matches)
                                    if len(turn_ids_found) >= 3:  # 限制最多3个Turn ID
                                        break
                                if len(turn_ids_found) >= 3:
                                    break
                            except Exception:
                                pass
                    
                    # 构建证据字符串，如果找到 turn_id 则包含
                    evidence_text = observation[:500]  # 限制长度
                    if turn_ids_found:
                        turn_ids_str = ", ".join(sorted(list(turn_ids_found)[:3]))  # 最多3个
                        evidence_text = f"[Turn ID: {turn_ids_str}] {evidence_text}"
                    
                    evidence_list.append(evidence_text)

        return SimpleNamespace(
            entities=retrieved_entities,
            evidence=evidence_list,  # 使用实际的证据列表而不是空列表
            extracted_entity_names=list(entities),
            retrieval_path=retrieval_path,
            candidates=[]
        )

    def answer(
        self,
        query: str,
        top_k: int = 10,
        use_enhanced_retrieval: bool = True,
        enable_post_processing: bool = True,
        return_retrieval_result: bool = False,
        return_full_result: bool = False,
        verbose: bool = False
    ):
        """生成答案（使用SearchAgent进行图谱推理）

        Args:
            query: 用户查询
            top_k: 返回的实体数量（保留参数以兼容旧代码）
            use_enhanced_retrieval: 是否使用增强版检索（保留参数以兼容旧代码）
            enable_post_processing: 是否启用答案后处理（保留参数以兼容旧代码）
            return_retrieval_result: 是否同时返回检索结果（如果为True，返回(answer, retrieval_result)元组）
            return_full_result: 是否返回完整结果字典（包含answer, reasoning_path, confidence）

        Returns:
            - 如果 return_full_result=True：返回完整字典 {answer, reasoning_path, confidence}
            - 如果 return_retrieval_result=True：返回 (answer, retrieval_result) 元组
            - 否则（默认）：只返回答案字符串
        """
        # 直接调用search_agent.run()获取完整结果，避免并发问题
        # 这样可以确保reasoning_path被正确提取
        search_agent = self.qa_orchestrator.search_agent
        result = search_agent.run(query, verbose=verbose)
        
        # 保存结果到last_result供向后兼容
        search_agent.last_result = result
        
        answer = result.get('answer', 'I could not generate an answer.')
        
        # 如果请求返回完整结果，返回整个字典
        if return_full_result:
            return result
        
        # 为了向后兼容，如果请求返回检索结果，返回元组
        if return_retrieval_result:
            # 从SearchAgent的history中提取实际的检索信息
            retrieval_result = self._extract_retrieval_info_from_agent()
            return answer, retrieval_result
        else:
            return answer

    def update_concept(self, concept_id: str, patch: Dict) -> None:
        """更新 Concept 节点
        
        Args:
            concept_id: Concept节点ID
            patch: 要更新的字段字典
        """
        c = self.graph.get_concept(concept_id)
        if not c:
            return
        for k, v in patch.items():
            if hasattr(c, k):
                setattr(c, k, v)
        # 更新更新时间戳（可读本地时间）
        try:
            import time
            c.metadata.updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        except Exception:
            pass
        self.graph.add_concepts([c])
    
    def update_entity(self, entity_id: str, patch: Dict) -> None:
        """向后兼容：重定向到 update_concept
        
        Args:
            entity_id: 实体ID（实际为Concept ID）
            patch: 要更新的字段字典
        """
        return self.update_concept(entity_id, patch)

    # ==================== 持久化 ====================
    def save(self, base_dir: Optional[str] = None) -> None:
        """保存索引到磁盘（线程安全）
        
        使用锁确保同时只有一个保存操作，避免文件写入冲突。
        """
        # 使用非阻塞锁：如果正在保存，跳过本次保存（避免阻塞索引过程）
        if not self._save_lock.acquire(blocking=False):
            print("[CogMem] 保存操作正在进行中，跳过本次保存请求")
            return
        
        try:
            base = base_dir or self.workspace_dir
            gpath, ipath, eidspath = build_paths(
                base, self.cfg.graph_file, self.cfg.faiss_index_file, self.cfg.entity_store_file
            )
            # 变更日志路径
            import os
            changelog_path = os.path.join(base, self.cfg.graph_changelog_file)
            
            print("[CogMem] 开始保存索引...")
            print(f"  - 保存图结构...")
            self.graph.save(gpath, changelog_path=changelog_path)
            print(f"  - 保存向量索引...")
            
            # TypedVectorStore保存到目录
            if isinstance(self.vstore, TypedVectorStore):
                typed_index_dir = os.path.join(base, "typed_indices")
                self.vstore.save(typed_index_dir)
            else:
                # 向后兼容：普通VectorStore保存到文件
                self.vstore.save(ipath, eidspath)
            
            # 片段索引
            if self.chunk_store:
                print(f"  - 保存片段索引...")
                self.chunk_store.save(
                    os.path.join(base, self.cfg.chunk_index_file),
                    os.path.join(base, self.cfg.chunk_meta_file),
                )
            
            # EvidenceStore（已处理的文档ID）
            print(f"  - 保存证据存储...")
            evidence_store_path = os.path.join(base, self.cfg.evidence_store_file)
            self.indexer.evidence.save(evidence_store_path)
            
            # 导出知识图谱为JSON格式（便于手工检查）
            try:
                graph_json_path = os.path.join(base, "knowledge_graph.json")
                self.graph.export_json(graph_json_path, indent=2)
                print(f"  - 导出知识图谱JSON: {graph_json_path}")
            except Exception as e:
                print(f"  [WARN] 导出知识图谱JSON失败（不影响主流程）: {e}")
            
            print("[CogMem] [OK] 索引保存完成")
        except Exception as e:
            print(f"[CogMem] [ERROR] 保存失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self._save_lock.release()

    def load(self, base_dir: Optional[str] = None) -> None:
        """从磁盘加载索引
        
        Args:
            base_dir: 索引目录，如果为None则使用workspace_dir
        """
        base = base_dir or self.workspace_dir
        gpath, ipath, eidspath = build_paths(
            base, self.cfg.graph_file, self.cfg.faiss_index_file, self.cfg.entity_store_file
        )
        # 变更日志路径
        import os
        changelog_path = os.path.join(base, self.cfg.graph_changelog_file)
        self.graph.load(gpath, changelog_path=changelog_path)
        
        # TypedVectorStore从目录加载
        try:
            if isinstance(self.vstore, TypedVectorStore):
                typed_index_dir = os.path.join(base, "typed_indices")
                if os.path.exists(typed_index_dir):
                    self.vstore.load(typed_index_dir)
                else:
                    print(f"[CogMem] Warning: Typed indices directory not found: {typed_index_dir}")
                    print(f"[CogMem] Index will be empty. You may need to rebuild it.")
            else:
                # 向后兼容：普通VectorStore从文件加载
                self.vstore.load(ipath, eidspath)
        except ValueError as e:
            # 维度不匹配错误
            print(f"[CogMem] [ERROR] Failed to load index: {e}")
            print(f"[CogMem] The index dimension doesn't match the current embedding model.")
            print(f"[CogMem] Please rebuild the index or switch to the correct embedding model.")
            raise
        # 片段索引
        if self.chunk_store:
            self.chunk_store.load(
                os.path.join(base, self.cfg.chunk_index_file),
                os.path.join(base, self.cfg.chunk_meta_file),
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
        
        # 检查图结构文件
        if not os.path.exists(gpath):
            return False
            
        # 检查向量索引（兼容旧版单文件和新版分片目录）
        typed_index_dir = os.path.join(base, "typed_indices")
        if os.path.exists(typed_index_dir):
            # 如果是分片目录，至少要存在目录
            return True
        
        # 否则检查旧版文件
        return os.path.exists(ipath) and os.path.exists(eidspath)

    def add_documents_incremental(self, docs: Iterable[Doc], force_rebuild: bool = False) -> List[Concept]:
        """增量索引：若已有索引且force_rebuild=False，只处理新文档；否则全量重建
        
        Args:
            docs: Documents to add
            force_rebuild: Whether to force a full rebuild
        """
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
            return self.add_documents(docs, incremental=False)

    def consolidate_memory(
        self,
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
    ) -> Dict:
        """Run memory consolidation to convert episodic memories (Events) into semantic memories (Facts)
        
        This method processes the existing graph, clusters similar events by concept,
        and synthesizes them into high-level Facts using LLM based on CLS theory.
        
        Args:
            min_events: Minimum number of events required to create a Fact (default: 3)
            archive_original_events: Whether to mark original events as archived (default: False)
            merge_concepts: Whether to merge synonymous concepts (default: True)
            deduplicate_facts: Whether to deduplicate facts (default: True)
            reconcile_claims: Whether to reconcile claims (default: True)
            concept_similarity_threshold: Threshold for concept similarity (default: 0.90)
            concept_high_threshold: High similarity threshold for full concept merge (default: 0.95)
            fact_high_threshold: High similarity threshold for full fact merge (default: 0.95)
            max_supported_by_per_fact: Maximum SUPPORTED_BY edges per fact (default: 5)
            cleanup_archived: Whether to delete archived nodes after consolidation (default: True)
        
        Returns:
            Statistics dictionary with consolidation results
        
        Example:
            >>> cm = CogMem(workspace_dir="./workspace")
            >>> cm.load()
            >>> stats = cm.consolidate_memory(min_events=3)
            >>> print(f"Created {stats['facts_created']} facts from {stats['events_clustered']} events")
        """
        from .memory.consolidator import MemoryConsolidator
        
        consolidator = MemoryConsolidator(
            config=self.cfg,
            graph=self.graph,
            vector_store=self.vstore,
            embedding_encoder=self._embed,
            evidence_store=self.indexer.evidence  # P0: Use indexer.evidence (single source of truth)
        )
        
        stats = consolidator.run_consolidation(
            source_workspace=None,  # Use current graph
            target_workspace=None,  # Modify in place
            min_events=min_events,
            archive_original_events=archive_original_events,
            merge_concepts=merge_concepts,
            deduplicate_facts=deduplicate_facts,
            reconcile_claims=reconcile_claims,
            concept_similarity_threshold=concept_similarity_threshold,
            concept_high_threshold=concept_high_threshold,
            fact_high_threshold=fact_high_threshold,
            max_supported_by_per_fact=max_supported_by_per_fact,
            cleanup_archived=cleanup_archived
        )
        
        # Save the updated graph
        self.save()
        
        return stats
