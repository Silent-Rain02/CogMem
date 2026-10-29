from __future__ import annotations

import hashlib
import json
from copy import deepcopy
import re
from datetime import datetime, timedelta
from typing import Iterable, List, Tuple, Dict, Optional, Any

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..config import CogMemConfig
from ..models.llm_client import create_llm_client
from ..models.embedding import EmbeddingEncoder
from .chunker import ChunkerFactory
from ..storage.schema import (
    Doc, Concept, Metadata, Relation, Person, Event, Fact, Claim, NodeTypeEnum, MemoryLifecycle
)
from ..storage.graph_store import GraphStore
from ..storage.vector_store import VectorStore
from ..storage.typed_vector_store import TypedVectorStore
from ..storage.evidence_store import EvidenceStore
from ..storage.chunk_store import ChunkStore, Chunk
from ..utils.json_utils import safe_json_loads
from ..utils.temporal import normalize_event_time
from ..utils.concept_manager import ConceptManager, get_default_manager
from ..prompt import ATOMIC_EXTRACTION_SYSTEM, ATOMIC_EXTRACTION_USER
from .entity_matcher import EntityMatcher

def _make_id(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()

def _resolve_relative_time(time_raw: str, session_time: str) -> Optional[str]:
    return normalize_event_time(time_raw, session_time)

class Indexer:
    """文档索引器：负责文档切分、实体抽取、关系抽取、向量索引和图存储"""

    def __init__(self, cfg: CogMemConfig, graph: GraphStore, vstore: VectorStore, chunk_store: ChunkStore | None = None, cogmem=None) -> None:
        self._cfg = cfg
        self._graph = graph
        # 支持新旧两种向量存储
        if isinstance(vstore, TypedVectorStore):
            self._typed_vstore = vstore
            self._vstore = None  # TypedVectorStore不需要_vstore
        else:
            self._vstore = vstore
            self._typed_vstore = None
        self._llm = create_llm_client(cfg)
        self._embed = EmbeddingEncoder(cfg)
        self._evidence = EvidenceStore()
        self._chunk_store = chunk_store
        self._cogmem = cogmem  # Reference to CogMem instance for saving
        # Concept manager for normalizing concepts
        self._concept_manager = get_default_manager()
        # Check if person-centered extraction is enabled
        self._use_person_centered = getattr(cfg.extraction, 'use_person_centered', False)

    def _batch_iter(self, items: List[str], batch_size: int) -> Iterable[List[str]]:
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]
    
    def _get_effective_concurrency(self) -> int:
        """根据provider动态决定并发度，避免资源竞争和rate limit
        
        策略：
        - Ollama本地推理：低并发（2），避免GPU/CPU竞争
        - 云端API：适度并发（4），避免rate limit
        - 其他：使用配置值
        
        Returns:
            有效的并发数
        """
        provider = self._cfg.model.provider.lower()
        
        if provider == "ollama":
            # 本地推理：低并发避免显存/CPU竞争
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            return min(4, max(1, cpu_count // 4))
        elif provider in ["bailian", "openai", "azure"]:
            # 云端API：适度并发避免rate limit
            return min(4, self._cfg.runtime.max_concurrency)
        else:
            # 其他provider：使用配置值
            return self._cfg.runtime.max_concurrency

    def add_documents(self, docs: Iterable[Doc], incremental: bool = False) -> None:
        """
        统一的文档索引入口。
        通过 person-centered, event-mediated 的方式处理文档。

        Args:
            docs: 文档迭代器
            incremental: 是否执行增量索引
        """
        docs_to_process = list(docs)

        # Validate source identity before incremental filtering or any writes.
        # One ID cannot identify two different originals in the same batch.
        source_texts = {}
        for doc in docs_to_process:
            if doc.id in source_texts and source_texts[doc.id] != doc.text:
                raise ValueError(
                    f"Conflicting source document id: {doc.id!r}. "
                    "Use distinct document IDs or separate source namespaces/workspaces."
                )
            source_texts[doc.id] = doc.text

        if incremental:
            processed_doc_ids = self._evidence.get_processed_doc_ids()
            docs_to_process = [doc for doc in docs_to_process if doc.id not in processed_doc_ids]
            if not docs_to_process:
                print("No new documents to process.")
                return

        print(f"Starting to process {len(docs_to_process)} documents...")

        all_nodes = []
        all_relations = []
        chunks_for_embedding = []

        def process_doc(doc: Doc) -> Tuple[List, List, List[Chunk]]:
            """处理单个文档"""
            try:
                # 1. 切分
                chunker = ChunkerFactory.create(doc, self._cfg)
                chunk_texts = chunker.chunk(doc)

                # 提取 session_id 和 turn_id
                session_id = None
                turn_id = None
                if doc.id.startswith("dialog-") or doc.id.startswith("turn-"):
                    parts = doc.id.split("-")
                    for part in parts:
                        if part.startswith("session_"):
                            session_id = part
                        elif ":" in part:
                            turn_id = part

                doc_nodes = []
                doc_relations = []
                doc_chunks = []

                # CRITICAL FIX: 对于已按session拼好的文档，累积所有chunks的提取结果，然后一次性构建图
                # 这样可以减少日志输出，每个文档只打印一次
                # 判断标准：如果doc.id包含session_，说明已经按session拼好了
                is_session_doc = "session_" in doc.id
                
                if is_session_doc and len(chunk_texts) > 1:
                    # 对于已拼好的session文档，如果被切分成多个chunks（不应该发生，但容错处理）
                    # 累积所有chunks的提取结果，然后一次性构建图
                    all_extraction_results = []
                    for chunk_text in chunk_texts:
                        extraction_result = self._extract_structured_from_chunk(
                            chunk=chunk_text,
                            doc_id=doc.id,
                            session_id=session_id,
                            turn_id=turn_id
                        )
                        all_extraction_results.append(extraction_result)
                        # 记录 chunk 用于后续的向量化
                        chunk_id = f"chunk-{doc.id}-{_make_id(chunk_text)}"
                        doc_chunks.append(Chunk(id=chunk_id, text=chunk_text, doc_id=doc.id, turn_id=turn_id))
                    
                    # 合并所有chunks的提取结果
                    merged_result = {
                        "events": [],
                        "facts": [],
                        "claims": [],
                        "entities": []  # 保持兼容性
                    }
                    for extraction_result in all_extraction_results:
                        if isinstance(extraction_result, dict):
                            merged_result["events"].extend(extraction_result.get("events", []))
                            merged_result["facts"].extend(extraction_result.get("facts", []))
                            merged_result["claims"].extend(extraction_result.get("claims", []))
                            merged_result["entities"].extend(extraction_result.get("entities", []))
                    
                    # 一次性构建图（只打印一次日志）
                    # Extract session_date and subject_title from the first chunk
                    session_date = None
                    subject_title = None
                    if chunk_texts:
                        session_date = self._parse_session_date(chunk_texts[0])
                        subject_title = self._parse_subject(chunk_texts[0])
                    
                    nodes, relations = self._build_event_mediated_graph(
                        merged_result, session_id, session_time=session_date, session_date=session_date, subject_title=subject_title, source_id=doc.id
                    )
                    doc_nodes.extend(nodes)
                    doc_relations.extend(relations)
                else:
                    # 对于普通文档或单个chunk的session文档，累积所有chunks的提取结果，然后一次性构建图
                    # 这样可以减少日志输出，每个文档只打印一次
                    all_extraction_results = []
                    for chunk_text in chunk_texts:
                        # 2. 结构化抽取
                        extraction_result = self._extract_structured_from_chunk(
                            chunk=chunk_text,
                            doc_id=doc.id,
                            session_id=session_id,
                            turn_id=turn_id
                        )
                        all_extraction_results.append(extraction_result)
                        
                        # 记录 chunk 用于后续的向量化
                        chunk_id = f"chunk-{doc.id}-{_make_id(chunk_text)}"
                        doc_chunks.append(Chunk(id=chunk_id, text=chunk_text, doc_id=doc.id, turn_id=turn_id))
                    
                    # 合并所有chunks的提取结果
                    merged_result = {
                        "events": [],
                        "facts": [],
                        "claims": [],
                        "entities": []
                    }
                    for extraction_result in all_extraction_results:
                        if isinstance(extraction_result, dict):
                            merged_result["events"].extend(extraction_result.get("events", []))
                            merged_result["facts"].extend(extraction_result.get("facts", []))
                            merged_result["claims"].extend(extraction_result.get("claims", []))
                            merged_result["entities"].extend(extraction_result.get("entities", []))
                    
                    # 一次性构建图（只打印一次日志）
                    # Extract session_date and subject_title from the first chunk
                    session_date = None
                    subject_title = None
                    if chunk_texts:
                        session_date = self._parse_session_date(chunk_texts[0])
                        subject_title = self._parse_subject(chunk_texts[0])
                    
                    nodes, relations = self._build_event_mediated_graph(
                        merged_result, session_id, session_time=session_date, session_date=session_date, subject_title=subject_title, source_id=doc.id
                    )
                    doc_nodes.extend(nodes)
                    doc_relations.extend(relations)
                    
                    # 调试：如果节点为空，打印详细信息（仅对普通文档）
                    if not is_session_doc and not doc_nodes and chunk_texts:
                        # 只在调试模式下打印，避免过多输出
                        if self._cfg.runtime.max_concurrency == 1:
                            print(f"[!]  警告：文档 {doc.id} 未提取到节点")
                            print(f"   合并后的提取结果: events={len(merged_result.get('events', []))}, "
                                  f"facts={len(merged_result.get('facts', []))}, "
                                  f"claims={len(merged_result.get('claims', []))}, "
                                  f"entities={len(merged_result.get('entities', []))}")
                
                # 对于session文档，如果节点为空，打印一次警告
                if is_session_doc and not doc_nodes:
                    if self._cfg.runtime.max_concurrency == 1:
                        print(f"[!]  警告：session文档 {doc.id} 未提取到节点")
                        # 如果有chunk_texts，打印第一个chunk的预览用于调试
                        if chunk_texts:
                            print(f"   第一个chunk预览: {chunk_texts[0][:200]}...")

                # 创建 PersonFacet 节点（如果启用）
                persons = [n for n in doc_nodes if isinstance(n, Person)]
                facts = [n for n in doc_nodes if isinstance(n, Fact)]

                # PersonFacet 已废弃，不再创建
                
                # 将证据绑定收紧到节点相关 chunk，避免整个 session 被粗暴挂到每个节点上。
                self._attach_relevant_evidence_chunks(doc_nodes, doc_chunks)

                return doc_nodes, doc_relations, doc_chunks
            except Exception as e:
                import traceback
                # 安全地获取doc.id，防止doc不是预期的对象
                doc_id = "unknown"
                try:
                    if hasattr(doc, 'id'):
                        doc_id = doc.id
                    elif hasattr(doc, '__str__'):
                        doc_id = str(doc)
                except Exception:
                    pass  # 保持 doc_id = "unknown"
                
                print(f"[!]  处理文档 {doc_id} 时出错: {type(e).__name__}: {str(e)}")
                
                # 详细调试信息
                if "'text'" in str(e):
                    print(f"     Doc类型: {type(doc)}")
                    try:
                        if hasattr(doc, '__dict__'):
                            print(f"     Doc属性: {list(doc.__dict__.keys())}")
                        if hasattr(doc, 'text'):
                            print(f"     doc.text 类型: {type(doc.text)}")
                            print(f"     doc.text 值: {repr(doc.text)[:100]}")
                    except Exception:
                        pass
                
                # 始终打印完整堆栈（用于调试）
                traceback.print_exc()
                return [], [], []

        # 使用 ThreadPoolExecutor 并行处理文档
        # 根据provider动态调整并发数
        max_workers = max(1, self._get_effective_concurrency())
        print(f"  使用并发数: {max_workers} (provider: {self._cfg.model.provider})")
        
        # 记录需要重试的文档（超时或其他错误）
        docs_to_retry = []
        doc_to_future_map = {}  # 用于追踪文档和future的映射
        
        # 定期保存的计数器（使用线程安全的计数器）
        import threading
        processed_docs_count = 0
        processed_docs_lock = threading.Lock()
        
        # 增加超时时间：文档处理可能包含多个LLM调用，需要更长的超时时间
        # 对于慢速的ollama，设置更长的超时时间（600秒 = 10分钟）
        doc_timeout = max(600.0, self._cfg.runtime.request_timeout_s * 5)
        
        # 最大重试次数
        max_retries = 5
        
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            # 第一轮：提交所有文档
            futures = []
            for doc in docs_to_process:
                future = ex.submit(process_doc, doc)
                futures.append(future)
                doc_to_future_map[future] = doc
            
            retry_round = 0
            while retry_round <= max_retries:
                if retry_round > 0:
                    print(f"\n[重试轮次 {retry_round}] 正在重试 {len(docs_to_retry)} 个文档（无超时限制）...")
                    # 重新提交需要重试的文档
                    futures = []
                    for doc in docs_to_retry:
                        future = ex.submit(process_doc, doc)
                        futures.append(future)
                        doc_to_future_map[future] = doc
                    docs_to_retry = []
                
                try:
                    for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Extracting from documents (round {retry_round + 1})"):
                        try:
                            # 第一轮使用超时，重试轮次不使用超时（None表示无超时限制）
                            timeout = None if retry_round > 0 else doc_timeout
                            nodes, relations, chunks = fut.result(timeout=timeout)
                            
                            # 检查是否成功处理（有节点或关系或chunks）
                            current_count = 0
                            if nodes or relations or chunks:
                                all_nodes.extend(nodes)
                                all_relations.extend(relations)
                                chunks_for_embedding.extend(chunks)
                                # ===== 启用 EvidenceStore：标记文档已处理 =====
                                doc = doc_to_future_map.get(fut)
                                if doc:
                                    self._evidence.mark_doc_processed(doc.id)
                                with processed_docs_lock:
                                    processed_docs_count += 1
                                    current_count = processed_docs_count
                            else:
                                # 如果返回空结果，记录需要重试
                                doc = doc_to_future_map.get(fut)
                                if doc:
                                    docs_to_retry.append(doc)
                                    print(f"[!]  文档 {doc.id} 返回空结果，将重试")
                            
                        except Exception as e:
                            if isinstance(e, KeyboardInterrupt):
                                raise
                            # 捕获超时或其他错误，记录需要重试的文档
                            error_msg = str(e)
                            doc = doc_to_future_map.get(fut)
                            if doc:
                                if "timeout" in error_msg.lower() or "ReadTimeout" in error_msg or "TimeoutError" in error_msg:
                                    print(f"[!]  文档 {doc.id} 处理超时，将重试（无超时限制）")
                                else:
                                    print(f"[!]  文档 {doc.id} 处理出错: {error_msg[:100]}，将重试")
                                docs_to_retry.append(doc)
                            else:
                                print(f"[!]  获取文档处理结果时出错: {error_msg[:100]}")
                
                except KeyboardInterrupt:
                    # Ctrl+C 中断时，保存已处理的节点
                    print("\n[!]  收到中断信号，正在保存已处理的节点...")
                    # 取消所有未完成的任务
                    for future in futures:
                        if not future.done():
                            future.cancel()
                    
                    # 保存已处理的节点（即使未完成所有文档）
                    if all_nodes and self._cogmem:
                        try:
                            print(f"  正在保存已处理的 {len(all_nodes)} 个节点...")
                            # 去重节点
                            temp_unique_nodes_dict = {}
                            for node in all_nodes:
                                temp_unique_nodes_dict[node.id] = node
                            temp_unique_nodes = list(temp_unique_nodes_dict.values())
                            
                            # 添加节点到图（如果还没有添加）
                            nodes_to_add = [node for node in temp_unique_nodes if node.id not in self._graph._g]
                            if nodes_to_add:
                                self._graph.add_nodes(nodes_to_add)
                            
                            # 添加关系
                            if all_relations:
                                temp_node_ids = {n.id for n in temp_unique_nodes}
                                temp_filtered_relations = [
                                    r for r in all_relations
                                    if r.head_id in temp_node_ids and r.tail_id in temp_node_ids
                                ]
                                if temp_filtered_relations:
                                    # 去重关系
                                    temp_unique_relations_dict = {}
                                    for r in temp_filtered_relations:
                                        key = (r.head_id, r.tail_id, r.rel_type)
                                        if key not in temp_unique_relations_dict or r.confidence > temp_unique_relations_dict[key].confidence:
                                            temp_unique_relations_dict[key] = r
                                    temp_unique_relations = list(temp_unique_relations_dict.values())
                                    
                                    # 只添加新关系
                                    for r in temp_unique_relations:
                                        if not self._graph._g.has_edge(r.head_id, r.tail_id, key=r.rel_type):
                                            self._graph.add_relations([r])
                            
                            # 保存检查点
                            self._cogmem.save()
                            print(f"  [OK] 已保存检查点（图中有 {self._graph.num_entities()} 个节点）")
                        except Exception as save_err:
                            print(f"  [ERR] 保存检查点失败: {str(save_err)}")
                    
                    raise
                
                # 如果没有需要重试的文档，退出循环
                if not docs_to_retry:
                    break
                
                retry_round += 1
                
                # 如果达到最大重试次数，打印警告但继续
                if retry_round > max_retries:
                    print(f"\n[!]  警告：已达到最大重试次数 {max_retries}，仍有 {len(docs_to_retry)} 个文档未成功处理")
                    print(f"    这些文档将被跳过，但已处理的文档已保存")
                    break
            
            # 如果还有未处理的文档，最后尝试一次（无超时限制）
            if docs_to_retry:
                print(f"\n[最终重试] 正在处理剩余的 {len(docs_to_retry)} 个文档（无超时限制，单线程处理）...")
                for doc in tqdm(docs_to_retry, desc="最终重试"):
                    try:
                        nodes, relations, chunks = process_doc(doc)
                        if nodes or relations or chunks:
                            all_nodes.extend(nodes)
                            all_relations.extend(relations)
                            chunks_for_embedding.extend(chunks)
                            # ===== 启用 EvidenceStore：标记文档已处理 =====
                            self._evidence.mark_doc_processed(doc.id)
                            processed_docs_count += 1
                            print(f"  [OK] 文档 {doc.id} 最终重试成功")
                        else:
                            print(f"  [ERR] 文档 {doc.id} 最终重试仍返回空结果")
                    except Exception as e:
                        print(f"  [ERR] 文档 {doc.id} 最终重试失败: {str(e)[:100]}")
            
            # 如果还有未处理的文档，打印最终统计
            if docs_to_retry:
                print(f"\n[!]  警告：仍有 {len(docs_to_retry)} 个文档未能成功处理")
                print(f"    已成功处理 {processed_docs_count}/{len(docs_to_process)} 个文档")
                # 取消所有未完成的任务
                for future in futures:
                    if not future.done():
                        future.cancel()
                
                # 保存已处理的节点（即使未完成所有文档）
                if all_nodes and self._cogmem:
                    try:
                        print(f"  正在保存已处理的 {len(all_nodes)} 个节点...")
                        # 去重节点
                        temp_unique_nodes_dict = {}
                        for node in all_nodes:
                            temp_unique_nodes_dict[node.id] = node
                        temp_unique_nodes = list(temp_unique_nodes_dict.values())
                        
                        # 添加节点到图（如果还没有添加）
                        nodes_to_add = [node for node in temp_unique_nodes if node.id not in self._graph._g]
                        if nodes_to_add:
                            self._graph.add_nodes(nodes_to_add)
                        
                        # 添加关系
                        if all_relations:
                            temp_node_ids = {n.id for n in temp_unique_nodes}
                            temp_filtered_relations = [
                                r for r in all_relations
                                if r.head_id in temp_node_ids and r.tail_id in temp_node_ids
                            ]
                            if temp_filtered_relations:
                                # 去重关系
                                temp_unique_relations_dict = {}
                                for r in temp_filtered_relations:
                                    key = (r.head_id, r.tail_id, r.rel_type)
                                    if key not in temp_unique_relations_dict or r.confidence > temp_unique_relations_dict[key].confidence:
                                        temp_unique_relations_dict[key] = r
                                temp_unique_relations = list(temp_unique_relations_dict.values())
                                
                                # 只添加新关系
                                for r in temp_unique_relations:
                                    if not self._graph._g.has_edge(r.head_id, r.tail_id, key=r.rel_type):
                                        self._graph.add_relations([r])
                        
                        # 保存检查点
                        self._cogmem.save()
                        print(f"  [OK] 已保存检查点（图中有 {self._graph.num_entities()} 个节点）")
                    except Exception as save_err:
                        print(f"  [ERR] 保存检查点失败: {str(save_err)}")

        # 4. 去重和存图（在循环外批量处理）
        # 去重 nodes（使用EntityMatcher进行智能合并）
        entity_matcher = EntityMatcher(self._graph)
        unique_nodes_dict = {}
        
        for node in all_nodes:
            if isinstance(node, Person):
                # 使用EntityMatcher查找或创建Person
                actual_id, merged_person = entity_matcher.find_or_create_person(node)
                unique_nodes_dict[actual_id] = merged_person
            else:
                # 其他节点类型保持原有逻辑
                unique_nodes_dict[node.id] = node
        
        unique_nodes = list(unique_nodes_dict.values())
        
        # 打印匹配统计信息
        stats = entity_matcher.get_stats()
        if stats["matched_by_name"] + stats["matched_by_alias"] > 0:
            print(f"  [EntityMatcher] 匹配统计:")
            print(f"    - 按名称匹配: {stats['matched_by_name']}")
            print(f"    - 按别名匹配: {stats['matched_by_alias']}")
            print(f"    - 创建新Person: {stats['created_new']}")
            print(f"    - 合并次数: {stats['merged_count']}")

        # Episode 已废弃，不再创建
        episode_nodes, episode_relations = [], []

        # 4.2 存图（nodes）- 确保所有节点都已添加
        if unique_nodes:
            print(f"  正在添加 {len(unique_nodes)} 个节点到图...")
            self._graph.add_nodes(unique_nodes)
            print(f"  [OK] 节点添加完成（图中共有 {self._graph.num_entities()} 个节点）")

        # 4.3 去重并合并所有 relations（原始关系 + episode 关系）
        all_relations.extend(episode_relations)

        # ROOT FIX: 改进关系去重逻辑，合并元数据而不是只保留置信度最高的
        # 对于多重边（multigraph），保留所有关系；对于相同的关系，合并元数据
        unique_relations_dict = {}
        for r in all_relations:
            key = (r.head_id, r.tail_id, r.rel_type)
            
            if key not in unique_relations_dict:
                # 新关系，直接添加
                unique_relations_dict[key] = r
            else:
                # 关系已存在，合并元数据
                existing = unique_relations_dict[key]
                
                # 选择置信度更高的关系作为基础
                if r.confidence > existing.confidence:
                    base_relation = r
                    other_relation = existing
                else:
                    base_relation = existing
                    other_relation = r
                
                # 合并元数据：保留更详细的信息
                # 合并 timestamp（保留更早的时间，如果有的话）
                if base_relation.timestamp and other_relation.timestamp:
                    # 选择更早的时间（更准确）
                    try:
                        from datetime import datetime
                        base_dt = datetime.fromisoformat(base_relation.timestamp.replace('Z', '+00:00'))
                        other_dt = datetime.fromisoformat(other_relation.timestamp.replace('Z', '+00:00'))
                        merged_timestamp = min(base_dt, other_dt).isoformat()
                    except:
                        merged_timestamp = base_relation.timestamp
                else:
                    merged_timestamp = base_relation.timestamp or other_relation.timestamp
                
                # 合并 edge_title（保留更详细的）
                merged_edge_title = base_relation.edge_title or other_relation.edge_title
                if base_relation.edge_title and other_relation.edge_title:
                    # 如果两个都有，选择更长的（更详细）
                    if len(other_relation.edge_title) > len(base_relation.edge_title):
                        merged_edge_title = other_relation.edge_title
                
                # 合并 edge_time（保留更早的时间）
                merged_edge_time = base_relation.edge_time or other_relation.edge_time
                if base_relation.edge_time and other_relation.edge_time:
                    try:
                        from datetime import datetime
                        base_dt = datetime.fromisoformat(base_relation.edge_time.replace('Z', '+00:00'))
                        other_dt = datetime.fromisoformat(other_relation.edge_time.replace('Z', '+00:00'))
                        merged_edge_time = min(base_dt, other_dt).isoformat()
                    except:
                        merged_edge_time = base_relation.edge_time
                
                # 合并 weight（取平均值，表示多次出现的强度）
                merged_weight = (base_relation.weight + other_relation.weight) / 2.0
                
                # 合并 confidence（取最大值，表示最高置信度）
                merged_confidence = max(base_relation.confidence, other_relation.confidence)
                
                # 创建合并后的关系
                merged_relation = Relation(
                    head_id=base_relation.head_id,
                    tail_id=base_relation.tail_id,
                    rel_type=base_relation.rel_type,
                    weight=merged_weight,
                    confidence=merged_confidence,
                    metadata=base_relation.metadata,  # 保留基础关系的元数据
                    version=max(base_relation.version, other_relation.version),
                    timestamp=merged_timestamp,
                    edge_title=merged_edge_title,
                    edge_time=merged_edge_time
                )
                
                unique_relations_dict[key] = merged_relation
        
        unique_relations = list(unique_relations_dict.values())

        # 4.4 存图（relations）- 确保所有关系都已添加
        if unique_relations:
            node_ids = {n.id for n in unique_nodes}
            filtered_relations = [
                r for r in unique_relations
                if r.head_id in node_ids and r.tail_id in node_ids
            ]
            if filtered_relations:
                print(f"  正在添加 {len(filtered_relations)} 个关系到图...")
                self._graph.add_relations(filtered_relations)
                print(f"  [OK] 关系添加完成（图中共有 {self._graph.num_relations()} 个关系）")

        # 5. 批量嵌入节点
        try:
            if unique_nodes:
                print(f"  正在生成节点向量（共 {len(unique_nodes)} 个节点）...")
                self._batch_embed_nodes(unique_nodes)
                print(f"  [OK] 节点向量化完成")
        except KeyboardInterrupt:
            print("\n[!]  节点嵌入被中断")
            # 即使嵌入中断，节点已经添加到图中，可以继续
            if self._cogmem:
                try:
                    print("  正在保存已添加的节点（不含向量）...")
                    self._cogmem.save()
                    print("  [OK] 节点已保存（向量将在下次索引时补充）")
                except Exception as save_err:
                    print(f"  [ERR] 保存失败: {str(save_err)}")
            raise

        # 6. 批量嵌入 Chunks（条件化：仅在 enable_chunk_embedding=True 时执行）
        # P0 FIX: 即使enable_chunk_embedding=False，也要索引chunks用于Naive RAG fallback
        if self._chunk_store and chunks_for_embedding:
            if self._cfg.chunk.enable_chunk_embedding:
                # 原有逻辑：Embedding chunks 并存入 VectorStore
                try:
                    chunk_texts = [c.text for c in chunks_for_embedding]
                    print(f"  正在生成文档片段向量（共 {len(chunk_texts)} 个片段）...")

                    # 分批处理chunks以显示进度
                    batch_size = 50
                    for i in tqdm(range(0, len(chunk_texts), batch_size), desc="片段向量化进度"):
                        batch_texts = chunk_texts[i:i+batch_size]
                        batch_chunks = chunks_for_embedding[i:i+batch_size]
                        batch_vectors = self._embed.encode(batch_texts)

                        if batch_vectors:
                            self._chunk_store.add(batch_chunks, batch_vectors)

                    print(f"  [OK] 文档片段向量化完成")
                except KeyboardInterrupt:
                    print("\n[!]  Chunk嵌入被中断")
                    # Chunk 嵌入中断不影响主索引
                    pass
            else:
                # P0 FIX: Pure Graph Mode下仍然索引chunks用于Naive RAG fallback
                # 即使不启用chunk embedding，也要生成向量并存储，确保fallback机制可用
                try:
                    chunk_texts = [c.text for c in chunks_for_embedding]
                    print(f"  [Indexer] Pure Graph Mode: 为Naive RAG fallback生成chunk向量（共 {len(chunk_texts)} 个片段）...")

                    # 分批处理chunks以显示进度
                    batch_size = 50
                    for i in tqdm(range(0, len(chunk_texts), batch_size), desc="Fallback向量化进度"):
                        batch_texts = chunk_texts[i:i+batch_size]
                        batch_chunks = chunks_for_embedding[i:i+batch_size]
                        batch_vectors = self._embed.encode(batch_texts)

                        if batch_vectors:
                            self._chunk_store.add(batch_chunks, batch_vectors)

                    print(f"  [OK] Chunk向量化完成（用于Naive RAG fallback）")
                except KeyboardInterrupt:
                    print("\n[!]  Chunk嵌入被中断")
                    pass
                except Exception as e:
                    # 如果fallback索引失败，记录警告但不影响主索引
                    print(f"  [!] 警告：Chunk fallback索引失败: {e}")
                    print(f"  [Indexer] 系统将无法使用Naive RAG fallback机制")
        
        # 7. 最终保存（确保所有数据都已持久化）
        if self._cogmem:
            try:
                # 获取向量索引大小（兼容TypedVectorStore和VectorStore）
                total_vectors = 0
                if self._typed_vstore:
                    # TypedVectorStore：统计所有类型的向量数
                    type_stats = self._typed_vstore.get_type_stats()
                    total_vectors = sum(type_stats.values())
                elif self._vstore:
                    # VectorStore：直接获取大小
                    total_vectors = self._vstore.size()
                
                print(f"  正在执行最终保存（图中有 {self._graph.num_entities()} 个节点，向量索引中有 {total_vectors} 个向量）...")
                self._cogmem.save()
                print(f"  [OK] 最终保存完成")
            except Exception as save_err:
                print(f"  [WARN] 最终保存失败: {str(save_err)}")
                # 不抛出异常，允许继续执行

    def _extract_structured_from_chunk(
        self,
        chunk: str,
        doc_id: str,
        session_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        question_date_iso: Optional[str] = None
    ) -> Dict:
        """Extract structured information using ATOMIC_EXTRACTION prompts

        Uses the new cognitive classification system to distinguish between:
        - Events (past tense actions)
        - Facts (objective attributes & self-reported preferences)
        - Claims (subjective opinions about others)
        - Concepts (normalized abstract ideas)
        """
        # Extract session date from doc_id if available
        session_date = self._parse_session_date(chunk)
        if session_date:
            question_date_iso = session_date

        # Historical text without an anchor must not acquire the indexing date.
        if not question_date_iso:
            question_date_iso = "unknown (do not resolve relative dates)"

        # Build user prompt with the chunk text and session date
        user_prompt = ATOMIC_EXTRACTION_USER.format(
            text=chunk,
            session_date=question_date_iso
        )

        # Call LLM with atomic extraction prompts
        # Note: LLMClient API is chat(system, user, model) -> str
        response = self._llm.chat(
            system=ATOMIC_EXTRACTION_SYSTEM,
            user=user_prompt,
            model=None  # Use default model
        )

        # Parse JSON response (response is already a string)
        extraction_result = safe_json_loads(response)

        # Post-process events: align extraction schema and normalize relative time expressions.
        session_date = session_date or question_date_iso
        if extraction_result and isinstance(extraction_result, dict):
            events = extraction_result.get("events", [])
            normalized_events = []
            for event in events:
                if isinstance(event, dict):
                    title = event.get("title") or event.get("description") or event.get("name")
                    if title and not event.get("title"):
                        event["title"] = str(title).strip()
                    if "concepts" not in event or not isinstance(event.get("concepts"), list):
                        event["concepts"] = []
                    # Keep the extracted expression unchanged. Resolve once at graph construction.
                normalized_events.append(event)
            extraction_result["events"] = normalized_events

        return extraction_result

    def _attach_relevant_evidence_chunks(self, nodes: List[Any], doc_chunks: List[Chunk]) -> None:
        """Register document-level provenance from extraction, never lexical guesses.

        The caller knows which document produced these nodes. Keep every source
        chunk as a recoverable address; query-aware read limits select the view.
        """
        for node in nodes:
            node_id = getattr(node, 'id', None)
            if not node_id:
                continue
            existing = set(self._evidence.get_chunks(node_id, limit=None))
            for chunk in doc_chunks:
                if chunk.text not in existing:
                    self._evidence.add_chunk_for_entity(node_id, chunk.text)
                    existing.add(chunk.text)

    @property
    def evidence(self) -> EvidenceStore:
        return self._evidence

    def _parse_session_date(self, chunk: str) -> Optional[str]:
        """Extract session date from chunk text and convert to ISO format

        Handles natural language dates like "1:56 pm on 8 May, 2023"
        and converts them to ISO format "2023-05-08".

        Args:
            chunk: Text containing "[Session Date: ...]" marker

        Returns:
            ISO date string (YYYY-MM-DD) or None if not found/invalid
        """
        import re
        # datetime is already imported at the top of the file

        match = re.search(r'\[Session Date:\s*([^\]]+)\]', chunk)
        if not match:
            return None

        date_str = match.group(1).strip()

        # Try ISO format first (fast path)
        try:
            # Check if it's already in YYYY-MM-DD format
            if re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
                datetime.strptime(date_str.split(' ')[0], "%Y-%m-%d")
                return date_str.split(' ')[0]
        except:
            pass

        # Parse natural language format: "1:56 pm on 8 May, 2023"
        # Pattern: [time] on [day] [month], [year]
        try:
            # Extract date part (after "on ")
            if ' on ' in date_str:
                date_part = date_str.split(' on ')[1]
            else:
                date_part = date_str

            # Clean up the date part
            date_part = date_part.strip()

            # Remove comma if present
            date_part = date_part.replace(',', '')

            # Parse using datetime.strptime with flexible format
            # Try different month formats
            month_formats = [
                "%d %B %Y",  # 8 May 2023
                "%d %b %Y",  # 8 May 2023 (abbrev)
                "%B %d %Y",  # May 8 2023
                "%b %d %Y",  # May 8 2023 (abbrev)
            ]

            for fmt in month_formats:
                try:
                    dt = datetime.strptime(date_part, fmt)
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue

            # If all else fails, try to extract year, month, day manually
            parts = date_part.split(' ')
            if len(parts) >= 3:
                # Look for a 4-digit year
                year = None
                month = None
                day = None

                for part in parts:
                    if part.isdigit() and len(part) == 4:
                        year = int(part)
                    elif part.isdigit() and int(part) <= 31:
                        day = int(part)
                    elif not month and len(part) > 2:
                        # Try to parse as month name
                        try:
                            # Create a fake date to parse month
                            dt = datetime.strptime(part, "%B")
                            month = dt.month
                        except:
                            try:
                                dt = datetime.strptime(part, "%b")
                                month = dt.month
                            except:
                                pass

                if year and month and day:
                    return f"{year:04d}-{month:02d}-{day:02d}"

        except Exception as e:
            print(f"[WARN] Failed to parse session date '{date_str}': {e}")

        return None

    def _parse_subject(self, chunk: str) -> Optional[str]:
        """从 Chunk 文本中提取 Subject: [Title] 标签"""
        import re
        match = re.search(r'Subject:\s*([^\n\]]+)', chunk)
        if match:
            return match.group(1).strip()
        return None

    def _normalize_relative_time(self, time_str: str, session_date: str) -> Optional[str]:
        return normalize_event_time(time_str, session_date)

    def _build_event_mediated_graph(
        self,
        extraction_result: Dict,
        session_id: Optional[str] = None,
        session_time: Optional[str] = None,
        session_date: Optional[str] = None,
        subject_title: Optional[str] = None,
        source_id: Optional[str] = None
    ) -> Tuple[List, List]:
        """Build graph using the new P-E-C-F architecture with Claims and Concepts

        Processes extraction result with:
        - Events with participants and concepts
        - Facts with subject-predicate-object structure (self-reported)
        - Claims with two-hop structure (source -> claim -> target)
        - Concepts normalized via ConceptManager
        
        **ROOT FIX: Speaker Identity Resolution & Subject Title Resolution**
        """
        extraction_result = deepcopy(extraction_result)
        nodes_to_add = []
        relations_to_add = []

        # Validate extraction_result type
        if not isinstance(extraction_result, dict):
            print(f"  [Indexer] Warning: extraction_result is not a dict (type: {type(extraction_result)}), returning empty graph")
            return nodes_to_add, relations_to_add

        # Extract sections from atomic extraction result
        events_data = extraction_result.get("events", [])
        # 消融实验：如果配置为不在索引时提取Fact，则跳过Fact提取
        if getattr(self._cfg.extraction, 'extract_facts_during_indexing', True):
            facts_data = extraction_result.get("facts", [])
        else:
            facts_data = []  # 不在索引时提取Fact，只在记忆巩固时创建
        claims_data = extraction_result.get("claims", [])

        # ========================================
        # ROOT FIX: Generic Pronoun Resolution (HotpotQA optimization)
        # ========================================
        # 如果提供了 subject_title (从 Subject: 标签解析而来)，
        # 我们将 extraction_result 中所有的通用代词替换为该标题。
        if subject_title:
            generic_pronouns = {
                "he", "she", "it", "they", "them", "his", "her", "him",
                "the film", "the series", "the book", "the person", "the group", 
                "the band", "the team", "the city", "the country", "the state",
                "the director", "the writer", "the actor", "the singer", "the artist"
            }
            
            # 替换事件参与者
            for event_data in events_data:
                participants = event_data.get("participants", [])
                event_data["participants"] = [
                    subject_title if p.lower() in generic_pronouns else p
                    for p in participants
                ]
            
            # 替换事实主体
            for fact_data in facts_data:
                if fact_data.get("subject", "").lower() in generic_pronouns:
                    fact_data["subject"] = subject_title
            
            # 替换 Claim 来源
            for claim_data in claims_data:
                if claim_data.get("source", "").lower() in generic_pronouns:
                    claim_data["source"] = subject_title
                    
        # ========================================
        # ROOT FIX: Speaker Identity Resolution
        # ========================================
        # **Purpose**: Resolve "Speaker" placeholder to actual person name when self-introduction is detected.
        # **Scope**: ONLY applies when explicit self-introduction exists (e.g., "My name is Kevin").
        # **Compatibility**:
        #   - LoCoMo (multi-party): Uses real names ([Caroline], [Melanie]) → No resolution needed
        #   - LongMemEval (human-AI): Uses generic roles (USER, ASSISTANT) → No resolution needed
        #   - Self-introduction: Uses "USER: My name is Kevin" → Resolution: Speaker → Kevin
        
        resolved_speaker_name = None  # Will be set to actual name like "kevin" if self-introduction detected
        
        # Strategy 1: Check for self-introduction patterns in Facts
        # Pattern A: Fact(subject="Speaker", predicate="has_name", object="c_kevin") ← Standard
        # Pattern B: Fact(subject="Speaker", predicate="is_a", object="c_kevin") ← Legacy (backward compatibility)
        for fact_data in facts_data:
            subject = fact_data.get("subject", "")
            predicate = fact_data.get("predicate", "")
            object_name = fact_data.get("object", "")
            
            # Detect self-introduction: "My name is Kevin" -> Fact(subject="Speaker", predicate="has_name"/"is_a", object="c_kevin")
            if subject == "Speaker" and predicate in ["has_name", "is_named", "is_a"] and object_name and isinstance(object_name, str):
                # Extract the name from concept ID (c_kevin -> kevin)
                if object_name.startswith("c_"):
                    candidate_name = object_name[2:].replace("_", " ").strip()
                    
                    # For "has_name"/"is_named", directly accept as name (highest priority)
                    if predicate in ["has_name", "is_named"]:
                        resolved_speaker_name = candidate_name
                        if self._cfg.runtime.max_concurrency <= 1:
                            print(f"  [Indexer] [INFO] Detected Speaker identity: '{resolved_speaker_name}' (from self-introduction: has_name)")
                        break
                    
                    # For "is_a", validate that it's a name (not occupation)
                    # Simple heuristic: If it's a single word and doesn't contain common occupation keywords
                    occupation_keywords = ["engineer", "doctor", "teacher", "manager", "developer", "designer", 
                                          "analyst", "consultant", "assistant", "director", "coordinator", "specialist",
                                          "programmer", "scientist", "researcher", "professor", "instructor", "trainer"]
                    if not any(keyword in candidate_name.lower() for keyword in occupation_keywords):
                        # Additional check: Names are usually short (1-3 words) and capitalized in normalized form
                        word_count = len(candidate_name.split())
                        if word_count <= 3:  # Reasonable name length
                            resolved_speaker_name = candidate_name
                            if self._cfg.runtime.max_concurrency <= 1:
                                print(f"  [Indexer] [INFO] Detected Speaker identity: '{resolved_speaker_name}' (from self-introduction: is_a)")
                            break
        
        # If no self-introduction found, keep Speaker unresolved for this chunk
        
        # Apply resolution: Replace "Speaker" with resolved name in all extraction results
        if resolved_speaker_name:
            # Replace in events.participants
            for event_data in events_data:
                participants = event_data.get("participants", [])
                event_data["participants"] = [
                    resolved_speaker_name if p == "Speaker" else p
                    for p in participants
                ]
            
            # Replace in facts.subject
            for fact_data in facts_data:
                if fact_data.get("subject") == "Speaker":
                    fact_data["subject"] = resolved_speaker_name
            
            # Replace in claims.source
            for claim_data in claims_data:
                if claim_data.get("source") == "Speaker":
                    claim_data["source"] = resolved_speaker_name
        
        # ========================================
        # End of Speaker Identity Resolution
        # ========================================

        # 减少日志输出，在并发模式下只在控制台打印摘要
        if self._cfg.runtime.max_concurrency <= 1:
            print(f"  [Indexer] Building graph: {len(events_data)} events, {len(facts_data)} facts, {len(claims_data)} claims")

        # Step 1: Create Person nodes for all participants
        # Collect all unique person names from events, facts, and claims
        person_names = set()

        # From events participants
        for event_data in events_data:
            for participant in event_data.get("participants", []):
                if participant and isinstance(participant, str):
                    person_names.add(participant)

        # From facts subjects
        for fact_data in facts_data:
            subject = fact_data.get("subject")
            if subject and isinstance(subject, str):
                person_names.add(subject)

        # From claim sources
        for claim_data in claims_data:
            source = claim_data.get("source")
            if source and isinstance(source, str):
                person_names.add(source)

        # Create Person nodes (speaker is special case)
        from ..utils.text_utils import unwrap_person_name, normalize_to_id

        # [ROOT FIX] Define generic pronouns to avoid identity merging
        # These will NEVER be allowed as global person IDs unless resolved.
        generic_names = {
            "he", "she", "it", "they", "them", "his", "her", "him", "their", "its",
            "the film", "the series", "the book", "the person", "the group", 
            "the band", "the team", "the city", "the country", "the state",
            "the director", "the writer", "the actor", "the singer", "the artist",
            "speaker", "listener", "assistant", "user", "someone", "somebody", "anyone"
        }

        person_map = {}  # name -> person_id
        for name in person_names:
            if name and isinstance(name, str):
                # Clean the name - handle cases where LLM returns IDs like "p_jon"
                canonical_name = unwrap_person_name(name)

                # Skip if unwrapping resulted in empty string or unknown_speaker
                if not canonical_name or canonical_name == "unknown_speaker":
                    continue

                # Generate person ID from canonical name
                person_id_normalized = normalize_to_id(canonical_name)
                
                # [ROOT FIX] If the name is generic, make it unique to this session/chunk 
                # to prevent global "p_he" nodes
                if canonical_name.lower() in generic_names:
                    # Use session_id or generate a unique suffix from extraction_result hash
                    # Since doc_id is not available in this method, we use session_id or a hash of events/facts
                    if session_id:
                        unique_suffix = session_id.replace("session_", "")[:8]
                    else:
                        # Generate a hash from extraction result to ensure uniqueness
                        import hashlib
                        result_str = str(extraction_result.get("events", [])) + str(extraction_result.get("facts", []))
                        unique_suffix = hashlib.md5(result_str.encode()).hexdigest()[:8]
                    person_id = f"p_{person_id_normalized}_{unique_suffix}"
                    if self._cfg.runtime.max_concurrency <= 1:
                        print(f"  [Indexer] [INFO] Isolated generic identity: '{canonical_name}' -> {person_id}")
                else:
                    person_id = f"p_{person_id_normalized}"

                # Check if we already have this person by canonical_name
                existing_person = None
                for existing in nodes_to_add:
                    if isinstance(existing, dict):
                        # Already in dict form
                        if existing.get("node_type") == "person" and existing.get("canonical_name") == canonical_name:
                            existing_person = existing
                            break
                    elif hasattr(existing, "canonical_name"):
                        # Still in Pydantic model form
                        if existing.canonical_name == canonical_name:
                            existing_person = existing
                            break

                if existing_person:
                    # Already exists, just map to it
                    person_map[name] = person_id
                    continue

                # Create new person node
                person = Person(
                    id=person_id,
                    canonical_name=canonical_name,  # ✅ Always clean name
                    content=f"Person: {canonical_name}",
                    session_time=session_time,  # Session time for temporal reasoning
                    metadata=Metadata(
                    source=source_id or session_id,
                        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                )
                nodes_to_add.append(person)
                person_map[name] = person_id

        # Step 2: Create Event nodes with concepts
        # Track created concepts to avoid duplicates
        created_concept_ids = set()
        
        event_relations: List[Relation] = []
        for i, event_data in enumerate(events_data):
            title = event_data.get("title")
            if not title:
                continue

            participants = event_data.get("participants", [])
            time_raw = event_data.get("time")  # Raw time from LLM
            concepts_raw = event_data.get("concepts", [])

            # Resolve concept IDs AND create Concept nodes
            # CRITICAL FIX: Actually create Concept nodes, not just IDs
            # ROOT FIX: Strip c_ prefix if LLM already added it (to avoid c_c_xxx)
            concept_ids = []
            for concept_name in concepts_raw:
                if concept_name and isinstance(concept_name, str):
                    # Strip c_ prefix if present (LLM sometimes adds it)
                    clean_name = concept_name[2:] if concept_name.startswith("c_") else concept_name
                    concept_id = self._concept_manager.resolve(clean_name)
                    concept_ids.append(concept_id)
                    
                    # Create Concept node if not already created
                    if concept_id not in created_concept_ids:
                        normalized_name = self._concept_manager.normalize_name(concept_name)
                        concept_node = Concept(
                            id=concept_id,
                            name=normalized_name,
                            aliases=[concept_name] if concept_name.lower() != normalized_name else [],
                            description=f"Concept: {concept_name}",
                            content=f"Concept: {concept_name}",
                            metadata=Metadata(
                    source=source_id or session_id,
                                created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                        )
                        nodes_to_add.append(concept_node)
                        created_concept_ids.add(concept_id)

            # Map participant names to IDs
            participant_ids = []
            for participant in participants:
                if participant in person_map:
                    participant_ids.append(person_map[participant])
                elif participant and isinstance(participant, str):
                    if participant.lower() == "speaker":
                        participant_ids.append("p_speaker")
                    else:
                        # Create person ID on the fly
                        normalized = participant.lower().replace(" ", "_")
                        pid = f"p_{normalized}"
                        participant_ids.append(pid)

            # CRITICAL FIX: Preserve fuzzy time information
            # Always save raw time string before any processing
            # Ensure time_raw is a string, not a dict or other type
            if time_raw:
                if isinstance(time_raw, dict):
                    # If time_raw is a dict, extract the raw string value
                    time_start_raw = time_raw.get("time_start_raw") or time_raw.get("time") or str(time_raw)
                elif isinstance(time_raw, str):
                    time_start_raw = time_raw
                else:
                    time_start_raw = str(time_raw)
            else:
                time_start_raw = None
            
            time_normalized = normalize_event_time(time_start_raw, session_date or session_time)

            # P0 FIX 1.1: 事件验证和完整性检查
            # 检查事件是否完整：至少需要标题和至少一个参与者
            if not title or not participant_ids:
                if self._cfg.runtime.max_concurrency <= 1:
                    print(f"  [Indexer] [WARN] 跳过不完整事件: title={title}, participants={len(participant_ids)}")
                continue  # 跳过不完整的事件
            
            # CRITICAL FIX: Enhance event title and content with more context
            # Build richer description from event data
            description_parts = [title]
            if time_start_raw and time_start_raw != "unknown":
                description_parts.append(f"on {time_start_raw}")
            if concepts_raw:
                description_parts.append(f"involving {', '.join(concepts_raw[:3])}")  # Limit to first 3 concepts
            
            enhanced_description = ". ".join(description_parts)

            # Create Event node with both raw and normalized times
            occurrence = json.dumps([source_id or session_id, session_time, sorted(participant_ids), title, time_start_raw, i], ensure_ascii=False)
            event_id = f"event_{_make_id(occurrence)}"
            event = Event(
                id=event_id,
                title=title,
                participants=participant_ids,
                concepts=concept_ids,
                time_start=time_normalized,  # None if fuzzy time or parsing failed
                time_start_raw=time_start_raw,  # Always preserve original raw time (must be string)
                granularity=("year" if len(time_normalized) == 4 else "month" if len(time_normalized) == 7 else "day") if time_normalized else None,
                time_resolved=time_normalized,  # P0 FIX: Store resolved time from _resolve_relative_time
                session_time=session_time,  # Session time for temporal reasoning
                content=f"Event: {enhanced_description}",  # Enhanced with time and context
                description=enhanced_description,  # Add description field for better retrieval
                metadata=Metadata(
                    source=source_id or session_id,
                    created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            )
            
            # P0 FIX 1.1: 事件创建后验证
            # 验证事件节点是否满足基本要求
            if not event.title or not event.participants:
                if self._cfg.runtime.max_concurrency <= 1:
                    print(f"  [Indexer] [WARN] 跳过无效事件节点: {event_id}")
                continue  # 跳过无效事件
            
            nodes_to_add.append(event)
            # The extraction contract lists participants, not ranked roles.
            # Array order is not evidence that later people were mere mentions.
            for pid in participant_ids:
                rel_type = "PARTICIPATED_IN"
                weight = 1.0

                event_relations.append(
                    Relation(
                        head_id=pid,
                        tail_id=event_id,
                        rel_type=rel_type,
                        weight=weight,
                        confidence=1.0,
                        edge_title=title,  # 添加事件标题到边属性
                        edge_time=time_normalized or time_start_raw,  # 添加时间到边属性
                    )
                )
            # 建立事件概念边：Event -> Concept
            for cid in concept_ids:
                event_relations.append(
                    Relation(
                        head_id=event_id,
                        tail_id=cid,
                        rel_type="INVOLVES",
                        weight=1.0,
                        confidence=1.0,
                    )
                )

        # Step 3: Create Fact nodes (self-reported, objective)
        fact_relations: List[Relation] = []
        for i, fact_data in enumerate(facts_data):
            subject = fact_data.get("subject")
            predicate = fact_data.get("predicate")
            object_name = fact_data.get("object")
            purpose_name = fact_data.get("purpose")

            if not subject or not predicate:
                continue

            # Get subject ID
            subject_id = person_map.get(subject, f"p_{subject.lower().replace(' ', '_')}")

            # Resolve object concept AND create Concept node
            # ROOT FIX: Strip c_ prefix if LLM already added it (to avoid c_c_xxx)
            object_id = None
            if object_name and isinstance(object_name, str):
                clean_object_name = object_name[2:] if object_name.startswith("c_") else object_name
                object_id = self._concept_manager.resolve(clean_object_name)
                # Create Concept node if not already created
                if object_id not in created_concept_ids:
                    normalized_name = self._concept_manager.normalize_name(clean_object_name)
                    concept_node = Concept(
                        id=object_id,
                        name=normalized_name,
                        aliases=[object_name] if object_name.lower() != normalized_name else [],
                        description=f"Concept: {object_name}",
                        content=f"Concept: {object_name}",
                        metadata=Metadata(
                    source=source_id or session_id,
                            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                    )
                    nodes_to_add.append(concept_node)
                    created_concept_ids.add(object_id)

            # Resolve purpose concept AND create Concept node
            # ROOT FIX: Strip c_ prefix if LLM already added it (to avoid c_c_xxx)
            purpose_id = None
            if purpose_name and isinstance(purpose_name, str):
                clean_purpose_name = purpose_name[2:] if purpose_name.startswith("c_") else purpose_name
                purpose_id = self._concept_manager.resolve(clean_purpose_name)
                # Create Concept node if not already created
                if purpose_id not in created_concept_ids:
                    normalized_name = self._concept_manager.normalize_name(clean_purpose_name)
                    concept_node = Concept(
                        id=purpose_id,
                        name=normalized_name,
                        aliases=[purpose_name] if purpose_name.lower() != normalized_name else [],
                        description=f"Concept: {purpose_name}",
                        content=f"Concept: {purpose_name}",
                        metadata=Metadata(
                    source=source_id or session_id,
                            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                    )
                    nodes_to_add.append(concept_node)
                    created_concept_ids.add(purpose_id)

            # Skip facts that are missing both object and purpose (incomplete facts)
            # This prevents ValidationError and reduces error rate
            if object_id is None and purpose_id is None:
                if getattr(self._cfg.extraction, 'skip_incomplete_facts', True):
                    print(f"[WARN] Skipping incomplete fact: {subject} {predicate} (no object or purpose)")
                    continue
                else:
                    # If configured to keep incomplete facts, set a default placeholder
                    # CRITICAL FIX: Use concept_manager.resolve() to get correct ID (c_unknown, not c_c_unknown)
                    print(f"[WARN] Incomplete fact (no object/purpose): {subject} {predicate}. Adding placeholder.")
                    object_id = self._concept_manager.resolve("unknown")  # Use concept_manager to avoid double prefix

            # Create Fact node
            fact_id = "fact_" + _make_id(json.dumps([source_id or session_id, session_time, subject, predicate, object_name or purpose_name, fact_data.get("content"), i], ensure_ascii=False))
            
            # ===== P0 FIX 1.2: 构建自然语言的content字段用于向量检索 =====
            # 1. 获取subject的自然语言名称
            subject_name_natural = subject  # 默认使用原始subject（通常是"Speaker"或person名称）
            # 反向查找person_map，找到subject_id对应的原始名称
            for name, pid in person_map.items():
                if pid == subject_id:
                    subject_name_natural = name
                    break
            # 如果没找到，且subject_id是person ID，从ID推断
            if subject_name_natural == subject and subject_id.startswith('p_'):
                subject_name_natural = subject_id.replace('p_', '').replace('_', ' ').title()
            
            # 2. 获取object的自然语言名称（P0 FIX 1.2: 确保使用完整、准确的object名称）
            object_name_natural = ""
            if object_name:
                # 去掉c_前缀并转换为自然语言
                if object_name.startswith('c_'):
                    object_name_natural = object_name[2:].replace('_', ' ').title()
                else:
                    object_name_natural = object_name.replace('_', ' ').title()
            # P0 FIX 1.2: 如果object_name_natural为空，尝试从object_id获取
            if not object_name_natural and object_id:
                if object_id.startswith('c_'):
                    object_name_natural = object_id[2:].replace('_', ' ').title()
                else:
                    object_name_natural = object_id.replace('_', ' ').title()
            
            # 3. 规范化predicate为自然语言（用于向量检索）
            pred_norm = (predicate or "").lower()
            predicate_natural = predicate  # 默认使用原始predicate
            
            # P0 FIX 1.2: 扩展predicate映射，确保"research"类查询能正确匹配
            predicate_mapping = {
                "is_keen_on": "researched",
                "is_interested_in": "researched",
                "is_curious_about": "researched",
                "is_excited_to_learn_about": "researched",
                "is_searching_for": "researched",
                "researches": "researched",  # 新增：直接匹配researches
                "researched": "researched",  # 新增：直接匹配researched
                "is_learning": "studied",
                "studies": "studied",
                "studied": "studied",  # 新增：直接匹配studied
                "learns": "studied",
                "has_habit": "regularly does",
                "uses": "uses",
                "likes": "likes",
                "loves": "loves",
                "enjoys": "enjoys",
                "is_a": "is",
                "works_as": "works as",
                "lives_in": "lives in",
                "works_at": "works at"
            }
            predicate_natural = predicate_mapping.get(pred_norm, predicate)
            
            # 4. 构建语义化的content字段
            if object_name_natural:
                content_natural = f"{subject_name_natural} {predicate_natural} {object_name_natural}"
            elif purpose_name:
                purpose_natural = purpose_name.replace('c_', '').replace('_', ' ').title() if purpose_name.startswith('c_') else purpose_name
                content_natural = f"{subject_name_natural} {predicate_natural} {purpose_natural}"
            else:
                content_natural = f"{subject_name_natural} {predicate_natural}"
            
            fact = Fact(
                id=fact_id,
                subject_id=subject_id,
                predicate=predicate,
                object_id=object_id,
                purpose_id=purpose_id,
                content=content_natural,  # P0 FIX: 使用自然语言而非节点ID
                confidence=1.0,
                session_time=session_time,  # Session time for temporal reasoning
                metadata=Metadata(
                    source=source_id or session_id,
                    created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ),
                memory_lifecycle=MemoryLifecycle.CANDIDATE,  # Indexing facts are candidates until consolidated
            )
            nodes_to_add.append(fact)
            # The reified Fact carries its own source and occurrence identity.
            # A subject-to-object shortcut cannot make that Fact reachable.
            fact_relations.append(Relation(
                head_id=subject_id,
                tail_id=fact_id,
                rel_type="HAS_ATTRIBUTE",
                weight=1.0,
                confidence=fact.confidence,
            ))
            # 关系映射：将自然谓词映射到图边类型
            pred_norm = (predicate or "").lower()
            if pred_norm in ["loves", "love", "likes", "like", "enjoys", "enjoy"]:
                edge_type = "LIKES"
            elif pred_norm in ["uses", "utilizes", "use"]:
                edge_type = "USES"
            elif pred_norm in ["is", "is_a", "works_as"]:
                edge_type = "IS_A"
            else:
                edge_type = "HAS_ATTRIBUTE"

            if object_id:
                fact_relations.append(
                    Relation(
                        head_id=subject_id,
                        tail_id=object_id,
                        rel_type=edge_type,
                        weight=1.0,
                        confidence=1.0,
                    )
                )
            if purpose_id:
                fact_relations.append(
                    Relation(
                        head_id=subject_id,
                        tail_id=purpose_id,
                        rel_type="USES",
                        weight=1.0,
                        confidence=1.0,
                    )
                )

        # Step 4: Create Claim nodes with two-hop structure
        # Source --[CLAIMS]--> Claim --[ABOUT]--> Target
        for i, claim_data in enumerate(claims_data):
            source = claim_data.get("source")
            target = claim_data.get("target")
            content = claim_data.get("content")
            sentiment = claim_data.get("sentiment")

            if not source or not target or not content:
                continue

            # Speaker identity and extraction-document provenance are separate.
            source_person_id = person_map.get(source, f"p_{source.lower().replace(' ', '_')}")

            # Resolve target (could be person or concept)
            # ROOT FIX: Strip c_ prefix if LLM already added it (to avoid c_c_xxx)
            target_id = None
            if isinstance(target, str):
                if target.startswith("p_") or target.startswith("person_"):
                    # Already a person ID
                    target_id = target
                elif target.startswith("c_"):
                    # LLM provided concept ID - strip c_ and resolve properly
                    clean_target = target[2:]
                    target_id = self._concept_manager.resolve(clean_target)
                    # Create Concept node if not already created
                    if target_id not in created_concept_ids:
                        normalized_name = self._concept_manager.normalize_name(clean_target)
                        concept_node = Concept(
                            id=target_id,
                            name=normalized_name,
                            aliases=[clean_target] if clean_target.lower() != normalized_name else [],
                            description=f"Concept: {clean_target}",
                            content=f"Concept: {clean_target}",
                            metadata=Metadata(
                    source=source_id or session_id,
                                created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                        )
                        nodes_to_add.append(concept_node)
                        created_concept_ids.add(target_id)
                elif target in person_map:
                    # Target is a known person
                    target_id = person_map[target]
                else:
                    # Target is likely a concept - resolve AND create Concept node
                    target_id = self._concept_manager.resolve(target)
                    # Create Concept node if not already created
                    if target_id not in created_concept_ids:
                        normalized_name = self._concept_manager.normalize_name(target)
                        concept_node = Concept(
                            id=target_id,
                            name=normalized_name,
                            aliases=[target] if target.lower() != normalized_name else [],
                            description=f"Concept: {target}",
                            content=f"Concept: {target}",
                            metadata=Metadata(
                    source=source_id or session_id,
                                created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                        )
                        nodes_to_add.append(concept_node)
                        created_concept_ids.add(target_id)

            # Create Claim node
            claim_id = "claim_" + _make_id(json.dumps([source_id or session_id, session_time, source, target, content, i], ensure_ascii=False))
            claim = Claim(
                id=claim_id,
                source_id=source_person_id,
                target_id=target_id,
                content=content,
                sentiment=sentiment,
                confidence=0.8,  # Claims have lower confidence than facts
                session_time=session_time,  # Session time for temporal reasoning
                metadata=Metadata(
                    source=source_id or session_id,
                    created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            )
            nodes_to_add.append(claim)

            # Create CLAIMS edge: Source -> Claim
            rel_claims = Relation(
                head_id=source_person_id,
                tail_id=claim_id,
                rel_type="CLAIMS",
                weight=1.0,
                confidence=0.8
            )
            relations_to_add.append(rel_claims)

            # Create ABOUT edge: Claim -> Target
            rel_about = Relation(
                head_id=claim_id,
                tail_id=target_id,
                rel_type="ABOUT",
                weight=1.0,
                confidence=0.8
            )
            relations_to_add.append(rel_about)

        # 只在有实际节点或关系时打印日志，减少噪音
        if nodes_to_add or relations_to_add:
            print(f"  [Indexer] Created {len(nodes_to_add)} nodes and {len(relations_to_add)} relations")

        # 合并事件/事实新增的关系
        relations_to_add.extend(event_relations)
        relations_to_add.extend(fact_relations)

        return nodes_to_add, relations_to_add

    # PersonFacet 已废弃，使用 Concept 和 Fact 来表达侧面信息

    # Episode 已废弃，此方法已删除
    # 使用 Event 作为核心动态单元，不再需要 Episode 聚合

    def _batch_embed_nodes(self, nodes: List) -> None:
        """批量嵌入节点文本（步骤1：为所有节点类型生成向量）
        
        CRITICAL FIX: Ensure Person and Concept nodes use meaningful text for embedding.
        For Person: Use canonical_name + " " + str(aliases)
        For Concept: Use name
        """
        if not nodes:
            return

        # 检查是否使用新的TypedVectorStore
        if not self._vstore and not self._typed_vstore:
            return

        # 提取节点文本和类型（步骤1改进：为每种节点类型生成富文本）
        node_texts = []
        node_types = []
        for node in nodes:
            # CRITICAL FIX: For Person and Concept, use explicit text generation
            if isinstance(node, Person):
                # Person: Use canonical_name + " " + str(aliases)
                text_parts = [node.canonical_name]
                if node.aliases:
                    if isinstance(node.aliases, list):
                        text_parts.extend([str(a) for a in node.aliases])
                    else:
                        text_parts.append(str(node.aliases))
                rich_text = " ".join(text_parts)
            elif isinstance(node, Concept):
                # Concept: Use name
                rich_text = node.name
            else:
                # Other node types: Use existing rich text generation
                rich_text = self._generate_rich_node_text(node)
            
            node_texts.append(rich_text[:800])  # 增加到 800 字符以包含更多信息
            # 获取节点类型
            node_type = self._get_node_type_str(node)
            node_types.append(node_type)

        if not node_texts:
            return

        print(f"  正在生成节点向量（共 {len(node_texts)} 个节点，这可能需要几分钟）...")
        try:
            # 批量编码（分批处理以显示进度）
            batch_size = max(1, self._cfg.runtime.embedding_batch_size)
            all_embeddings = []

            for i in tqdm(range(0, len(node_texts), batch_size), desc="节点向量化进度"):
                batch_texts = node_texts[i:i+batch_size]
                batch_nodes = nodes[i:i+batch_size]
                batch_node_types = node_types[i:i+batch_size]
                batch_embeddings = self._embed.encode(batch_texts)
                if len(batch_embeddings) != len(batch_nodes):
                    raise ValueError("Embedding result count does not match source nodes")

                if batch_embeddings:
                    all_embeddings.extend(batch_embeddings)
                    # 立即添加到向量库（避免内存占用过大）
                    batch_node_ids = [node.id for node in batch_nodes]
                    
                    # 使用TypedVectorStore（如果可用）
                    if self._typed_vstore:
                        self._typed_vstore.add_batch_typed(
                            batch_node_ids, batch_embeddings, batch_node_types
                        )
                    else:
                        # 向后兼容：使用旧的VectorStore
                        self._vstore.add(batch_node_ids, batch_embeddings)

            print(f"  [OK] 节点向量化完成，共生成 {len(all_embeddings)} 个向量")
        except Exception as e:
            print(f"[!]  节点嵌入失败: {e}")
            # 如果是embedding维度问题，提供详细错误
            if "dimension" in str(e).lower():
                print("   提示: 检查embedding模型配置")
            raise

    def _get_node_type_str(self, node) -> str:
        """获取节点类型字符串（用于TypedVectorStore）
        
        返回与 NodeTypeEnum 一致的小写类型名，以确保索引和检索使用相同的类型名。
        """
        from ..storage.schema import Person, Event, Fact, Concept, Claim, NodeTypeEnum

        if isinstance(node, Event):
            return NodeTypeEnum.EVENT.value
        elif isinstance(node, Fact):
            return NodeTypeEnum.FACT.value
        elif isinstance(node, Person):
            return NodeTypeEnum.PERSON.value
        elif isinstance(node, Concept):
            return NodeTypeEnum.CONCEPT.value
        elif isinstance(node, Claim):
            return NodeTypeEnum.CLAIM.value
        else:
            # 默认类型
            return NodeTypeEnum.CONCEPT.value

    def _generate_rich_node_text(self, node):
        """P0 FIX 1.3: 生成更丰富的节点文本用于向量嵌入，提高检索质量"""
        """生成富文本表示，包含节点内容+连接的实体名（步骤1核心）"""
        from ..storage.schema import Person, Event, Fact, Concept, Claim

        parts = []

        # 根据节点类型生成富文本（增强版：包含更多上下文信息）
        # Episode 已废弃，不再处理
        
        if isinstance(node, Event):
            # P0 FIX 1.3: Event节点 - 使用更丰富的文本表示用于向量嵌入
            # 格式: title + description + time + participants + concepts
            parts.append(node.title or "")
            if hasattr(node, 'description') and node.description:
                parts.append(node.description)
            if node.location:
                parts.append(f"at {node.location}")
            
            # P0 FIX 1.3: 优先使用time_start_raw（包含更多上下文），如果没有则使用time_start
            if hasattr(node, 'time_start_raw') and node.time_start_raw and node.time_start_raw != "unknown":
                parts.append(f"on {node.time_start_raw}")
            elif node.time_start:
                # 只提取日期部分，更简洁
                time_str = node.time_start.split('T')[0] if 'T' in node.time_start else node.time_start
                parts.append(f"on {time_str}")
            elif hasattr(node, 'session_time') and node.session_time:
                # 如果没有明确时间，使用session_time作为上下文
                parts.append(f"session: {node.session_time}")
            
            # 添加参与者信息（重要！）- 尝试解析为实际名称
            if node.participants:
                participant_names = []
                for pid in node.participants[:3]:  # 限制前3个
                    try:
                        person = self._graph.get_person(pid)
                        if person:
                            participant_names.append(person.canonical_name)
                        else:
                            # 如果找不到Person节点，使用原始ID
                            participant_names.append(pid.replace('person_', '').replace('p_', '').replace('_', ' ').title())
                    except:
                        participant_names.append(pid.replace('person_', '').replace('p_', '').replace('_', ' ').title())
                if participant_names:
                    parts.append(f"with {', '.join(participant_names)}")
            
            # CRITICAL FIX: 解析 Event 的 concepts 字段为人类可读文本
            # concepts 是 Concept ID 列表，如 ["c_dancing", "c_destress"]
            if hasattr(node, 'concepts') and node.concepts:
                concept_names = []
                for cid in node.concepts[:5]:  # 限制前5个
                    try:
                        # 尝试从图存储中获取 Concept 节点
                        concept_node = self._graph.get_concept(cid)
                        if concept_node:
                            concept_names.append(concept_node.name)
                        else:
                            # 如果找不到，使用 ID 转换为可读名称
                            concept_name = cid.replace('c_', '').replace('concept_', '').replace('_', ' ')
                            concept_names.append(concept_name)
                    except:
                        # 容错：直接转换 ID
                        concept_name = cid.replace('c_', '').replace('concept_', '').replace('_', ' ')
                        concept_names.append(concept_name)
                if concept_names:
                    parts.append(f"related to {', '.join(concept_names)}")
            
            # Topic 已废弃，不再处理

        elif isinstance(node, Fact):
            # P0 FIX: 改进Fact节点的向量化文本生成，确保包含完整信息
            # 优先使用content字段（已包含自然语言），如果没有则构建
            if hasattr(node, 'content') and node.content:
                # 验证content字段是否为自然语言（不包含节点ID格式）
                content_str = str(node.content)
                # P0 FIX: 确保content字段包含完整的语义信息
                # 如果content不包含subject信息，尝试添加
                subject_id = getattr(node, 'subject_id', '')
                if subject_id and subject_id not in content_str:
                    # 尝试解析subject_id为实际名称
                    try:
                        subject_node = self._graph.get_person(subject_id)
                        if subject_node:
                            subject_name = subject_node.canonical_name
                            # 如果content不包含subject名称，添加
                            if subject_name.lower() not in content_str.lower():
                                content_str = f"{subject_name} {content_str}"
                    except:
                        pass
                
                # P0 FIX: 添加predicate信息以提高检索精度（特别是"research"类查询）
                predicate = getattr(node, 'predicate', '')
                if predicate and predicate not in content_str.lower():
                    # 如果predicate不在content中，添加到向量化文本
                    # 但不要重复添加subject信息
                    parts.append(f"{content_str} {predicate}")
                else:
                    parts.append(content_str)
            else:
                # 回退到旧逻辑（兼容性）
                # Fact: subject + predicate + object + 上下文 + 时间有效性
                subject_id = getattr(node, 'subject_id', getattr(node, 'subject', ''))
                predicate = getattr(node, 'predicate', '')
                object_id = getattr(node, 'object_id', getattr(node, 'object', ''))
                value = getattr(node, 'value', '')
                
                # 尝试解析 subject_id 为实际名称
                subject_name = subject_id
                if subject_id and (subject_id.startswith('p_') or subject_id.startswith('person_')):
                    try:
                        subject_node = self._graph.get_person(subject_id)
                        if subject_node:
                            subject_name = subject_node.canonical_name
                    except:
                        # 如果无法获取，从ID推断
                        subject_name = subject_id.replace('p_', '').replace('person_', '').replace('_', ' ').title()
                
                # P0 FIX: 构建更完整的Fact文本表示
                fact_text = f"{subject_name} {predicate}"
                if object_id:
                    # 尝试解析 object_id 为实际名称
                    if object_id.startswith('c_') or object_id.startswith('concept_'):
                        try:
                            obj_node = self._graph.get_concept(object_id)
                            if obj_node:
                                obj_name = obj_node.name.replace('_', ' ').title()
                                fact_text = f"{fact_text} {obj_name}"
                            else:
                                # 如果无法获取，从ID推断
                                obj_name = object_id.replace('c_', '').replace('concept_', '').replace('_', ' ').title()
                                fact_text = f"{fact_text} {obj_name}"
                        except:
                            obj_name = object_id.replace('c_', '').replace('concept_', '').replace('_', ' ').title()
                            fact_text = f"{fact_text} {obj_name}"
                    else:
                        fact_text = f"{fact_text} {object_id}"
                elif value:
                    fact_text = f"{fact_text} {value}"
                
                parts.append(fact_text)
                
                # P0 FIX: 添加时间有效性信息，确保时间信息包含在向量化文本中
                # 这对于时间查询非常重要
                if hasattr(node, 'valid_from') and node.valid_from:
                    # 只提取日期部分，更简洁
                    valid_from_str = node.valid_from.split('T')[0] if 'T' in node.valid_from else node.valid_from
                    parts.append(f"from {valid_from_str}")
                if hasattr(node, 'valid_to') and node.valid_to:
                    # 只提取日期部分，更简洁
                    valid_to_str = node.valid_to.split('T')[0] if 'T' in node.valid_to else node.valid_to
                    parts.append(f"until {valid_to_str}")
                
                # P0 FIX: 如果Fact节点有SUPPORTED_BY关系，尝试从Event节点提取时间信息
                # 这对于没有valid_from/valid_to的Fact节点很重要
                if not hasattr(node, 'valid_from') or not node.valid_from:
                    # 尝试从关联的Event节点获取时间信息
                    fact_id = getattr(node, 'id', '')
                    if fact_id:
                        try:
                            # 查找SUPPORTED_BY关系
                            fact_node_data = self._graph._g.nodes.get(fact_id, {})
                            if fact_node_data:
                                # 查找出边中的SUPPORTED_BY关系
                                for _, target_id, edge_data in self._graph._g.out_edges(fact_id, data=True):
                                    if edge_data.get('relation_type') == 'SUPPORTED_BY':
                                        # 获取Event节点的时间信息
                                        event_data = self._graph._g.nodes.get(target_id, {})
                                        if event_data and event_data.get('node_type') == 'event':
                                            time_start = event_data.get('time_start_raw') or event_data.get('time_start')
                                            if time_start:
                                                # 只提取日期部分
                                                time_str = time_start.split('T')[0] if 'T' in time_start else time_start
                                                parts.append(f"related to event on {time_str}")
                                                break
                        except Exception:
                            pass

        elif isinstance(node, Claim):
            # Occurrence IDs and indexing metadata are addresses, not semantics.
            def reference_name(reference):
                data = self._graph.get_node(reference) or {}
                return data.get("canonical_name") or data.get("name") or data.get("title") or ""
            source_name = reference_name(node.source_id)
            target_name = reference_name(node.target_id)
            if source_name:
                parts.append(f"{source_name} says")
            if target_name:
                parts.append(f"about {target_name}:")
            parts.append(node.content or "")

        elif isinstance(node, Person):
            # P0 FIX 1.3: Person节点 - 使用更丰富的文本表示
            # 格式: canonical_name + description + known_facts (如果可用)
            parts.append(node.canonical_name or "")
            if hasattr(node, 'description') and node.description:
                parts.append(node.description)
            if hasattr(node, 'aliases') and node.aliases:
                if isinstance(node.aliases, list):
                    parts.extend([str(a) for a in node.aliases[:3]])  # 添加前3个别名
                else:
                    parts.append(str(node.aliases))
            # Optional: Add profession and traits if available (but prioritize name + aliases)
            if hasattr(node, 'profession') and node.profession:
                parts.append(f"profession: {node.profession}")
            if hasattr(node, 'persona_traits') and node.persona_traits:
                parts.append(f"traits: {', '.join(node.persona_traits[:3])}")  # 前3个特征
            if hasattr(node, 'background') and node.background:
                parts.append(f"background: {node.background}")
            if node.identity:
                parts.append(f"identity: {node.identity}")

        elif isinstance(node, Concept):
            # CRITICAL FIX: Concept - Use name (simple and direct)
            # This ensures meaningful text for embedding
            parts.append(node.name)
            if node.description:
                parts.append(node.description)
            if hasattr(node, 'aliases') and node.aliases:
                if isinstance(node.aliases, list):
                    parts.extend([str(a) for a in node.aliases[:3]])
                else:
                    parts.append(str(node.aliases))

        else:
            # 其他节点类型
            if hasattr(node, 'name') and node.name:
                parts.append(str(node.name))
            elif hasattr(node, 'description') and node.description:
                parts.append(str(node.description))
            else:
                parts.append(str(node))

        return ' '.join(parts)
