from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Iterable, List, Tuple, Dict, Optional, Any

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..config import CogMemConfig
from ..models.llm_client import create_llm_client
from ..models.embedding import EmbeddingEncoder
from ..extraction.chunker import ChunkerFactory
from ..extraction.structured_extractor import StructuredExtractor
from ..storage.schema import (
    Doc, Entity, Metadata, Relation, Person, Event, Fact, Preference, PersonFacet, Episode, NodeType
)
from ..storage.graph_store import GraphStore
from ..storage.vector_store import VectorStore
from ..storage.evidence_store import EvidenceStore
from ..storage.chunk_store import ChunkStore, Chunk
from ..utils.json_utils import safe_json_loads


def _make_id(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()


class Indexer:
    """文档索引器：负责文档切分、实体抽取、关系抽取、向量索引和图存储"""

    def __init__(self, cfg: CogMemConfig, graph: GraphStore, vstore: VectorStore, chunk_store: ChunkStore | None = None, cogmem=None) -> None:
        self._cfg = cfg
        self._graph = graph
        self._vstore = vstore
        self._llm = create_llm_client(cfg)
        self._embed = EmbeddingEncoder(cfg)
        self._evidence = EvidenceStore()
        self._chunk_store = chunk_store
        self._cogmem = cogmem  # Reference to CogMem instance for saving
        # Structured extractor for person-centered graph
        self._structured_extractor = StructuredExtractor(self._llm)
        # Check if person-centered extraction is enabled
        self._use_person_centered = getattr(cfg.extraction, 'use_person_centered', False)


    def _batch_iter(self, items: List[str], batch_size: int) -> Iterable[List[str]]:
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def add_documents(self, docs: Iterable[Doc], incremental: bool = False) -> None:
        """
        统一的文档索引入口。
        通过 person-centered, event-mediated 的方式处理文档。

        Args:
            docs: 文档迭代器
            incremental: 是否执行增量索引
        """
        docs_to_process = list(docs)

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

                for chunk_text in chunk_texts:
                    # 2. 结构化抽取
                    extraction_result = self._extract_structured_from_chunk(
                        chunk=chunk_text,
                        doc_id=doc.id,
                        session_id=session_id,
                        turn_id=turn_id
                    )

                    # 3. 构建图元素
                    nodes, relations = self._build_event_mediated_graph(extraction_result, session_id)
                    doc_nodes.extend(nodes)
                    doc_relations.extend(relations)

                    # 调试：如果节点为空，打印详细信息
                    if not nodes and chunk_text.strip():
                        # 只在调试模式下打印，避免过多输出
                        if self._cfg.runtime.max_concurrency == 1:
                            # 检查提取结果的结构
                            nodes_dict = extraction_result.get("nodes", {})
                            print(f"[!]  警告：文档 {doc.id} 的 chunk 未提取到节点")
                            print(f"   提取结果结构: persons={len(nodes_dict.get('persons', []))}, "
                                  f"events={len(nodes_dict.get('events', []))}, "
                                  f"facts={len(nodes_dict.get('facts', []))}, "
                                  f"entities={len(nodes_dict.get('entities', []))}")
                            # 打印原始提取结果的前200字符（用于调试）
                            import json
                            try:
                                result_str = json.dumps(extraction_result, indent=2, ensure_ascii=False)[:500]
                                print(f"   提取结果预览: {result_str}...")
                            except:
                                print(f"   无法序列化提取结果")

                    # 记录 chunk 用于后续的向量化
                    chunk_id = f"chunk-{doc.id}-{_make_id(chunk_text)}"
                    doc_chunks.append(Chunk(id=chunk_id, text=chunk_text[:500], doc_id=doc.id, turn_id=turn_id))

                # 创建 PersonFacet 节点（如果启用）
                persons = [n for n in doc_nodes if isinstance(n, Person)]
                facts = [n for n in doc_nodes if isinstance(n, Fact)]
                preferences = [n for n in doc_nodes if isinstance(n, Preference)]

                for person in persons:
                    facets = self._create_person_facets(person.id, facts, preferences)
                    doc_nodes.extend(facets)

                return doc_nodes, doc_relations, doc_chunks
            except Exception as e:
                import traceback
                print(f"[!]  处理文档 {doc.id} 时出错: {str(e)}")
                if self._cfg.runtime.max_concurrency == 1:
                    traceback.print_exc()
                return [], [], []

        # 使用 ThreadPoolExecutor 并行处理文档
        max_workers = max(1, self._cfg.runtime.max_concurrency)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(process_doc, doc) for doc in docs_to_process]

            # 定期保存的计数器
            processed_docs_count = 0
            save_interval = 50  # 每处理50个文档保存一次

            try:
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting from documents"):
                    try:
                        nodes, relations, chunks = fut.result(timeout=30.0)
                        all_nodes.extend(nodes)
                        all_relations.extend(relations)
                        chunks_for_embedding.extend(chunks)

                        processed_docs_count += 1

                        # 定期保存索引（每50个文档一次）
                        if self._cogmem and processed_docs_count % save_interval == 0:
                            print(f"\n[Checkpoint] 已处理 {processed_docs_count}/{len(docs_to_process)} 个文档，正在保存索引...")
                            try:
                                # 在保存前，先将已处理的节点临时添加到图中（用于checkpoint）
                                # 注意：这里只处理已收集的节点，不处理episodes（episodes在最后统一处理）
                                if all_nodes:
                                    # 临时去重并添加节点（不处理episodes）
                                    temp_unique_nodes_dict = {}
                                    for node in all_nodes:
                                        temp_unique_nodes_dict[node.id] = node
                                    temp_unique_nodes = list(temp_unique_nodes_dict.values())
                                    
                                    # 临时添加节点到图（如果还没有添加）
                                    nodes_to_add = [node for node in temp_unique_nodes if node.id not in self._graph._g]
                                    if nodes_to_add:
                                        self._graph.add_nodes(nodes_to_add)
                                    
                                    # 临时添加关系（只添加已存在的节点之间的关系）
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
                                            
                                            # 只添加新关系（避免重复）
                                            for r in temp_unique_relations:
                                                if not self._graph._g.has_edge(r.head_id, r.tail_id, key=r.rel_type):
                                                    self._graph.add_relations([r])
                                
                                self._cogmem.save()
                                print(f"[Checkpoint] ✓ 索引保存成功（已处理 {processed_docs_count} 个文档，图中有 {self._graph.num_entities()} 个节点）")
                            except Exception as save_err:
                                print(f"[Checkpoint] ✗ 索引保存失败: {str(save_err)}")

                    except Exception as e:
                        if isinstance(e, KeyboardInterrupt):
                            raise
                        print(f"[!]  获取文档处理结果时出错: {str(e)}")
                        continue
            except KeyboardInterrupt:
                # Ctrl+C 直接退出，不等待
                print("\n[!]  收到中断信号，正在退出...")
                # 取消所有未完成的任务
                for future in futures:
                    if not future.done():
                        future.cancel()
                raise

        # 4. 去重和存图（在循环外批量处理）
        # 去重 nodes（基于 id）
        unique_nodes_dict = {}
        for node in all_nodes:
            unique_nodes_dict[node.id] = node
        unique_nodes = list(unique_nodes_dict.values())

        # 4.1 按时间分组 Events，创建 Episodes
        events_to_group = [n for n in unique_nodes if isinstance(n, Event)]
        episode_nodes, episode_relations = [], []

        if events_to_group:
            print(f"  正在按周分组事件（共 {len(events_to_group)} 个事件）...")
            episode_nodes, episode_relations = self._group_by_time(events_to_group)

            # 将 episodes 添加到 unique_nodes
            unique_nodes.extend(episode_nodes)

            # 为每个 episode 中的参与者创建 PARTICIPATES_IN 关系
            # Person -> Episode
            episode_ids = {ep.id for ep in episode_nodes}
            person_ids = {n.id for n in unique_nodes if isinstance(n, Person)}

            for episode in episode_nodes:
                for person_id in episode.participant_ids:
                    if person_id in person_ids:
                        rel = Relation(
                            head_id=person_id,
                            tail_id=episode.id,
                            rel_type="PARTICIPATES_IN",
                            weight=1.0,
                            confidence=1.0,
                            metadata=Metadata(
                                source="episode_participation",
                                created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                        )
                        episode_relations.append(rel)

            print(f"  ✓ 创建了 {len(episode_nodes)} 个 episode 和 {len(episode_relations)} 个关系")

        # 4.2 存图（nodes）
        if unique_nodes:
            self._graph.add_nodes(unique_nodes)

        # 4.3 去重并合并所有 relations（原始关系 + episode 关系）
        all_relations.extend(episode_relations)

        # 去重 relations
        unique_relations_dict = {}
        for r in all_relations:
            key = (r.head_id, r.tail_id, r.rel_type)
            if key not in unique_relations_dict or r.confidence > unique_relations_dict[key].confidence:
                unique_relations_dict[key] = r
        unique_relations = list(unique_relations_dict.values())

        # 4.4 存图（relations）
        if unique_relations:
            node_ids = {n.id for n in unique_nodes}
            filtered_relations = [
                r for r in unique_relations
                if r.head_id in node_ids and r.tail_id in node_ids
            ]
            if filtered_relations:
                self._graph.add_relations(filtered_relations)

        # 5. 批量嵌入节点
        try:
            self._batch_embed_nodes(unique_nodes)
        except KeyboardInterrupt:
            print("\n[!]  节点嵌入被中断")
            raise

        # 6. 批量嵌入 Chunks
        if self._chunk_store and chunks_for_embedding:
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

                print(f"  ✓ 文档片段向量化完成")
            except KeyboardInterrupt:
                print("\n[!]  Chunk嵌入被中断")
                raise

    def _extract_structured_from_chunk(
        self,
        chunk: str,
        doc_id: str,
        session_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        question_date_iso: Optional[str] = None
    ) -> Dict:
        """Extract structured information using structured extractor"""
        # Extract session date from doc_id if available
        context_start = None
        context_end = None

        # Try to extract date from session info in doc text
        if "[Session Date:" in chunk:
            import re
            match = re.search(r'\[Session Date:\s*([^\]]+)\]', chunk)
            if match:
                question_date_iso = match.group(1).strip()

        return self._structured_extractor.extract(
            input_text=chunk,
            doc_id=doc_id,
            session_id=session_id,
            turn_id=turn_id,
            question_date_iso=question_date_iso,
            context_window_start_iso=context_start,
            context_window_end_iso=context_end
        )

    @property
    def evidence(self) -> EvidenceStore:
        return self._evidence

    def _build_event_mediated_graph(
        self,
        extraction_result: Dict,
        session_id: Optional[str] = None
    ) -> Tuple[List, List]:
        """Build graph using event-mediated approach: Person → Event → Fact/Preference

        Returns:
            Tuple of (nodes_to_add, relations_to_add)
        """
        # Import EntityType at function level for shadow entity creation
        from ..storage.schema import Entity as EntityType

        nodes_to_add = []
        relations_to_add = []

        nodes = extraction_result.get("nodes", {})
        edges = extraction_result.get("edges", {})

        # 1. Add Person nodes
        persons = nodes.get("persons", [])
        for person in persons:
            nodes_to_add.append(person)

        # 2. Add Event nodes
        events = nodes.get("events", [])
        event_map = {}  # event_id -> Event
        for event in events:
            nodes_to_add.append(event)
            event_map[event.id] = event

        # 3. Add Entity nodes (non-person entities)
        entities_dict = nodes.get("entities", [])
        entity_map = {}  # Track created entities for later reference checking
        for entity_dict in entities_dict:
            # Convert dict to Entity for compatibility
            try:
                entity = EntityType(
                    id=entity_dict.get("id", f"entity_{hash(entity_dict.get('canonical_name', '')) % 1000000}"),
                    name=entity_dict.get("canonical_name", ""),
                    description="",
                    type="world",
                    aliases=entity_dict.get("aliases", []),
                    confidence=entity_dict.get("confidence", 1.0),
                    metadata=Metadata(
                        source=entity_dict.get("evidence_span", {}).get("text", ""),
                        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        source_session_id=entity_dict.get("source_session_id"),
                        source_turn_id=entity_dict.get("source_turn_id")
                    )
                )
                nodes_to_add.append(entity)
                entity_map[entity.id] = entity
            except Exception:
                continue

        # 4. Add Fact and Preference nodes
        facts = nodes.get("facts", [])
        preferences = nodes.get("preferences", [])

        # 5. Create event-mediated relations: Person → Event → Fact/Preference
        # Changed: Person INVOLVED_IN Event edges are now deprecated.
        # Events will be connected to Persons via Episodes created by _group_by_time method.
        # The hierarchy is: Person → Episode → Event → Fact/Preference

        # REMOVED: Direct Person INVOLVED_IN Event relations (line 393-406)
        # These relations will be created indirectly through Episodes during add_documents phase.

        # Event ASSERTS Fact/Preference (keep this direct edge as Event is the source of facts)
        for edge in edges.get("asserts", []):
            # 支持 "source"/"target" 或 "from"/"to" 格式
            from_id = edge.get("source") or edge.get("from")  # Turn/Event ID
            to_id = edge.get("target") or edge.get("to")  # Fact/Preference ID
            if from_id and to_id:
                # Check if from_id is an event, if not, try to find associated event
                if from_id in event_map:
                    rel = Relation(
                        head_id=from_id,
                        tail_id=to_id,
                        rel_type="ASSERTS",
                        weight=1.0,
                        confidence=1.0
                    )
                    relations_to_add.append(rel)

        # Fallback: Connect facts to events based on shared participants
        # If a fact's subject is a person who participates in an event, connect them
        for fact in facts:
            fact_id = fact.id
            subject_id = getattr(fact, 'subject', None)

            if subject_id and subject_id.startswith('person_'):
                # Find events where this person participates
                for event in events:
                    if subject_id in getattr(event, 'participants', []):
                        # Create ASSERTS relationship
                        rel = Relation(
                            head_id=event.id,
                            tail_id=fact_id,
                            rel_type="ASSERTS",
                            weight=1.0,
                            confidence=1.0
                        )
                        relations_to_add.append(rel)
                        break  # Connect to first matching event

        # Add facts and preferences to nodes
        for fact in facts:
            nodes_to_add.append(fact)

        for pref in preferences:
            nodes_to_add.append(pref)

        # Handle SUPERSEDES edges
        for edge in edges.get("supersedes", []):
            from_id = edge.get("source") or edge.get("from")
            to_id = edge.get("target") or edge.get("to")
            if from_id and to_id:
                rel = Relation(
                    head_id=from_id,
                    tail_id=to_id,
                    rel_type="SUPERSEDES",
                    weight=1.0,
                    confidence=1.0
                )
                relations_to_add.append(rel)

        # Handle SAME_AS edges for alias resolution
        for edge in edges.get("same_as", []):
            from_id = edge.get("source") or edge.get("from")
            to_id = edge.get("target") or edge.get("to")
            if from_id and to_id:
                rel = Relation(
                    head_id=from_id,
                    tail_id=to_id,
                    rel_type="SAME_AS",
                    weight=1.0,
                    confidence=1.0
                )
                relations_to_add.append(rel)

        # === ELEGANT IMPROVEMENT: Reference Integrity Check ===
        # Collect all entity IDs referenced in Facts and Events
        referenced_entity_ids = set()

        # From facts (subject and object fields)
        for fact in facts:
            if hasattr(fact, 'subject') and fact.subject and fact.subject.startswith('entity_'):
                referenced_entity_ids.add(fact.subject)
            if hasattr(fact, 'object') and fact.object and fact.object.startswith('entity_'):
                referenced_entity_ids.add(fact.object)

        # From events (participants and location)
        for event in events:
            if hasattr(event, 'participants'):
                for participant_id in event.participants:
                    if participant_id and participant_id.startswith('entity_'):
                        referenced_entity_ids.add(participant_id)
            if hasattr(event, 'location') and event.location and event.location.startswith('entity_'):
                referenced_entity_ids.add(event.location)

        # Create shadow entities for referenced but not explicitly extracted entities
        created_shadow_entities = 0
        for entity_id in referenced_entity_ids:
            if entity_id not in entity_map:
                # Extract entity name from ID (e.g., "entity_sweden" -> "Sweden")
                entity_name = entity_id.replace('entity_', '').replace('_', ' ').title()

                shadow_entity = EntityType(
                    id=entity_id,
                    name=entity_name,
                    description=f"Referenced entity: {entity_name}",
                    type="world",
                    subtype=None,  # Shadow Entity: cannot determine specific subtype
                    confidence=0.7,  # Lower confidence for inferred entities
                    metadata=Metadata(
                        source="inferred_from_reference",
                        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        source_session_id=session_id
                    )
                )
                nodes_to_add.append(shadow_entity)
                entity_map[entity_id] = shadow_entity
                created_shadow_entities += 1

        if created_shadow_entities > 0:
            print(f"[Indexer] Created {created_shadow_entities} shadow entities for reference integrity")

        # Add other structural edges
        for edge in edges.get("belongs_to", []):
            from_id = edge.get("source") or edge.get("from")
            to_id = edge.get("target") or edge.get("to")
            if from_id and to_id:
                rel = Relation(
                    head_id=from_id,
                    tail_id=to_id,
                    rel_type="BELONGS_TO",
                    weight=1.0,
                    confidence=1.0
                )
                relations_to_add.append(rel)

        for edge in edges.get("spoken_by", []):
            from_id = edge.get("source") or edge.get("from")
            to_id = edge.get("target") or edge.get("to")
            if from_id and to_id:
                rel = Relation(
                    head_id=from_id,
                    tail_id=to_id,
                    rel_type="SPOKEN_BY",
                    weight=1.0,
                    confidence=1.0
                )
                relations_to_add.append(rel)

        return nodes_to_add, relations_to_add

    def _create_person_facets(self, person_id: str, facts: List[Fact], preferences: List[Preference]) -> List[PersonFacet]:
        """Create PersonFacet nodes to partition person information"""
        if not getattr(self._cfg.extraction, 'use_person_facets', False):
            return []

        facets = []
        facet_types = ["work", "health", "finance", "relationships", "preferences", "tasks", "other"]

        # Group facts/preferences by facet type (simplified: based on predicate keywords)
        facet_keywords = {
            "work": ["work", "job", "company", "office", "career", "employer"],
            "health": ["health", "illness", "doctor", "hospital", "medicine", "sick"],
            "finance": ["money", "salary", "income", "bank", "payment", "cost"],
            "relationships": ["friend", "family", "spouse", "partner", "relationship", "marry"],
            "preferences": ["like", "love", "hate", "prefer", "favorite"],
            "tasks": ["task", "todo", "plan", "meeting", "appointment"]
        }

        for facet_type in facet_types:
            facet_facts = []
            facet_prefs = []

            keywords = facet_keywords.get(facet_type, [])
            for fact in facts:
                if fact.subject == person_id:
                    if any(kw in fact.predicate.lower() for kw in keywords):
                        facet_facts.append(fact.id)

            for pref in preferences:
                if pref.holder == person_id:
                    if any(kw in pref.aspect.lower() for kw in keywords):
                        facet_prefs.append(pref.id)

            if facet_facts or facet_prefs:
                facet = PersonFacet(
                    id=f"facet_{person_id}_{facet_type}",
                    person_id=person_id,
                    facet_type=facet_type,
                    description=f"{facet_type} related information for person {person_id}"
                )
                facets.append(facet)

        return facets

    def _group_by_time(self, events: List[Event]) -> Tuple[List[Episode], List[Relation]]:
        """按周（Week）为单位对 Events 进行分组，创建或查找 Episode

        Args:
            events: Event 节点列表

        Returns:
            Tuple of (episodes_to_add, relations_to_add)
            - episodes_to_add: Episode 节点列表（新建或已存在）
            - relations_to_add: CONTAINS 关系（Episode -> Event）
        """
        from datetime import datetime
        from collections import defaultdict
        import uuid

        episodes_to_add = []
        relations_to_add = []
        week_to_episode = {}  # (year, week_number) -> Episode
        week_to_participants = defaultdict(set)  # (year, week_number) -> set of person_ids

        # Group events by week
        for event in events:
            event_time = None
            if event.time_start:
                event_time = datetime.fromisoformat(event.time_start.replace('Z', '+00:00'))
            elif event.time_end:
                event_time = datetime.fromisoformat(event.time_end.replace('Z', '+00:00'))

            if not event_time:
                continue  # Skip events without time info

            # Get ISO week (Monday-based)
            year, week_number, _ = event_time.isocalendar()
            week_key = (year, week_number)

            # Calculate week start (Monday) and week end (Sunday)
            week_start = event_time - datetime.timedelta(days=event_time.weekday())
            week_end = week_start + datetime.timedelta(days=6)

            week_start_str = week_start.strftime("%Y-%m-%d")
            week_end_str = week_end.strftime("%Y-%m-%d")

            # Get or create episode for this week
            if week_key not in week_to_episode:
                episode_id = f"episode_{year}_week{week_number}"

                # Check if episode already exists in graph
                existing_episode = None
                if self._graph.get_node(episode_id):
                    try:
                        existing_episode_data = self._graph._g.nodes[episode_id]
                        existing_episode = Episode(**existing_episode_data)
                    except Exception:
                        existing_episode = None

                if existing_episode:
                    episode = existing_episode
                else:
                    # Create new episode
                    episode = Episode(
                        id=episode_id,
                        week_start=week_start_str,
                        week_end=week_end_str,
                        year=year,
                        week_number=week_number,
                        participant_ids=[],
                        event_ids=[],
                        metadata=Metadata(
                            source="time_based_grouping",
                            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                    )
                    episodes_to_add.append(episode)

                week_to_episode[week_key] = episode
            else:
                episode = week_to_episode[week_key]

            # Add event to episode
            episode.add_event(event.id)

            # Collect participants from this event
            for participant_id in event.participants:
                if participant_id.startswith('person_'):
                    week_to_participants[week_key].add(participant_id)

            # Create CONTAINS relationship: Episode -> Event
            contains_relation = Relation(
                head_id=episode.id,
                tail_id=event.id,
                rel_type="CONTAINS",
                weight=1.0,
                confidence=1.0,
                metadata=Metadata(
                    source="time_based_grouping",
                    created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            )
            relations_to_add.append(contains_relation)

        # Add participants to episodes
        for week_key, participants in week_to_participants.items():
            episode = week_to_episode[week_key]
            for person_id in participants:
                episode.add_participant(person_id)

        print(f"  按周分组：创建了 {len(episodes_to_add)} 个新的 episode，建立了 {len(relations_to_add)} 个 CONTAINS 关系")
        return episodes_to_add, relations_to_add

    def _batch_embed_nodes(self, nodes: List) -> None:
        """批量嵌入节点文本（步骤1：为所有节点类型生成向量）"""
        if not nodes or not self._vstore:
            return

        # 提取节点文本（步骤1改进：为每种节点类型生成富文本）
        node_texts = []
        for node in nodes:
            rich_text = self._generate_rich_node_text(node)
            node_texts.append(rich_text[:500])  # 截断到500字符

        if not node_texts:
            return

        print(f"  正在生成节点向量（共 {len(node_texts)} 个节点，这可能需要几分钟）...")
        try:
            # 批量编码（分批处理以显示进度）
            batch_size = 50  # 每批50个节点
            all_embeddings = []

            for i in tqdm(range(0, len(node_texts), batch_size), desc="节点向量化进度"):
                batch_texts = node_texts[i:i+batch_size]
                batch_nodes = nodes[i:i+batch_size]
                batch_embeddings = self._embed.encode(batch_texts)

                if batch_embeddings:
                    all_embeddings.extend(batch_embeddings)
                    # 立即添加到向量库（避免内存占用过大）
                    self._vstore.add(batch_nodes, batch_embeddings)

            print(f"  ✓ 节点向量化完成，共生成 {len(all_embeddings)} 个向量")
        except Exception as e:
            print(f"[!]  节点嵌入失败: {e}")
            # 如果是embedding维度问题，提供详细错误
            if "dimension" in str(e).lower():
                print("   提示: 检查embedding模型配置")

    def _generate_rich_node_text(self, node):
        """生成富文本表示，包含节点内容+连接的实体名（步骤1核心）"""
        from ..storage.schema import Person, Event, Fact, Preference, Entity

        parts = []

        # 根据节点类型生成富文本
        if isinstance(node, Event):
            # Event: "Melanie painted a sunrise in 2022 (Time: 2022-06-15)"
            parts.append(f"Event: {node.title}")
            if node.participants:
                parts.append(f"Participants: {', '.join(node.participants)}")
            if node.time_start:
                parts.append(f"Time: {node.time_start}")
            if node.location:
                parts.append(f"Location: {node.location}")

        elif isinstance(node, Fact):
            # Fact: "Caroline works_as software engineer"
            subject = getattr(node, 'subject', '')
            predicate = getattr(node, 'predicate', '')
            obj = getattr(node, 'object', '')
            value = getattr(node, 'value', '')

            parts.append(f"Fact: {subject} {predicate}")
            if obj:
                parts.append(obj)
            elif value:
                parts.append(str(value))

        elif isinstance(node, Preference):
            # Preference: "Caroline likes hiking"
            holder = getattr(node, 'holder', '')
            pref_type = getattr(node, 'preference_type', '')
            aspect = getattr(node, 'aspect', '')
            value = getattr(node, 'value', '')

            parts.append(f"Preference: {holder} {pref_type}")
            if aspect:
                parts.append(f"{aspect}")
            if value:
                parts.append(f"{value}")

        elif isinstance(node, Person):
            # Person: "Caroline (software engineer, LGBTQ+ activist)"
            parts.append(f"Person: {node.canonical_name}")
            if node.profession:
                parts.append(f"Profession: {node.profession}")
            if node.background:
                parts.append(f"Background: {node.background}")
            if node.identity:
                parts.append(f"Identity: {node.identity}")

        elif isinstance(node, Entity):
            # Entity: "Google (search engine company)"
            parts.append(f"Entity: {node.name}")
            if hasattr(node, 'description') and node.description:
                parts.append(f"Description: {node.description}")
            elif hasattr(node, 'content') and node.content:
                parts.append(f"Content: {node.content}")
            if node.subtype:
                parts.append(f"Type: {node.subtype}")

        else:
            # 其他节点类型
            if hasattr(node, 'name') and node.name:
                parts.append(str(node.name))
            elif hasattr(node, 'description') and node.description:
                parts.append(str(node.description))
            else:
                parts.append(str(node))

        return ' '.join(parts)
