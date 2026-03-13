"""端到端问答协调器：集成 SearchAgent 进行基于图谱的认知推理 (v3.1)

v3.1 更新：用 SearchAgent 替换旧版 RetrievalAgent，支持 P-E-C-F 架构
策略：
1. 使用 SearchAgent 的 ReAct 模式进行图谱推理
2. 核心能力：opinion queries, common interests, factual queries
3. 自动处理 CLAIMS 边，隔离主观评价
4. 支持 find_common_ground 查找共同兴趣
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cogmem import CogMem

from .search_agent import SearchAgent
from .tool_registry import ToolRegistry
from .skill_runtime import SkillRuntime
from ..storage.graph_service import GraphService


class QAOrchestrator:
    """端到端问答协调器：基于图谱推理的 SearchAgent + Skill Runtime"""

    def __init__(self, cogmem: CogMem):
        """
        Args:
            cogmem: CogMem实例 (提供 graph_store, vector_store, llm)
        """
        self.cogmem = cogmem

        # 创建 GraphService (底层图谱操作)
        use_vector_search = cogmem.cfg.faiss.use_vector_search if hasattr(cogmem.cfg.faiss, 'use_vector_search') else True
        self.graph_service = GraphService(
            graph_store=cogmem.graph,
            vector_store=cogmem.vstore,
            embedding_encoder=cogmem._embed,
            use_vector_search=use_vector_search,
            chunk_store=cogmem.chunk_store  # 传递ChunkStore用于Naive RAG fallback
        )

        # Runtime 层：ToolRegistry + SkillRuntime
        agent_cfg = getattr(cogmem.cfg, 'agent', None)
        enable_trace = getattr(agent_cfg, 'enable_trace', True)
        self.tool_registry = ToolRegistry(enable_trace=enable_trace)

        # 创建 SearchAgent (上层推理逻辑)，并注册工具到 registry
        max_iterations = cogmem.cfg.agent.max_iterations if hasattr(cogmem.cfg, 'agent') else 3
        self.search_agent = SearchAgent(
            graph_service=self.graph_service,
            llm=cogmem.llm,
            max_iterations=max_iterations,
            config=cogmem.cfg,
            evidence_store=cogmem.indexer.evidence,
            tool_registry=self.tool_registry,
        )

        # 装配 SkillRuntime（skill 开关从 config 读取，缺省为 True）
        self.skill_runtime = SkillRuntime(
            self.tool_registry,
            enable_routing_skill=getattr(agent_cfg, 'enable_routing_skill', True),
            enable_evidence_skill=getattr(agent_cfg, 'enable_evidence_skill', True),
            enable_temporal_skill=getattr(agent_cfg, 'enable_temporal_skill', True),
            enable_common_ground_skill=getattr(agent_cfg, 'enable_common_ground_skill', True),
            enable_trace=enable_trace,
            enable_runtime_cache=getattr(agent_cfg, 'enable_runtime_cache', True),
        )
        self.search_agent.skill_runtime = self.skill_runtime

    def answer(self, query: str, verbose: bool = False):
        """极简问答流程：SearchAgent 接管所有推理

        Args:
            query: 用户查询
            verbose: 是否打印推理过程

        Returns:
            Answer string (为了向后兼容，只返回答案字符串)
            如果需要完整的result（包含reasoning_path），请直接调用search_agent.run()
        """
        # 直接使用 SearchAgent 的 ReAct 推理
        result = self.search_agent.run(query, verbose=verbose)

        # 打印统计信息
        if verbose:
            print("\n" + "="*80)
            print(f"[QAOrchestrator] SearchAgent 推理完成:")
            print(f"  查询: {query}")
            print(f"  置信度: {result.get('confidence', 'unknown')}")
            print("  推理路径:")
            for step in result.get('reasoning_path', []):
                if isinstance(step, dict):
                    step_num = step.get('step', '?')
                    action = step.get('action', 'unknown')
                    print(f"    Step {step_num}: {action}")
                else:
                    # 如果是 ThoughtActionObservation 对象
                    print(f"    Step {step.step}: {step.action}")
            print("="*80 + "\n")

        # CRITICAL FIX: 保存完整的result到search_agent，供评测脚本访问
        # 注意：在并发情况下，last_result可能被覆盖，但这是可以接受的
        # 因为评测脚本应该在每次answer()调用后立即获取reasoning_path
        # 更好的方案是直接从result中获取，而不是依赖last_result
        self.search_agent.last_result = result

        # CRITICAL FIX: Extract and clean the answer from result
        answer = result.get('answer', 'I could not generate an answer.')
        
        # Parse JSON if the answer is a JSON string
        if isinstance(answer, str):
            import json
            import re
            
            # Try to parse as JSON string
            try:
                # Check if it looks like JSON
                answer_stripped = answer.strip()
                if answer_stripped.startswith('{') and answer_stripped.endswith('}'):
                    parsed = json.loads(answer_stripped)
                    # Extract final_answer field
                    if isinstance(parsed, dict):
                        # Try common field names
                        for field in ['final_answer', 'answer', 'response', 'message']:
                            if field in parsed:
                                value = parsed[field]
                                if isinstance(value, str):
                                    answer = value
                                    break
                                elif isinstance(value, dict):
                                    # Nested JSON, try to extract text
                                    for sub_field in ['message', 'text', 'content', 'answer']:
                                        if sub_field in value and isinstance(value[sub_field], str):
                                            answer = value[sub_field]
                                            break
                                    if answer != result.get('answer', ''):
                                        break
                                elif isinstance(value, list) and value:
                                    # List with first element
                                    first = value[0]
                                    if isinstance(first, dict):
                                        for sub_field in ['message', 'text', 'content', 'answer']:
                                            if sub_field in first and isinstance(first[sub_field], str):
                                                answer = first[sub_field]
                                                break
                                    elif isinstance(first, str):
                                        answer = first
                                    if answer != result.get('answer', ''):
                                        break
            except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
                # Not JSON or parsing failed, try to clean the string
                # Remove common prefixes like "Thought:", "Action:", etc.
                answer = re.sub(r'^(Thought|Action|Observation|Reflection):\s*', '', answer, flags=re.IGNORECASE | re.MULTILINE)
                answer = answer.strip()
        
        return answer
        self.cogmem = cogmem

        # 创建 GraphService (底层图谱操作)
        use_vector_search = cogmem.cfg.faiss.use_vector_search if hasattr(cogmem.cfg.faiss, 'use_vector_search') else True
        self.graph_service = GraphService(
            graph_store=cogmem.graph,
            vector_store=cogmem.vstore,
            embedding_encoder=cogmem._embed,
            use_vector_search=use_vector_search,
            chunk_store=cogmem.chunk_store  # 传递ChunkStore用于Naive RAG fallback
        )

        # Runtime 层：ToolRegistry + SkillRuntime
        agent_cfg = getattr(cogmem.cfg, 'agent', None)
        enable_trace = getattr(agent_cfg, 'enable_trace', True)
        self.tool_registry = ToolRegistry(enable_trace=enable_trace)

        # 创建 SearchAgent (上层推理逻辑)，并注册工具到 registry
        max_iterations = cogmem.cfg.agent.max_iterations if hasattr(cogmem.cfg, 'agent') else 3
        self.search_agent = SearchAgent(
            graph_service=self.graph_service,
            llm=cogmem.llm,
            max_iterations=max_iterations,
            config=cogmem.cfg,
            evidence_store=cogmem.indexer.evidence,
            tool_registry=self.tool_registry,
        )

        # 装配 SkillRuntime（skill 开关从 config 读取，缺省为 True）
        self.skill_runtime = SkillRuntime(
            self.tool_registry,
            enable_routing_skill=getattr(agent_cfg, 'enable_routing_skill', True),
            enable_evidence_skill=getattr(agent_cfg, 'enable_evidence_skill', True),
            enable_temporal_skill=getattr(agent_cfg, 'enable_temporal_skill', True),
            enable_common_ground_skill=getattr(agent_cfg, 'enable_common_ground_skill', True),
            enable_trace=enable_trace,
            enable_runtime_cache=getattr(agent_cfg, 'enable_runtime_cache', True),
        )
        self.search_agent.skill_runtime = self.skill_runtime

    def answer(self, query: str, verbose: bool = False):
        """极简问答流程：SearchAgent 接管所有推理

        Args:
            query: 用户查询
            verbose: 是否打印推理过程

        Returns:
            Answer string (为了向后兼容，只返回答案字符串)
            如果需要完整的result（包含reasoning_path），请直接调用search_agent.run()
        """
        # 直接使用 SearchAgent 的 ReAct 推理
        result = self.search_agent.run(query, verbose=verbose)

        # 打印统计信息
        if verbose:
            print("\n" + "="*80)
            print(f"[QAOrchestrator] SearchAgent 推理完成:")
            print(f"  查询: {query}")
            print(f"  置信度: {result.get('confidence', 'unknown')}")
            print("  推理路径:")
            for step in result.get('reasoning_path', []):
                if isinstance(step, dict):
                    step_num = step.get('step', '?')
                    action = step.get('action', 'unknown')
                    print(f"    Step {step_num}: {action}")
                else:
                    # 如果是 ThoughtActionObservation 对象
                    print(f"    Step {step.step}: {step.action}")
            print("="*80 + "\n")

        # CRITICAL FIX: 保存完整的result到search_agent，供评测脚本访问
        # 注意：在并发情况下，last_result可能被覆盖，但这是可以接受的
        # 因为评测脚本应该在每次answer()调用后立即获取reasoning_path
        # 更好的方案是直接从result中获取，而不是依赖last_result
        self.search_agent.last_result = result

        # CRITICAL FIX: Extract and clean the answer from result
        answer = result.get('answer', 'I could not generate an answer.')
        
        # Parse JSON if the answer is a JSON string
        if isinstance(answer, str):
            import json
            import re
            
            # Try to parse as JSON string
            try:
                # Check if it looks like JSON
                answer_stripped = answer.strip()
                if answer_stripped.startswith('{') and answer_stripped.endswith('}'):
                    parsed = json.loads(answer_stripped)
                    # Extract final_answer field
                    if isinstance(parsed, dict):
                        # Try common field names
                        for field in ['final_answer', 'answer', 'response', 'message']:
                            if field in parsed:
                                value = parsed[field]
                                if isinstance(value, str):
                                    answer = value
                                    break
                                elif isinstance(value, dict):
                                    # Nested JSON, try to extract text
                                    for sub_field in ['message', 'text', 'content', 'answer']:
                                        if sub_field in value and isinstance(value[sub_field], str):
                                            answer = value[sub_field]
                                            break
                                    if answer != result.get('answer', ''):
                                        break
                                elif isinstance(value, list) and value:
                                    # List with first element
                                    first = value[0]
                                    if isinstance(first, dict):
                                        for sub_field in ['message', 'text', 'content', 'answer']:
                                            if sub_field in first and isinstance(first[sub_field], str):
                                                answer = first[sub_field]
                                                break
                                    elif isinstance(first, str):
                                        answer = first
                                    if answer != result.get('answer', ''):
                                        break
            except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
                # Not JSON or parsing failed, try to clean the string
                # Remove common prefixes like "Thought:", "Action:", etc.
                answer = re.sub(r'^(Thought|Action|Observation|Reflection):\s*', '', answer, flags=re.IGNORECASE | re.MULTILINE)
                answer = answer.strip()
        
        return answer