"""SearchAgent: Cognitive Search Agent with ReAct Loop (v3.3 Enhanced)

v3.3 Enhancements:
- Question type classification for optimal tool selection
- Dynamic configuration based on question type
- Enhanced evidence grounding
- Improved tool suggestion based on question patterns
"""
from cogmem.models.execution_errors import raise_if_execution_failure

import json
import re
import time
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from cogmem.config import CogMemConfig, classify_question_type, QuestionType
from cogmem.storage.graph_service import GraphService
from cogmem.models.llm_client import LLMClient
from cogmem.utils.text_utils import extract_clean_answer, extract_json_from_markdown
from cogmem.utils.temporal import date_bounds, normalize_event_time
from cogmem.prompt import SEARCH_AGENT_STEP_PROMPT, SEARCH_AGENT_SYSTEM_PROMPT, SEARCH_AGENT_CAN_ANSWER_PROMPT

@dataclass
class ThoughtActionObservation:
    step: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    reflection: Optional[str] = None

class SearchAgent:
    """Cognitive Search Agent with ReAct Loop (v3.3 Enhanced)
    
    v3.3 Features:
    - Question type classification for optimal tool selection
    - Dynamic configuration based on question type
    - Enhanced evidence grounding requirement
    - Tool suggestion based on question patterns
    """
    
    def __init__(self, graph_service: GraphService, llm: LLMClient, max_iterations: int = 10, config: Optional[CogMemConfig] = None, evidence_store=None, tool_registry=None):
        self.graph_service = graph_service
        self.llm = llm
        self.cfg = config
        self.max_iterations = max_iterations
        self.max_tool_calls = getattr(config.agent, "max_tool_calls", None) if config else None
        self.evidence_store = evidence_store  # EvidenceStore 实例，用于获取实体的原文证据
        self.tool_registry = tool_registry  # 可选：ToolRegistry，用于 trace/消融
        self.skill_runtime = None  # 由 QAOrchestrator 注入 SkillRuntime
        
        # 从配置中获取搜索参数（如果配置存在）
        if config and hasattr(config, 'agent'):
            self.semantic_search_top_k = config.agent.semantic_search_top_k
            self.find_anchors_top_k = config.agent.find_anchors_top_k
            self.traverse_neighbors_max_results = config.agent.traverse_neighbors_max_results
            self.traverse_neighbors_max_depth = config.agent.traverse_neighbors_max_depth
            self.find_common_ground_top_k = config.agent.find_common_ground_top_k
            self.vector_search_expansion_factor = config.agent.vector_search_expansion_factor
            # v3.3: Evidence grounding settings
            self.require_evidence_for_answer = getattr(config.agent, 'require_evidence_for_answer', True)
            self.auto_get_evidence = getattr(config.agent, 'auto_get_evidence', True)
        else:
            # 默认值（基于 Qwen2.5-14B-Instruct 的 128K context window）
            self.semantic_search_top_k = 8
            self.find_anchors_top_k = 5
            self.traverse_neighbors_max_results = 10
            self.traverse_neighbors_max_depth = 1
            self.find_common_ground_top_k = 5
            self.vector_search_expansion_factor = 2.0
            self.require_evidence_for_answer = True
            self.auto_get_evidence = True
        
        self.system_prompt = SEARCH_AGENT_SYSTEM_PROMPT.format(
            max_iterations=self.max_iterations,
            tools_description=self._get_tools_description()
        )
        self.tools = {
            "semantic_search": self._tool_semantic_search,  # 核心工具：语义搜索
            "find_anchors": self._tool_find_anchors,  # 简化版：实体查找
            "traverse_neighbors": self._tool_traverse_neighbors,  # 关系遍历（支持时间查询）
            "timeline_query": self._tool_timeline_query,  # 显式时间线查询
            "fact_lookup": self._tool_fact_lookup,  # 事实直达查询
            "claim_lookup": self._tool_claim_lookup,  # 观点直达查询
            "find_common_ground": self._tool_find_common_ground,  # 多跳查询
            "get_evidence": self._tool_get_evidence,  # 获取实体的原文证据
        }
        if tool_registry is not None:
            for tool_name, fn in self.tools.items():
                tool_registry.register(tool_name, fn)
        self.history: List[ThoughtActionObservation] = []
        # v3.3: Track question type for dynamic configuration
        self._current_question_type: Optional[QuestionType] = None
        self._evidence_collected: bool = False  # Track if evidence has been collected
        self._current_query: str = ""  # 当前 run() 的 query，供 runtime trace 使用
        self._last_tool_node_hits: Dict[str, List[str]] = {}

    def _apply_question_type_config(self, question_type: QuestionType) -> None:
        """Apply dynamic configuration based on question type (v3.3 Enhancement)"""
        if self.cfg and hasattr(self.cfg, 'agent'):
            overrides = self.cfg.agent.get_config_for_question_type(question_type)
            for key, value in overrides.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def _reset_runtime_config(self) -> None:
        """Reset mutable runtime search knobs from config before each run."""
        if self.cfg and hasattr(self.cfg, 'agent'):
            agent_cfg = self.cfg.agent
            self.max_iterations = agent_cfg.max_iterations
            self.max_tool_calls = getattr(agent_cfg, "max_tool_calls", None)
            self.semantic_search_top_k = agent_cfg.semantic_search_top_k
            self.find_anchors_top_k = agent_cfg.find_anchors_top_k
            self.traverse_neighbors_max_results = agent_cfg.traverse_neighbors_max_results
            self.traverse_neighbors_max_depth = agent_cfg.traverse_neighbors_max_depth
            self.find_common_ground_top_k = agent_cfg.find_common_ground_top_k
            self.vector_search_expansion_factor = agent_cfg.vector_search_expansion_factor
            self.require_evidence_for_answer = getattr(agent_cfg, 'require_evidence_for_answer', True)
            self.auto_get_evidence = getattr(agent_cfg, 'auto_get_evidence', True)

    def _operator_budget(self) -> int:
        """Planning rounds and operator executions have different units.

        Allow initial anchors + seeded retrieval/source (3 operations), then a
        retrieval/source pair per planning round. Extra recovery operations also
        consume this finite budget. An explicit caller limit overrides it.
        """
        explicit = getattr(self, 'max_tool_calls', None)
        if explicit is not None:
            if not isinstance(explicit, int) or isinstance(explicit, bool) or explicit < 1:
                raise ValueError('max_tool_calls must be a positive integer')
            return explicit
        return 3 + 2 * max(0, int(self.max_iterations))

    def _enabled_skill_names(self) -> List[str]:
        """Return currently enabled skill names."""
        if self.skill_runtime is None:
            return []
        return self.skill_runtime.enabled_skill_names()

    def _build_skill_context(self) -> Any:
        """Build runtime skill context for this query."""
        from .runtime_schema import SkillContext
        return SkillContext(
            query=getattr(self, "_current_query", ""),
            question_type=str(self._current_question_type) if self._current_question_type else None,
            enabled_skills=self._enabled_skill_names(),
            budgets={
                "max_iterations": self.max_iterations,
                "max_tool_calls": self._operator_budget(),
                "semantic_search_top_k": self.semantic_search_top_k,
                "find_anchors_top_k": self.find_anchors_top_k,
                "traverse_neighbors_max_results": self.traverse_neighbors_max_results,
                "traverse_neighbors_max_depth": self.traverse_neighbors_max_depth,
                "find_common_ground_top_k": self.find_common_ground_top_k,
            },
            flags=self.skill_runtime.skill_flags() if self.skill_runtime is not None else {},
        )

    def _run_skill(self, name: str) -> Any:
        """Execute a skill when runtime is available."""
        if self.skill_runtime is None:
            return None
        return self.skill_runtime.execute_skill(name, self._build_skill_context())

    def _tool_skill_name(self, name: str, params: Dict[str, Any]) -> Optional[str]:
        """Map tool call to controlling skill when applicable."""
        if name == "get_evidence":
            return "evidence_retrieval_skill"
        if name == "timeline_query":
            return "temporal_grounding_skill"
        if name == "find_common_ground":
            return "common_ground_skill"
        if name == "traverse_neighbors" and params.get("time_query"):
            return "temporal_grounding_skill"
        return None

    def _effective_agent_config(self) -> Dict[str, Any]:
        """Expose effective knobs used for the current run."""
        return {
            "max_iterations": self.max_iterations,
            "max_tool_calls": self._operator_budget(),
            "semantic_search_top_k": self.semantic_search_top_k,
            "find_anchors_top_k": self.find_anchors_top_k,
            "traverse_neighbors_max_results": self.traverse_neighbors_max_results,
            "traverse_neighbors_max_depth": self.traverse_neighbors_max_depth,
            "find_common_ground_top_k": self.find_common_ground_top_k,
            "vector_search_expansion_factor": self.vector_search_expansion_factor,
            "require_evidence_for_answer": self.require_evidence_for_answer,
            "auto_get_evidence": self.auto_get_evidence,
        }

    def _current_routing_plan(self, routing_result: Any) -> Dict[str, Any]:
        if routing_result is None:
            return {}
        output = getattr(routing_result, "output", {}) or {}
        return output if isinstance(output, dict) else {}

    def _format_anchor_seed(self, metadata: Dict[str, Any], node_id: str) -> str:
        node_type = metadata.get("node_type")
        if node_type == "person":
            return f"Person: {metadata.get('canonical_name', node_id)}"
        if node_type == "concept":
            return f"Concept: {self._format_node_to_rich_text(metadata, node_id)}"
        if node_type == "event":
            title = metadata.get("title") or metadata.get("name") or node_id
            return f"Event: {title}"
        if node_type == "fact":
            return f"Fact: {metadata.get('content') or metadata.get('description') or node_id}"
        if node_type == "claim":
            return f"Claim: {metadata.get('content') or metadata.get('description') or node_id}"
        return self._format_node_to_rich_text(metadata, node_id)

    def _has_substantive_retrieval(self, history: List[ThoughtActionObservation]) -> bool:
        substantive_actions = {"semantic_search", "fact_lookup", "claim_lookup", "timeline_query", "find_common_ground", "traverse_neighbors"}
        for step in history:
            if step.action in substantive_actions:
                return True
        return False

    def _remember_tool_hits(self, tool_name: str, node_ids: List[str]) -> None:
        cleaned = []
        for node_id in node_ids:
            if node_id and node_id not in cleaned:
                cleaned.append(node_id)
        self._last_tool_node_hits[tool_name] = cleaned

    def _select_evidence_target(self, anchor_results: List[Any], history: List[ThoughtActionObservation]) -> Optional[str]:
        for step in reversed(history):
            params = step.action_input or {}
            if step.action in {"timeline_query", "semantic_search", "fact_lookup", "claim_lookup", "find_common_ground", "traverse_neighbors"}:
                recent_hits = self._last_tool_node_hits.get(step.action, [])
                if recent_hits:
                    return recent_hits[0]
            if step.action == "get_evidence" and params.get("entity_id"):
                return params.get("entity_id")
            if step.action == "fact_lookup":
                subject_id = params.get("subject_id") or self._resolve_entity_id(params.get("subject", ""), types=["person"])
                if subject_id:
                    return subject_id
            if step.action == "claim_lookup":
                target_id = params.get("target_id") or self._resolve_entity_id(params.get("target", ""), types=["person", "concept"])
                if target_id:
                    return target_id
            if step.action == "timeline_query":
                entity_ids = params.get("entity_ids") or []
                if entity_ids:
                    return entity_ids[0]
                if params.get("entity_id"):
                    return params.get("entity_id")
            if step.action == "find_common_ground":
                entity_ids = params.get("entity_ids") or []
                if entity_ids:
                    return entity_ids[0]
        if anchor_results:
            return getattr(anchor_results[0], "node_id", None)
        return None

    def _build_seed_retrieval_plan(self, query: str, anchor_results: List[Any], routing_plan: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any], str]]:
        person_ids = []
        for anchor in anchor_results:
            if getattr(anchor, "node_type", None) == "person" and anchor.node_id not in person_ids:
                person_ids.append(anchor.node_id)

        primary_action = routing_plan.get("primary_action") or "semantic_search"
        semantic_node_types = routing_plan.get("semantic_node_types") or ["fact", "event"]

        if primary_action == "find_common_ground" and len(person_ids) >= 2:
            return (
                "find_common_ground",
                {"entity_ids": person_ids[:2], "node_types": routing_plan.get("node_types", ["concept", "event", "fact"])},
                "Routing skill selected shared-grounding retrieval before answer synthesis.",
            )

        if primary_action == "timeline_query" and person_ids:
            return (
                "timeline_query",
                {"entity_ids": person_ids[:1], "time_query": routing_plan.get("time_query", "")},
                "Routing skill selected temporal retrieval conditioned on the question time expression.",
            )

        if primary_action == "claim_lookup" and person_ids:
            params: Dict[str, Any] = {"target_id": person_ids[-1], "limit": 6}
            if len(person_ids) >= 2:
                params["source_id"] = person_ids[0]
            return (
                "claim_lookup",
                params,
                "Routing skill selected claim retrieval to gather speaker-attributed judgments.",
            )

        if primary_action == "fact_lookup" and person_ids:
            return (
                "fact_lookup",
                {"subject_id": person_ids[0], "latest_only": False, "limit": 6, "query": query},
                "Routing skill selected direct fact retrieval for grounded attribute lookup.",
            )

        return (
            "semantic_search",
            {"query": query, "node_types": semantic_node_types, "top_k": min(self.semantic_search_top_k, 6)},
            "Routing skill selected query-conditioned semantic retrieval before answer synthesis.",
        )

    def _suggest_tool_for_question(self, query: str, question_type: QuestionType) -> str:
        """Suggest optimal tool based on question type and patterns (v3.3 Enhancement)
        
        Returns a hint to include in the system prompt for better tool selection.
        """
        query_lower = query.lower()
        
        suggestions = []
        
        if question_type == "multi_hop":
            suggestions.append("HINT: This appears to be a multi-hop question. Consider using find_common_ground to find shared concepts/events between entities.")
        
        elif question_type == "temporal":
            suggestions.append("HINT: This appears to be a temporal question. Use traverse_neighbors with time_query parameter to filter events by time.")
            # Extract potential time values
            year_match = re.search(r'\b(19|20)\d{2}\b', query)
            if year_match:
                suggestions.append(f"HINT: Detected year '{year_match.group()}' - use this in time_query.")
        
        elif question_type == "single_hop":
            suggestions.append("HINT: This appears to be a single-hop question. semantic_search should find the answer directly. Remember to use get_evidence before final_answer.")
        
        # Pattern-specific suggestions
        if any(kw in query_lower for kw in ['like', 'prefer', 'enjoy', 'habit', 'hobby']):
            suggestions.append("HINT: This question is about preferences/habits. Search for node_types=['fact'] to find relevant facts.")
        
        if any(kw in query_lower for kw in ['think', 'believe', 'opinion', 'feel about']):
            suggestions.append("HINT: This question is about opinions/beliefs. Search for node_types=['claim'] to find relevant claims.")
        
        return "\n".join(suggestions) if suggestions else ""

    def _get_tools_description(self) -> str:
        return f"""
1. semantic_search(query: str, node_types: List[str] = None, top_k: int = {self.semantic_search_top_k})
   - **CORE TOOL**: Semantic search based on question meaning (like RAG).
   - Use this for most questions - it finds relevant nodes based on semantic similarity.
   - Example: semantic_search("What did X research?", node_types=["fact"])
   - This is the PRIMARY tool for answering questions - use it first!

2. find_anchors(query: str, types: List[str] = None, top_k: int = 5)
   - Find entity nodes by name/ID (Person, Concept, Event).
   - Use this when you need to find a specific entity first.
   - Example: find_anchors("Alice", types=["person"])

3. timeline_query(entity_ids: List[str] = None, entity_id: str = "", time_query: str = "", relation_types: List[str] = None)
   - Explicit timeline lookup for temporal questions.
   - Use this first for "when", "before", "after", "during", or date/time questions.
   - Supports time values (e.g. 2023, 2023-05) and rough natural-language time phrases.

4. fact_lookup(subject_id: str = "", subject: str = "", predicate: str = "", object_id: str = "", at_time: str = "", latest_only: bool = False, query: str = "")
   - Directly retrieve Fact nodes for a person/concept pair.
   - Use this for preferences, habits, stable attributes, and "what does X like/do/use" questions.

5. claim_lookup(target_id: str = "", target: str = "", source_id: str = "", source: str = "", sentiment: str = "")
   - Directly retrieve Claim nodes about a target person/concept.
   - Use this for "what does X think about Y", opinions, beliefs, and judgments.

6. traverse_neighbors(node_id: str, relation_types: List[str] = None, direction: str = "out", max_depth: int = 1, time_query: str = "")
   - Explore graph connections from a node.
   - If time_query is provided, filters and sorts events by time.
   - Useful relation_types: "PARTICIPATED_IN", "CLAIMS", "HAS_ATTRIBUTE", "LIKES".

7. find_common_ground(entity_ids: List[str], node_types: List[str] = ["concept", "event"])
   - Find shared concepts/events between multiple entities (MANDATORY for multi-hop questions).
   - Use this for questions with "both", "shared", "common", etc.

8. get_evidence(entity_id: str, limit: int = 3, query: str = None)
   - Get original text evidence (chunks) for an entity from the EvidenceStore.
   - Use this when you need to see the original text where an entity was mentioned.
   - Example: get_evidence("p_alice", limit=3) to get original text chunks mentioning Alice.
   - This is useful for verifying answers or getting more context about an entity.
"""

    def _looks_like_tool_protocol(self, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return False
        upper = stripped.upper()
        return (
            upper.startswith("TOOL:")
            or upper.startswith("ARGS:")
            or upper.startswith("FINAL_ANSWER:")
            or upper.startswith("THOUGHT:")
            or upper.startswith("ACTION:")
            or upper.startswith("OBSERVATION:")
        )

    def _parse_step_response(self, response: str) -> Dict[str, Any]:
        text = (response or "").strip()
        if not text:
            return {}

        data = extract_json_from_markdown(text)
        if not data:
            try:
                data = json.loads(text)
            except Exception as execution_error:
                raise_if_execution_failure(execution_error)
                data = {}

        if isinstance(data, dict) and data:
            normalized = dict(data)
            if "tool" in normalized and "action" not in normalized:
                normalized["action"] = normalized.pop("tool")
            if "args" in normalized and "params" not in normalized:
                normalized["params"] = normalized.pop("args")
            if "action_input" in normalized and "params" not in normalized:
                normalized["params"] = normalized.pop("action_input")
            # A JSON object inside ARGS is not an action envelope. In that
            # case continue parsing TOOL/ARGS rather than discarding TOOL.
            if "action" in normalized or "final_answer" in normalized:
                return normalized

        final_match = re.search(r"FINAL_ANSWER\s*:\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
        if final_match:
            return {"final_answer": final_match.group(1).strip()}

        tool_match = re.search(r"TOOL\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", text, re.IGNORECASE)
        if not tool_match:
            return {}

        params: Dict[str, Any] = {}
        args_match = re.search(r"ARGS\s*:\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
        if args_match:
            raw_args = args_match.group(1).strip()
            params = extract_json_from_markdown(raw_args)
            if not params:
                try:
                    params = json.loads(raw_args)
                except Exception as execution_error:
                    raise_if_execution_failure(execution_error)
                    params = {}

        return {
            "action": tool_match.group(1),
            "params": params if isinstance(params, dict) else {},
            "thought": "",
        }

    def run(self, query: str, verbose: bool = False) -> Dict[str, Any]:
        """执行 ReAct 推理循环 (v3.3 Enhanced)
        
        v3.3 Enhancements:
        - Question type classification at start
        - Dynamic configuration based on question type
        - Tool suggestions in system prompt
        - Evidence grounding enforcement
        """
        local_history: List[ThoughtActionObservation] = []
        step_scratchpad: List[str] = []
        step_counter = 0
        retry_state = {
            "semantic_retry_used": False,
            "event_retry_used": False,
            "timeline_fallback_used": False,
            "common_ground_followup_used": False,
        }
        self._current_query = query
        self._last_tool_node_hits = {}
        self._tool_calls_used = 1  # Initial anchor lookup is an operator call.
        self._planning_rounds_used = 0

        if self.skill_runtime is not None:
            self.skill_runtime.clear_trace()
        self._reset_runtime_config()

        # v3.3: Classify question type and apply dynamic configuration
        self._current_question_type = classify_question_type(query)
        self._evidence_collected = False
        routing_result = self._run_skill("routing_skill")
        routing_plan = self._current_routing_plan(routing_result)
        if routing_result is None or routing_result.triggered:
            self._apply_question_type_config(self._current_question_type)
        self.system_prompt = SEARCH_AGENT_SYSTEM_PROMPT.format(
            max_iterations=self.max_iterations, tools_description=self._get_tools_description()
        ) + f"\nOperator executions have a separate cap of {self._operator_budget()}, including automatic source retrieval."
        
        if verbose:
            print(f"\n[SearchAgent] Starting ReAct loop for query: {query}")
            print(f"[SearchAgent] Question type: {self._current_question_type}")
        
        # v3.3: Generate tool suggestion based on question type
        tool_suggestion = ""
        if routing_result is None or routing_result.triggered:
            tool_suggestion = self._suggest_tool_for_question(query, self._current_question_type)
        
        # Step 0: Pre-fetching Anchors
        anchor_results = self.graph_service.find_anchors(query, types=None, top_k=self.find_anchors_top_k)
        readable_anchors = []
        for anchor in anchor_results:
            metadata = anchor.metadata or {}
            if metadata.get("node_type") == "chunk":
                readable_anchors.append(f"[ID: {anchor.node_id}] Raw Chunk: {metadata.get('text', 'unknown')[:300]}...")
            else:
                readable_anchors.append(f"[ID: {anchor.node_id}] {self._format_anchor_seed(metadata, anchor.node_id)}")

        obs_str = "\n".join(readable_anchors) if readable_anchors else "No anchors found"
        
        # v3.3: Include tool suggestion in initial observation
        if tool_suggestion:
            obs_str = f"{obs_str}\n\n{tool_suggestion}"
        
        step_counter, early_result = self._record_and_maybe_finalize(
            query=query,
            history=local_history,
            scratchpad=step_scratchpad,
            step_counter=step_counter,
            thought="Initial search",
            action="find_anchors",
            action_input={"query": query},
            observation=obs_str,
        )
        if early_result is not None:
            return early_result

        seed_plan = self._build_seed_retrieval_plan(query, anchor_results, routing_plan)
        if seed_plan is not None:
            seed_action, seed_params, seed_thought = seed_plan
            if verbose:
                print(f"[SearchAgent] Seed retrieval via {seed_action}: {seed_params}")
            seed_obs = self._execute_tool(seed_action, seed_params)
            if seed_action == "get_evidence":
                self._evidence_collected = True
            step_counter, early_result = self._record_and_maybe_finalize(
                query=query,
                history=local_history,
                scratchpad=step_scratchpad,
                step_counter=step_counter,
                thought=seed_thought,
                action=seed_action,
                action_input=seed_params,
                observation=seed_obs,
            )
            if early_result is not None:
                return early_result

        # Question-type-specific budgets are already applied in _apply_question_type_config().
        effective_max_iterations = max(1, int(self.max_iterations))
        
        for round_idx in range(1, effective_max_iterations + 1):
            if self._tool_calls_used >= self._operator_budget():
                break
            history_str = "\n".join(step_scratchpad)
            # v3.3: 使用动态的最大迭代次数
            user_msg = SEARCH_AGENT_STEP_PROMPT.format(
                query=query,
                history=history_str,
                tools=self._get_tools_description()
            )

            if verbose:
                print(f"\n[Round {round_idx}] Calling LLM...")

            self._planning_rounds_used += 1
            resp = self.llm.chat(self.system_prompt, user_msg)

            if verbose:
                print(f"\n[Round {round_idx}] Response:\n{resp}")

            data = self._parse_step_response(resp)
            if "final_answer" in data:
                injected_result = self._maybe_inject_required_shared_evidence(
                    query=query,
                    history=local_history,
                    scratchpad=step_scratchpad,
                    step_counter=step_counter,
                    anchor_results=anchor_results,
                )
                if injected_result is not None:
                    step_counter, early_result = injected_result
                    if early_result is not None:
                        return early_result
                    continue
                # v3.3: Check if evidence was collected (for single_hop and adversarial questions)
                if self.require_evidence_for_answer and not self._evidence_collected:
                    if self._current_question_type in ["single_hop", "adversarial"]:
                        # Force get_evidence before final answer
                        if verbose:
                            print(f"[SearchAgent] Evidence required but not collected. Auto-fetching evidence...")
                        
                        # Try to get evidence for the first anchor
                        if anchor_results:
                            first_anchor_id = self._select_evidence_target(anchor_results, local_history)
                            evidence_result = self._run_skill("evidence_retrieval_skill")
                            if first_anchor_id and (evidence_result is None or evidence_result.triggered):
                                evidence_obs = self._execute_tool("get_evidence", {"entity_id": first_anchor_id, "limit": 3, "query": query})
                                step_counter += 1
                                local_history.append(ThoughtActionObservation(
                                    step=step_counter,
                                    thought="Auto-fetching evidence before final answer",
                                    action="get_evidence",
                                    action_input={"entity_id": first_anchor_id},
                                    observation=evidence_obs
                                ))
                                step_scratchpad.append(f"Step {step_counter}:\nThought: Auto-fetching evidence\nAction: get_evidence\nObservation: {evidence_obs}\n---")
                                self._evidence_collected = True
                
                raw_ans = extract_clean_answer(data["final_answer"], query)
                confidence = data.get("confidence", "medium")
                raw_ans, confidence = self._postprocess_final_answer(
                    query, raw_ans, confidence, local_history
                )
                return self._return_run_result(local_history, raw_ans, confidence)

            if "action" in data:

                action = self._validate_and_fix_tool_name(data["action"])
                params = data.get("params", {})
                thought = data.get("thought", "")
                
                if verbose:
                    print(f"\n[Action] {action} with {params}")
                
                obs = self._execute_tool(action, params)
                
                # v3.3: Track if evidence was collected
                if action == "get_evidence":
                    self._evidence_collected = True
                
                recovery_result = self._handle_semantic_search_recovery(
                    query=query,
                    action=action,
                    params=params,
                    obs=obs,
                    round_idx=round_idx,
                    effective_max_iterations=effective_max_iterations,
                    anchor_results=anchor_results,
                    history=local_history,
                    scratchpad=step_scratchpad,
                    step_counter=step_counter,
                    retry_state=retry_state,
                    verbose=verbose,
                )
                if recovery_result is not None:
                    step_counter, maybe_result, should_continue = recovery_result
                    if maybe_result is not None:
                        return maybe_result
                    if should_continue:
                        continue

                # Multi-hop convergence: after common_ground succeeds, allow exactly one focused recall
                if (
                    action == "find_common_ground"
                    and not retry_state["common_ground_followup_used"]
                    and self._current_question_type == "multi_hop"
                    and obs
                    and "Common Nodes Found" in obs
                    and round_idx < effective_max_iterations
                ):
                    retry_state["common_ground_followup_used"] = True
                    focused_obs = self._execute_tool("semantic_search", {
                        "query": query,
                        "node_types": ["fact", "event"],
                        "top_k": min(6, self.semantic_search_top_k),
                    })
                    step_counter, early_result = self._record_and_maybe_finalize(
                        query=query,
                        history=local_history,
                        scratchpad=step_scratchpad,
                        step_counter=step_counter,
                        thought="Auto-injected: common ground found. Running one focused recall to converge on the answer.",
                        action="semantic_search",
                        action_input={"query": query, "node_types": ["fact", "event"]},
                        observation=focused_obs,
                    )
                    if early_result is not None:
                        return early_result
                    continue
                
                step_counter, early_result = self._record_and_maybe_finalize(
                    query=query,
                    history=local_history,
                    scratchpad=step_scratchpad,
                    step_counter=step_counter,
                    thought=thought,
                    action=action,
                    action_input=params,
                    observation=obs,
                )
                if early_result is not None:
                    return early_result
            else:
                # 容错：如果 LLM 没有返回 JSON 格式但看起来像是答案
                if round_idx > 2 and not self._looks_like_tool_protocol(resp):
                    ans = extract_clean_answer(resp, query)
                    if ans:
                        ans, confidence = self._postprocess_final_answer(query, ans, "low", local_history)
                        return self._return_run_result(local_history, ans, confidence)

        # 如果达到最大迭代次数仍未得到答案，使用最后一次观察作为答案
        last_observation = local_history[-1].observation if local_history else ""
        final_fallback_answer = extract_clean_answer(last_observation, query) or ""

        # P1 FIX: Evidence Grounding enforcement at end of reasoning loop
        # If evidence was not collected during reasoning, try to get it now
        if self.require_evidence_for_answer and not self._evidence_collected:
            if self._current_question_type in ["single_hop", "temporal", "adversarial"]:
                if verbose:
                    print(f"[SearchAgent] Evidence required but not collected at end. Attempting final evidence fetch...")
                
                # Try to get evidence for the first anchor or most relevant node
                if anchor_results:
                    first_anchor_id = self._select_evidence_target(anchor_results, local_history)
                    try:
                        evidence_result = self._run_skill("evidence_retrieval_skill")
                        if not first_anchor_id:
                            evidence_obs = ""
                        elif evidence_result is not None and not evidence_result.triggered:
                            evidence_obs = "Skill Disabled: evidence_retrieval_skill"
                        else:
                            evidence_obs = self._execute_tool("get_evidence", {"entity_id": first_anchor_id, "limit": 3, "query": query})
                        if evidence_obs and "No original text evidence found" not in evidence_obs and "Skill Disabled:" not in evidence_obs:
                            step_counter += 1
                            local_history.append(ThoughtActionObservation(
                                step=step_counter,
                                thought="Final evidence fetch before fallback answer",
                                action="get_evidence",
                                action_input={"entity_id": first_anchor_id},
                                observation=evidence_obs
                            ))
                            self._evidence_collected = True
                            # Update final answer to include evidence
                            if evidence_obs and len(evidence_obs) > 50:
                                final_fallback_answer = f"{final_fallback_answer}\n\nSupporting evidence: {evidence_obs[:300]}..."
                    except Exception as e:
                        raise_if_execution_failure(e)
                        if verbose:
                            print(f"[SearchAgent] Failed to fetch evidence: {e}")

        final_fallback_answer, final_confidence = self._postprocess_final_answer(
            query, final_fallback_answer, "low", local_history
        )
        if not self._has_substantive_retrieval(local_history):
            final_fallback_answer = ""
        elif self._looks_like_tool_protocol(final_fallback_answer):
            final_fallback_answer = ""
        
        return self._return_run_result(
            local_history,
            final_fallback_answer or "I don't have enough information.",
            final_confidence,
        )

    def _format_node_to_rich_text(self, node: Dict[str, Any], node_id: str = None) -> str:
        ntype = node.get("node_type") or node.get("type", "unknown")
        if ntype == "person":
            parts = [f"Person: {node.get('canonical_name', 'unknown')}"]
            if node.get("description"): parts.append(f"Description: {node['description']}")
            # Update to new enrichment keys
            if node.get("known_facts"): parts.append(f"Known Facts: {node['known_facts']}")
            if node.get("event_previews"): parts.append(f"Recent Event Previews: {node['event_previews']} (Use timeline_query for details)")
            return " | ".join(parts)
        elif ntype == "event":
            parts = [f"Event: {node.get('title', 'unknown')}"]
            if node.get("description"): parts.append(f"Desc: {node['description'][:150]}")
            # P0 FIX: 优先显示原始时间表达式（time_start_raw），如果没有则显示解析后的时间（time_start）
            if node.get("time_start_raw"):
                parts.append(f"Occurred: {node['time_start_raw']}")
            elif node.get("time_start"):
                parts.append(f"Occurred: {node['time_start']}")
            if node.get("session_time"): 
                parts.append(f"Recorded At: {node['session_time']} (Note: Log date, NOT necessarily event date)")
            if node.get("location"): parts.append(f"Location: {node['location']}")
            return " | ".join(parts)
        elif ntype == "fact":
            # P0 CRITICAL FIX: 优先使用content字段（已包含自然语言），如果没有则构建
            if node.get("content"):
                # 使用改进后的自然语言content字段
                return f"Fact: {node.get('content')}"
            else:
                # 回退到旧逻辑（兼容性）
                # ROOT FIX: 确保从Fact节点提取完整信息（subject_id, predicate, object_id）
                subject_id = node.get("subject_id", node.get("subject", ""))
                predicate = node.get("predicate", "")
                object_id = node.get("object_id", node.get("object", ""))
                
                # ROOT FIX: 如果object_id是Concept ID，转换为自然语言
                object_display = object_id
                if object_id and (object_id.startswith("c_") or "_" in object_id):
                    # 尝试从图谱获取Concept节点
                    try:
                        concept_node = self.graph_service.graph_store.get_node(object_id)
                        if concept_node and concept_node.get("node_type") == "concept":
                            concept_name = concept_node.get("name", object_id)
                            # 转换为自然语言
                            if "_" in concept_name:
                                object_display = concept_name.replace("_", " ").title()
                            else:
                                object_display = concept_name
                        else:
                            # 如果无法获取，从ID推断
                            object_display = object_id.replace("c_", "").replace("_", " ").title()
                    except:
                        # 如果出错，从ID推断
                        object_display = object_id.replace("c_", "").replace("_", " ").title()
                
                return f"Fact: {subject_id} {predicate} {object_display}"
        elif ntype == "claim":
            parts = [f"Claim: {node.get('content', '')}"]
            if node.get("source"): parts.append(f"Source: {node['source']}")
            if node.get("target"): parts.append(f"Target: {node['target']}")
            if node.get("sentiment"): parts.append(f"Sentiment: {node['sentiment']}")
            return " | ".join(parts)
        elif ntype == "chunk":
            return f"Raw Evidence: {node.get('text', '')[:300]}..."
        elif ntype == "concept":
            # P0 FIX: 将下划线格式的 concept 名称转换为自然语言
            concept_name = node.get('name', node.get('title', 'unknown'))
            # 如果名称包含下划线，转换为自然语言（下划线转空格，首字母大写）
            if '_' in concept_name and not concept_name.startswith('c_'):
                # 已经是自然语言格式，直接使用
                natural_name = concept_name.replace('_', ' ').title()
            elif concept_name.startswith('c_'):
                # 去除 c_ 前缀并转换
                natural_name = concept_name.replace('c_', '').replace('_', ' ').title()
            else:
                natural_name = concept_name
            return f"Concept: {natural_name}"
        return f"{ntype}: {node.get('name', node.get('title', 'unknown'))}"

    def _refine_academic(self, question, raw_answer, evidence, is_category3=False):
        """Generate from the supplied source view without post-hoc factual rewriting."""
        from cogmem.prompt import ANSWER_REFINEMENT_PROMPT
        response = self.llm.chat(
            system=("Answer only from the supplied original source passages. "
                    "Preserve speaker attribution and the specificity of explicit statements. "
                    "A session date records when someone spoke, not when an event occurred. "
                    "Resolve relative dates only against the date of their own source passage; "
                    "preserve the stated year/month/day precision. "
                    "If the sources do not establish the answer, say so. "
                    "Return the concise answer without a reasoning preamble."),
            user=ANSWER_REFINEMENT_PROMPT.format(question=question, evidence=evidence))
        return extract_clean_answer(response, question)


    def _build_run_result(self, answer: str, reasoning_path: list, confidence: str, **extra: Any) -> Dict[str, Any]:
        """Build run() return dict with optional skill_flags and skill_usage for eval."""
        out = {
            "answer": answer,
            "reasoning_path": reasoning_path,
            "confidence": confidence,
            "question_type": self._current_question_type,
            "effective_agent_config": self._effective_agent_config(),
            "operator_calls_used": getattr(self, "_tool_calls_used", 0),
            "planning_rounds_used": getattr(self, "_planning_rounds_used", 0),
            **extra,
        }
        if self.skill_runtime is not None:
            out["skill_flags"] = self.skill_runtime.skill_flags()
            out["skill_usage"] = [e.model_dump() if hasattr(e, "model_dump") else e for e in self.skill_runtime.get_trace()]
        return out

    def _history_dump(self, history: List[ThoughtActionObservation]) -> List[Dict[str, Any]]:
        return [asdict(step) for step in history]

    def _return_run_result(self, history: List[ThoughtActionObservation], answer: str, confidence: str) -> Dict[str, Any]:
        # One guard for every exit path, including early answers and fallbacks.
        if self.require_evidence_for_answer and not self._has_raw_evidence(history):
            answer, confidence = "I don't have enough information.", "low"
        self.history = history
        return self._build_run_result(answer, self._history_dump(history), confidence)

    def _needs_evidence_for_target(self, target, history):
        """An earlier source does not ground a newly selected retrieval address."""
        raw_steps = [step for step in history if self._has_raw_evidence([step])]
        if any((step.action_input or {}).get('entity_id') == target for step in raw_steps):
            return False
        # Different graph addresses can already resolve to the exact same source.
        store = getattr(self, 'evidence_store', None)
        possible = store.get_chunks(target, limit=None) if store else []
        loaded = {item['content'].strip() for step in raw_steps
                  for item in self._structured_observation_payload(step)['items']
                  if isinstance(item, dict) and item.get('content')}
        if possible and all(text.strip() in loaded for text in possible):
            return False
        return True

    def _has_raw_evidence(self, history: List[ThoughtActionObservation]) -> bool:
        for step in history:
            if step.action != "get_evidence" or not (step.observation or "").startswith("Original Text Evidence for "):
                continue
            payload = self._structured_observation_payload(step)
            if isinstance(payload, dict) and isinstance(payload.get("items"), list):
                if any(isinstance(item, dict) and str(item.get("content", "")).strip()
                       for item in payload["items"]):
                    return True
        return False

    def _record_step(
        self,
        history: List[ThoughtActionObservation],
        scratchpad: List[str],
        step_counter: int,
        *,
        thought: str,
        action: str,
        action_input: Dict[str, Any],
        observation: str,
    ) -> int:
        next_step = step_counter + 1
        history.append(
            ThoughtActionObservation(
                step=next_step,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
            )
        )
        action_display = action if action == "find_anchors" else f"{action}({json.dumps(action_input, ensure_ascii=False)})"
        scratchpad.append(
            f"Step {next_step}:\nThought: {thought}\nAction: {action_display}\nObservation: {observation}\n---"
        )
        return next_step

    def _record_and_maybe_finalize(
        self,
        *,
        query: str,
        history: List[ThoughtActionObservation],
        scratchpad: List[str],
        step_counter: int,
        thought: str,
        action: str,
        action_input: Dict[str, Any],
        observation: str,
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        step_counter = self._record_step(
            history,
            scratchpad,
            step_counter,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
        )
        if (self.require_evidence_for_answer and action in {
                "semantic_search", "fact_lookup", "claim_lookup", "timeline_query", "find_common_ground", "traverse_neighbors"}):
            targets = list(self._last_tool_node_hits.get(action, []))
            if not targets:
                target = self._select_evidence_target([], history)
                targets = [target] if target else []
            targets = list(dict.fromkeys(target for target in targets if self._needs_evidence_for_target(target, history)))
            if targets:
                address = {"entity_id":targets[0]} if len(targets)==1 else {"entity_ids":targets}
                params = {**address, "limit": 3, "query": query}
                raw = self._execute_tool("get_evidence", params)
                step_counter = self._record_step(
                    history, scratchpad, step_counter,
                    thought="Resolve retrieved index cues to original source before answering.",
                    action="get_evidence", action_input=params, observation=raw)
                self._evidence_collected = self._has_raw_evidence(history)
        early_answer = self._maybe_finalize_after_step(query, history, action=action)
        if early_answer is None:
            return step_counter, None
        answer_text, answer_confidence = early_answer
        return step_counter, self._return_run_result(history, answer_text, answer_confidence)

    def _handle_semantic_search_recovery(
        self,
        *,
        query: str,
        action: str,
        params: Dict[str, Any],
        obs: str,
        round_idx: int,
        effective_max_iterations: int,
        anchor_results: List[Any],
        history: List[ThoughtActionObservation],
        scratchpad: List[str],
        step_counter: int,
        retry_state: Dict[str, bool],
        verbose: bool = False,
    ) -> Optional[Tuple[int, Optional[Dict[str, Any]], bool]]:
        if action != "semantic_search" or not obs:
            return None

        original_query = params.get("query", query)
        query_lower = original_query.lower()

        if ("No semantically similar nodes found" in obs or "Graph search found no results" in obs):
            if (
                not retry_state["timeline_fallback_used"]
                and any(kw in query_lower for kw in ["when", "time", "date"])
                and round_idx < effective_max_iterations
                and (self.skill_runtime is None or self.skill_runtime.is_skill_enabled("temporal_grounding_skill"))
            ):
                person_ids = [anchor.node_id for anchor in anchor_results if anchor.node_type == "person"]
                if person_ids:
                    retry_state["timeline_fallback_used"] = True
                    time_value_match = re.search(r"\d{4}(?:-\d{2})?(?:-\d{2})?", original_query)
                    time_value = time_value_match.group(0) if time_value_match else ""
                    obs_timeline = self._execute_tool("timeline_query", {
                        "entity_ids": person_ids[:1],
                        "time_query": time_value,
                    })
                    step_counter, early_result = self._record_and_maybe_finalize(
                        query=query,
                        history=history,
                        scratchpad=scratchpad,
                        step_counter=step_counter,
                        thought=f"Auto-injected: semantic_search found no results. Trying timeline_query with time_query='{time_value}'.",
                        action="timeline_query",
                        action_input={"entity_ids": person_ids[:1], "time_query": time_value},
                        observation=obs_timeline,
                    )
                    return step_counter, early_result, True

            if not retry_state["semantic_retry_used"] and round_idx < effective_max_iterations:
                simplified_query = re.sub(r"^(what|when|where|who|how|why)\s+", "", original_query, flags=re.IGNORECASE)
                simplified_query = re.sub(r"\?$", "", simplified_query).strip()
                if simplified_query != original_query:
                    retry_state["semantic_retry_used"] = True
                    obs_retry = self._execute_tool("semantic_search", {
                        "query": simplified_query,
                        "node_types": params.get("node_types"),
                    })
                    if obs_retry and "No semantically similar nodes found" not in obs_retry:
                        step_counter, early_result = self._record_and_maybe_finalize(
                            query=query,
                            history=history,
                            scratchpad=scratchpad,
                            step_counter=step_counter,
                            thought=f"Auto-injected: Original query failed. Retrying with simplified query: '{simplified_query}'.",
                            action="semantic_search",
                            action_input={"query": simplified_query, "node_types": params.get("node_types")},
                            observation=obs_retry,
                        )
                        return step_counter, early_result, True

        if "Semantic Search Results" not in obs:
            return None

        generic_patterns = ["likes", "has", "is", "enjoys", "prefers", "frequently"]
        obs_lines = obs.split("\n")
        fact_count = sum(1 for line in obs_lines if "Fact:" in line)
        generic_fact_count = sum(
            1 for line in obs_lines
            if "Fact:" in line and any(pattern in line.lower() for pattern in generic_patterns)
        )
        current_node_types = params.get("node_types", [])
        if (
            fact_count > 0
            and generic_fact_count >= fact_count * 0.9
            and (not current_node_types or "event" not in current_node_types or "fact" in current_node_types)
            and not retry_state["event_retry_used"]
            and round_idx < effective_max_iterations
        ):
            if verbose:
                print("[SearchAgent] Detected generic Facts. Retrying with Event nodes only...")
            retry_state["event_retry_used"] = True
            obs_retry = self._execute_tool("semantic_search", {
                "query": original_query,
                "node_types": ["event"],
                "top_k": params.get("top_k", self.semantic_search_top_k),
            })
            if obs_retry and "No semantically similar nodes found" not in obs_retry:
                step_counter, early_result = self._record_and_maybe_finalize(
                    query=query,
                    history=history,
                    scratchpad=scratchpad,
                    step_counter=step_counter,
                    thought="Auto-injected: Generic Facts detected. Retrying with Event nodes to preserve details.",
                    action="semantic_search",
                    action_input={"query": original_query, "node_types": ["event"]},
                    observation=obs_retry,
                )
                return step_counter, early_result, True

        return None

    def _execute_tool(self, name: str, params: Dict[str, Any]) -> str:
        if name not in self.tools:
            return f"Error: Tool {name} not found."
        if name in {"semantic_search","fact_lookup","claim_lookup","timeline_query","find_common_ground","traverse_neighbors"} and hasattr(self,'_last_tool_node_hits'):
            self._remember_tool_hits(name, [])  # A failed/empty retrieval cannot reuse old hits.
        used = getattr(self, "_tool_calls_used", 0)
        if used >= self._operator_budget():
            return "Tool budget exhausted."
        self._tool_calls_used = used + 1
        if self.skill_runtime is not None:
            ctx = self._build_skill_context()
            skill_name = self._tool_skill_name(name, params)
            if skill_name is not None:
                skill_result = self.skill_runtime.execute_skill(skill_name, ctx)
                if not skill_result.triggered:
                    return f"Skill Disabled: {skill_name}"
                skill_output = getattr(skill_result, "output", {}) or {}
                if name == "get_evidence" and skill_output.get("query_conditioned") and not params.get("query"):
                    params = {**params, "query": self._current_query}
                if name == "timeline_query" and skill_output.get("time_query") and not params.get("time_query"):
                    params = {**params, "time_query": skill_output.get("time_query")}
            result = self.skill_runtime.execute_tool(name, params, context=ctx, skill_name=skill_name)
            return result.summary
        try:
            res = self.tools[name](**params)
            return str(res)
        except Exception as e:
            raise_if_execution_failure(e)
            return f"Tool Error: {str(e)}"

    def _validate_and_fix_tool_name(self, name: str) -> str:
        mapping = {
            "semantic": "semantic_search",
            "search": "semantic_search",
            "anchor": "find_anchors", 
            "timeline_query": "timeline_query",
            "timeline": "timeline_query",
            "time": "timeline_query",
            "fact": "fact_lookup",
            "claim": "claim_lookup",
            "opinion": "claim_lookup",
            "traverse": "traverse_neighbors", 
            "common": "find_common_ground", 
        }
        name_lower = name.lower()
        for k, v in mapping.items():
            if k in name_lower: return v
        return name

    def _resolve_entity_id(self, query_or_id: str, types: Optional[List[str]] = None) -> Optional[str]:
        """Resolve a user-facing entity mention or node ID to a graph node ID."""
        if not query_or_id:
            return None
        if query_or_id in getattr(self.graph_service.graph_store, "_g", {}):
            node = self.graph_service.graph_store.get_node(query_or_id)
            return query_or_id if node and not node.get('is_archived') and (not types or node.get('node_type') in types) else None
        anchors = self.graph_service.find_anchors(query_or_id, types=types, top_k=1)
        if anchors:
            return anchors[0].node_id
        return None

    def _should_attempt_stepwise_answer(self, action: str, history: List[ThoughtActionObservation]) -> bool:
        if not history:
            return False
        if self.require_evidence_for_answer and not self._has_raw_evidence(history):
            return False
        if action not in {"semantic_search", "claim_lookup", "fact_lookup", "timeline_query", "find_common_ground", "traverse_neighbors", "get_evidence"}:
            return False
        if action == "get_evidence" and not self._has_substantive_retrieval(history[:-1]):
            return False
        latest_obs = (history[-1].observation or "").strip()
        if not latest_obs:
            return False
        if latest_obs.startswith("No ") or "Skill Disabled:" in latest_obs:
            return False
        return self._can_attempt_answer(history, action=action)

    def _shared_evidence_status(self, history: List[ThoughtActionObservation]) -> Tuple[bool, bool]:
        attempted = False
        confirmed = False
        for step in history:
            if step.action != "find_common_ground":
                continue
            attempted = True
            observation = (step.observation or "").strip()
            if observation and "Common Nodes Found" in observation and "No common ground found" not in observation:
                confirmed = True
        return attempted, confirmed

    def _can_attempt_answer(
        self,
        history: List[ThoughtActionObservation],
        action: Optional[str] = None,
    ) -> bool:
        if self._current_question_type != "multi_hop":
            return True
        if action == "find_common_ground":
            return True
        _, has_shared = self._shared_evidence_status(history)
        return has_shared

    def _maybe_finalize_after_step(
        self,
        query: str,
        history: List[ThoughtActionObservation],
        action: str,
    ) -> Optional[tuple[str, str]]:
        if not self._should_attempt_stepwise_answer(action, history):
            return None
        result = self._stepwise_answer_check(query, history)
        if result is None:
            return None
        answer, confidence = result
        answer, confidence = self._postprocess_final_answer(query, answer, confidence, history)
        if not answer or self._is_low_signal_answer(answer):
            return None
        return answer, confidence

    def _stepwise_answer_check(
        self,
        question: str,
        history: List[ThoughtActionObservation],
    ) -> Optional[tuple[str, str]]:
        if self.require_evidence_for_answer and not self._has_raw_evidence(history):
            return None
        if not self._has_substantive_retrieval(history):
            return None
        evidence = self._build_synthesis_evidence(history)
        if not evidence:
            return None
        try:
            user_prompt = SEARCH_AGENT_CAN_ANSWER_PROMPT.format(
                query=question,
                evidence=evidence,
            )
            resp = self.llm.chat(
                system="You are a strict evidence sufficiency judge. Return JSON only.",
                user=user_prompt,
            )
            data = extract_json_from_markdown(resp)
            if not data:
                try:
                    data = json.loads(resp)
                except Exception as execution_error:
                    raise_if_execution_failure(execution_error)
                    data = {}
            if not data and isinstance(resp, str):
                stripped = resp.strip()
                if stripped.upper().startswith("YES:"):
                    data = {"can_answer": True, "answer": stripped.split(":", 1)[1].strip(), "confidence": "medium"}
                elif stripped.upper().startswith("NO:"):
                    data = {"can_answer": False, "missing": stripped.split(":", 1)[1].strip()}
        except Exception as execution_error:
            raise_if_execution_failure(execution_error)
            return None
        if not isinstance(data, dict):
            return None
        if data.get("can_answer") is not True:
            return None
        answer = extract_clean_answer(str(data.get("answer", "")), question)
        confidence = str(data.get("confidence", "medium"))
        if not answer:
            return None
        return answer, confidence

    def _maybe_inject_required_shared_evidence(
        self,
        *,
        query: str,
        history: List[ThoughtActionObservation],
        scratchpad: List[str],
        step_counter: int,
        anchor_results: List[Any],
    ) -> Optional[Tuple[int, Optional[Dict[str, Any]]]]:
        if self._current_question_type != "multi_hop":
            return None
        attempted, has_shared = self._shared_evidence_status(history)
        if has_shared:
            return None
        if attempted:
            return step_counter, None

        entity_ids: List[str] = []
        for anchor in anchor_results:
            if getattr(anchor, "node_type", None) == "person" and anchor.node_id not in entity_ids:
                entity_ids.append(anchor.node_id)
        if len(entity_ids) < 2:
            robust = self._robust_find_anchors(query).get("entities", [])
            for node_id in robust:
                if node_id.startswith("p_") and node_id not in entity_ids:
                    entity_ids.append(node_id)
                if len(entity_ids) >= 2:
                    break
        if len(entity_ids) < 2:
            return step_counter, None

        obs = self._execute_tool("find_common_ground", {
            "entity_ids": entity_ids[:2],
            "node_types": ["concept", "event", "fact"],
        })
        return self._record_and_maybe_finalize(
            query=query,
            history=history,
            scratchpad=scratchpad,
            step_counter=step_counter,
            thought="Auto-injected: multi-hop question requires shared evidence before answering.",
            action="find_common_ground",
            action_input={"entity_ids": entity_ids[:2], "node_types": ["concept", "event", "fact"]},
            observation=obs,
        )

    def _normalize_tool_time_query(self, time_query: str) -> str:
        """Normalize a complete calendar expression without dropping qualifiers."""
        from ..utils.temporal import normalize_event_time
        if not time_query:
            return ""
        text = time_query.strip()
        if text.lower() == 'autumn':
            return 'fall'
        return normalize_event_time(text) or text

    def _tool_find_anchors(self, query: str, types: List[str] = None, top_k: Optional[int] = None, include_evidence: bool = False) -> str:
        # 使用配置的默认值，如果调用时指定了 top_k 则使用指定的值
        if top_k is None:
            top_k = self.find_anchors_top_k
        anchors = self.graph_service.find_anchors(query, types=types, top_k=top_k)
        
        results = []
        for a in anchors:
            node_text = self._format_node_to_rich_text(a.metadata or {}, a.node_id)
            results.append(f"[ID: {a.node_id}] {node_text}")
            
            # ===== 如果启用且EvidenceStore可用，自动附加原文证据 =====
            if include_evidence and self.evidence_store:
                try:
                    evidence_chunks = self.evidence_store.get_chunks(a.node_id, limit=2)
                    if evidence_chunks:
                        evidence_text = " | ".join(
                            [chunk[:150] + "..." if len(chunk) > 150 else chunk for chunk in evidence_chunks]
                        )
                        results.append(f"  [Original Text] {evidence_text}")
                except Exception as execution_error:
                    raise_if_execution_failure(execution_error)
                    pass  # 如果获取证据失败，继续处理其他节点
        
        return "\n".join(results)

    def _tool_semantic_search(self, query: str, node_types: List[str] = None, top_k: Optional[int] = None) -> str:
        """Semantic search: typed recall, filter archived, optional importance rerank."""
        if not self.graph_service.use_vector_search:
            return "Vector search is disabled. Use find_anchors or traverse_neighbors instead."

        if top_k is None:
            top_k = self.semantic_search_top_k

        if node_types is None:
            query_lower = query.lower()
            specific_keywords = ['favorite', 'first', 'best', 'worst', 'won', 'achieved', 'lost',
                                'visited', 'went', 'got', 'became', 'started', 'finished',
                                'when', 'where', 'what happened', 'memory']
            node_types = ['event', 'fact'] if any(kw in query_lower for kw in specific_keywords) else ['fact', 'event']

        expansion = getattr(self, 'vector_search_expansion_factor', 2.0)
        per_type_k = max(1, int(top_k * expansion))

        try:
            embeddings = self.graph_service.embedding_encoder.encode([query])
            if not embeddings:
                return "Embedding failed; no query vector generated."
            query_embedding = embeddings[0]
            vstore = self.graph_service.vector_store
            # Typed recall: search per node_type and merge (prefer search_by_type when available)
            raw_results = []
            if hasattr(vstore, 'search_by_type'):
                for nt in node_types:
                    raw_results.extend(vstore.search_by_type(query_embedding, nt, top_k=per_type_k))
                raw_results.sort(key=lambda x: x[1], reverse=True)
                # Filter and apply the existing importance score before final top-k.
            else:
                raw_results = vstore.search(query_embedding, top_k=top_k * 3)
        except Exception as e:
            raise_if_execution_failure(e)
            return f"Semantic search failed: {e}"

        results_by_type = {nt: [] for nt in node_types}
        for node_id, score in raw_results:
            node_data = self.graph_service.graph_store.get_node(node_id)
            if not node_data:
                continue
            if node_data.get("is_archived"):
                continue
            node_type = node_data.get("node_type")
            if node_type not in results_by_type:
                continue
            importance = node_data.get("importance_score")
            if importance is not None and isinstance(importance, (int, float)):
                score = score * (0.7 + 0.3 * min(1.0, max(0.0, float(importance))))
            node_text = self._format_node_to_rich_text(node_data, node_id)
            results_by_type[node_type].append((score, node_text, node_id))

        ranked = sorted(
            (item for group in results_by_type.values() for item in group),
            key=lambda item: (-item[0], item[2]))[:top_k]
        results = [f"[Score: {score:.3f}] [ID: {node_id}] {text}" for score, text, node_id in ranked]
        hit_ids = [node_id for _, _, node_id in ranked]

        if not results:
            return "No semantically similar nodes found."
        self._remember_tool_hits("semantic_search", hit_ids)

        return f"Semantic Search Results ({len(results)} nodes):\n" + "\n".join(results)

    def _tool_traverse_neighbors(self, node_ids: List[str] = None, node_id: str = None, relation_types: List[str] = None, direction: str = "out", max_depth: int = 1, time_query: str = "") -> str:
        """遍历邻居节点，支持时间查询。"""
        ids = node_ids or ([node_id] if node_id else [])
        if not ids: return "No node IDs provided."
        
        # 如果提供了 time_query，执行时间线查询
        if time_query:
            observation = self._tool_scan_timeline(ids, time_query)
            self._remember_tool_hits("traverse_neighbors", self._last_tool_node_hits.get("timeline_query", []))
            return observation
        
        # 普通遍历：使用配置的默认 max_depth
        if max_depth is None or max_depth == 1:
            max_depth = self.traverse_neighbors_max_depth
        
        res = self.graph_service.traverse_associations(ids, relation_types=relation_types, direction=direction, max_depth=max_depth)
        focus_terms = self._extract_focus_terms(getattr(self, "_current_query", ""))
        if focus_terms:
            res = sorted(res, key=lambda result: (
                -self._score_query_node(result.target_node or {}, focus_terms),
                result.target_id, result.source_id, result.relation_type,
            ))
        selected = []
        seen = set()
        for result in res:
            if result.target_id in seen:
                continue
            seen.add(result.target_id)
            selected.append(result)
            if len(selected) >= self.traverse_neighbors_max_results:
                break
        self._remember_tool_hits("traverse_neighbors", [r.target_id for r in selected])
        return "\n".join([f"[{r.relation_type}] -> [ID: {r.target_id}] {self._format_node_to_rich_text(r.target_node or {}, r.target_id)}" for r in selected])
    
    def _tool_scan_timeline(self, entity_ids: List[str], time_query: str = "") -> str:
        """扫描时间线（内部方法，由 traverse_neighbors 调用）"""
        self._remember_tool_hits("timeline_query", [])
        season_months = {
            "spring": {3, 4, 5},
            "summer": {6, 7, 8},
            "fall": {9, 10, 11},
            "winter": {12, 1, 2},
        }

        def _extract_month(value: str) -> Optional[int]:
            if not value:
                return None
            match = re.search(r"(\d{4})-(\d{2})", str(value))
            if match:
                try:
                    month = int(match.group(2))
                    if 1 <= month <= 12:
                        return month
                except ValueError:
                    return None
            return None

        def _matches_season(item: Dict[str, Any], season: str) -> bool:
            season = "fall" if season == "autumn" else season
            months = season_months.get(season, set())
            haystack = " ".join([
                str(item.get("title", "")),
                str(item.get("description", "")),
                str(item.get("content", "")),
                str(item.get("time_start_raw", "")),
            ]).lower()
            if season in haystack:
                return True
            for field_name in ("time_start", "time_resolved"):
                month = _extract_month(str(item.get(field_name, "")))
                if month in months:
                    return True
            return False

        # 1. 获取所有关联事件
        person_results = self.graph_service.traverse_associations(
            entity_ids, 
            relation_types=["PARTICIPATED_IN", "MENTIONED_IN"], 
            direction="out"
        )
        
        events = []
        seen_ids = set()
        focus_terms = self._extract_focus_terms(self._current_query or "")
        for r in person_results:
            if r.target_node and r.target_node.get("node_type") == "event" and not r.target_node.get('is_archived'):
                event_id = r.target_id
                if event_id not in seen_ids:
                    event_payload = dict(r.target_node)
                    event_payload["id"] = event_id
                    events.append(event_payload)
                    seen_ids.add(event_id)
        
        # 2. 如果直接提供了 Event ID，也加入
        for eid in entity_ids:
            if eid not in seen_ids:
                node = self.graph_service.graph_store.get_node(eid)
                if node and node.get('node_type') == 'event' and not node.get('is_archived'):
                    event_payload = dict(node)
                    event_payload["id"] = eid
                    events.append(event_payload)
                    seen_ids.add(eid)
        
        # 3. 时间过滤（P0 FIX: 区分语义查询和时间值查询）
        time_query_lower = time_query.lower().strip()
        if time_query_lower and time_query_lower != "*":
            filtered_events = []
            is_season_query = time_query_lower in season_months
            
            # 判断time_query是语义查询还是时间值
            # 时间值格式：YYYY, YYYY-MM, YYYY-MM-DD
            is_time_value = bool(re.match(r'^\d{4}(-\d{2})?(-\d{2})?$', time_query_lower))
            
            for e in events:
                match = False
                if is_season_query:
                    match = _matches_season(e, time_query_lower)
                elif is_time_value:
                    # Recording time is never a substitute for occurrence time.
                    bounds = date_bounds(e.get('time_start')) or date_bounds(e.get('time_resolved'))
                    requested = date_bounds(time_query_lower)
                    match = bool(bounds and requested and bounds[0] <= requested[1] and requested[0] <= bounds[1])
                else:
                    # 语义查询：在事件标题和描述中搜索
                    title = str(e.get("title", "")).lower()
                    desc = str(e.get("description", "")).lower()
                    content = str(e.get("content", "")).lower()
                    # 分词匹配：如果查询中的任何token在标题或描述中，则匹配
                    query_tokens = time_query_lower.split()
                    match = any(token in title or token in desc or token in content 
                               for token in query_tokens if len(token) > 2)  # 忽略太短的token
                
                if match:
                    filtered_events.append(e)
            events = filtered_events

        def _timeline_score(item: Dict[str, Any]) -> float:
            haystack = " ".join([
                str(item.get("title", "")),
                str(item.get("description", "")),
                str(item.get("content", "")),
                str(item.get("time_start_raw", "")),
            ]).lower()
            score = 0.0
            for term in focus_terms:
                if term in haystack:
                    score += 2.0
            if time_query_lower in season_months and _matches_season(item, time_query_lower):
                score += 3.0
            elif time_query_lower and time_query_lower in haystack:
                score += 2.5
            if item.get("time_start"):
                score += 1.0
            return score

        if events:
            # Keep query-conditioned ranking as the primary order signal.
            events.sort(
                key=lambda item: (
                    _timeline_score(item),
                    str(item.get("time_start") or item.get("time_resolved") or ""),
                ),
                reverse=True,
            )
        
        if not events:
            # Fallback: scan fact nodes with temporal metadata when event nodes are sparse.
            fact_matches = []
            graph = self.graph_service.graph_store._g
            for node_id, raw_node in graph.nodes(data=True):
                node_data = self.graph_service.graph_store.get_node(node_id) or raw_node
                if not node_data or node_data.get("node_type") != "fact":
                    continue
                if node_data.get("subject_id") not in entity_ids or node_data.get('is_archived'):
                    continue
                content = str(node_data.get("content", "")).lower()
                if focus_terms and not any(term in content for term in focus_terms):
                    continue
                if time_query_lower and time_query_lower != "*":
                    requested = date_bounds(time_query_lower)
                    if requested:
                        start = date_bounds(node_data.get('valid_from'))
                        end = date_bounds(node_data.get('valid_to'))
                        start, end = start or end, end or start
                        if not (start and end and start[0] <= requested[1] and requested[0] <= end[1]):
                            continue
                    elif time_query_lower not in content:
                        continue
                fact_matches.append({**node_data, "id": node_data.get("id") or node_id})
            fact_matches.sort(
                key=lambda item: (
                    sum(2.0 for term in focus_terms if term in str(item.get("content", "")).lower()),
                    str(item.get("id", "")),
                ),
                reverse=True,
            )
            if fact_matches:
                hit_ids = [item["id"] for item in fact_matches[:self.traverse_neighbors_max_results]]
                self._remember_tool_hits("timeline_query", hit_ids)
                lines = [
                    f"{idx}. [ID: {item['id']}] Fact: {item.get('content', '')} | Session: {item.get('session_time', '')}"
                    for idx, item in enumerate(fact_matches[:self.traverse_neighbors_max_results], start=1)
                ]
                return f"Timeline Fact Fallback ({len(lines)} items):\n" + "\n".join(lines)
            return f"No events found for {entity_ids} matching query '{time_query}'"

        hit_ids = []
        formatted_lines = []
        for idx, event in enumerate(events[:self.traverse_neighbors_max_results], start=1):
            event_id = event.get("id") or event.get("node_id")
            if event_id and event_id not in hit_ids:
                hit_ids.append(event_id)
            formatted_lines.append(f"{idx}. [ID: {event_id}] {self._format_node_to_rich_text(event, event_id)}")
        self._remember_tool_hits("timeline_query", hit_ids)
        return f"Timeline ({len(formatted_lines)} events):\n" + "\n".join(formatted_lines)

    def _tool_timeline_query(self, entity_ids: List[str] = None, entity_id: str = "", entity_queries: List[str] = None, relation_types: List[str] = None, time_query: str = "") -> str:
        """Explicit timeline query tool for temporal recall."""
        resolved_ids = list(entity_ids or [])
        if entity_id:
            resolved_ids.append(entity_id)
        for query in entity_queries or []:
            resolved = self._resolve_entity_id(query, types=["person", "event"])
            if resolved:
                resolved_ids.append(resolved)
        deduped_ids = []
        for item in resolved_ids:
            if item and item not in deduped_ids:
                deduped_ids.append(item)
        if not deduped_ids:
            return "No entity IDs provided for timeline_query."
        normalized_time_query = self._normalize_tool_time_query(time_query)
        return self._tool_scan_timeline(deduped_ids, normalized_time_query)

    def _tool_fact_lookup(self, subject_id: str = "", subject: str = "", predicate: str = "", object_id: str = "", at_time: str = "", latest_only: bool = False, limit: int = 8, query: str = "") -> str:
        """Direct fact lookup for a subject, optional predicate/object/time filtering."""
        resolved_subject_id = subject_id or self._resolve_entity_id(subject, types=["person"])
        if not resolved_subject_id:
            return "No subject found for fact_lookup."
        normalized_time = self._normalize_tool_time_query(at_time)
        effective_query = (query or self._current_query or "").strip()
        focus_terms = self._extract_focus_terms(effective_query)
        results = []
        graph = self.graph_service.graph_store._g
        if hasattr(graph, "nodes"):
            for node_id, raw_node in graph.nodes(data=True):
                node_data = self.graph_service.graph_store.get_node(node_id) or raw_node
                if not node_data or node_data.get("node_type") != "fact":
                    continue
                if node_data.get("subject_id") != resolved_subject_id:
                    continue
                if node_data.get("is_archived"):
                    continue
                if predicate and predicate.lower() not in str(node_data.get("predicate", "")).lower():
                    continue
                if object_id and object_id not in {node_data.get("object_id"), node_data.get("purpose_id")}:
                    continue
                if latest_only and node_data.get("is_latest") is False:
                    continue
                if normalized_time:
                    start = date_bounds(node_data.get('valid_from'))
                    end = date_bounds(node_data.get('valid_to'))
                    requested = date_bounds(normalized_time)
                    # Unknown validity is not the date the statement was recorded.
                    # A single explicit bound certifies only its stated precision.
                    start, end = start or end, end or start
                    if not (start and end and requested and start[0] <= requested[1] and requested[0] <= end[1]):
                        continue
                item = {**node_data, "id": node_data.get("id") or node_id}
                haystack = " ".join([
                    str(item.get("content", "")),
                    str(item.get("predicate", "")),
                    str(item.get("object_id", "")),
                    str(item.get("purpose_id", "")),
                ]).lower()
                query_score = 0.0
                for term in focus_terms:
                    if term in haystack:
                        query_score += 2.0
                if predicate and predicate.lower() in str(item.get("predicate", "")).lower():
                    query_score += 3.0
                if item.get("is_latest") is not False:
                    query_score += 0.25
                item["_query_score"] = query_score
                results.append(item)
        if not results:
            return f"No facts found for {resolved_subject_id}."
        results.sort(
            key=lambda item: (
                item.get("_query_score", 0.0),
                item.get("is_latest") is not False,
                item.get("valid_to") or item.get("valid_from") or "",
                item.get("confidence", 0.0),
            ),
            reverse=True,
        )
        lines = []
        hit_ids = []
        for idx, item in enumerate(results[:limit], start=1):
            validity = ""
            if item.get("valid_from") or item.get("valid_to"):
                validity = f" | Valid: {item.get('valid_from') or '?'} -> {item.get('valid_to') or '?'}"
            elif item.get("session_time"):
                validity = f" | Session: {item.get('session_time')}"
            lines.append(f"{idx}. [ID: {item.get('id')}] {self._format_node_to_rich_text(item, item.get('id'))}{validity}")
            hit_ids.append(item.get("id"))
        self._remember_tool_hits("fact_lookup", hit_ids)
        return f"Fact Lookup Results ({len(results[:limit])} facts):\n" + "\n".join(lines)

    def _tool_claim_lookup(self, target_id: str = "", target: str = "", source_id: str = "", source: str = "", sentiment: str = "", limit: int = 8) -> str:
        """Direct claim lookup for opinion/judgment questions."""
        resolved_target_id = target_id or self._resolve_entity_id(target, types=["person", "concept"])
        resolved_source_id = source_id or self._resolve_entity_id(source, types=["person"])
        if source and not resolved_source_id:
            return "No source found for the requested claim attribution."
        if target and not resolved_target_id:
            return "No target found for the requested claim attribution."
        if not resolved_target_id and not resolved_source_id:
            return "No source or target found for claim_lookup."
        claims = self.graph_service.find_claims_about_target(
            source_id=resolved_source_id,
            target_id=resolved_target_id,
            sentiment=sentiment or None,
        )
        if not claims:
            return "No claims found."
        lines = []
        hit_ids = []
        for idx, claim in enumerate(claims[:limit], start=1):
            source_name = claim.get("source_id", "").replace("p_", "").replace("_", " ")
            target_name = claim.get("target_id", "").replace("p_", "").replace("c_", "").replace("_", " ")
            session_time = claim.get("session_time")
            meta = f" | Sentiment: {claim.get('sentiment', 'unknown')}"
            if session_time:
                meta += f" | Session: {session_time}"
            claim_id = claim.get("id", "")
            lines.append(f"{idx}. [ID: {claim_id}] Claim: {source_name} about {target_name}: {claim.get('content', '')}{meta}")
            if claim_id:
                hit_ids.append(claim_id)
        self._remember_tool_hits("claim_lookup", hit_ids)
        return f"Claim Lookup Results ({len(claims[:limit])} claims):\n" + "\n".join(lines)

    def _tool_find_common_ground(self, entity_ids: List[str], node_types: List[str] = ["concept", "event"]) -> str:
        """Find common ground between entities.
        
        P2 FIX: Enhanced to support Fact semantic matching for multi-hop questions.
        """
        cg = self.graph_service.find_common_ground(entity_ids, node_types=node_types)
        # Ranking must precede the budget. The service returns a set intersection,
        # whose iteration order is not a relevance order.
        common_nodes = list(cg.common_nodes) if cg.common_nodes else []
        query = self._current_query or ""
        focus_terms = self._extract_focus_terms(query)
        
        # P2 FIX: If "fact" is in node_types or if no common nodes found, try semantic Fact matching
        if ("fact" in node_types or not common_nodes) and len(entity_ids) >= 2:
            try:
                # Get Facts for each entity
                facts_by_entity = {}
                for entity_id in entity_ids:
                    facts = []
                    if hasattr(self.graph_service.graph_store, '_g'):
                        for node_id, raw_node in self.graph_service.graph_store._g.nodes(data=True):
                            node_data = self.graph_service.graph_store.get_node(node_id) or raw_node
                            if node_data and node_data.get('node_type') == 'fact' and node_data.get('subject_id') == entity_id:
                                facts.append({**node_data, 'id': node_data.get('id') or node_id})
                    facts_by_entity[entity_id] = facts
                
                # Find semantically similar Facts across entities
                if len(facts_by_entity) >= 2:
                    entity_list = list(facts_by_entity.keys())
                    similar_fact_pairs = []
                    
                    # Simple pairwise comparison (for 2 entities)
                    if len(entity_list) == 2:
                        e1_facts = facts_by_entity[entity_list[0]]
                        e2_facts = facts_by_entity[entity_list[1]]
                        
                        for f1 in e1_facts[:20]:  # Limit to avoid too many comparisons
                            for f2 in e2_facts[:20]:
                                # Compute similarity based on content and predicate
                                f1_content = f1.get('content', '').lower()
                                f2_content = f2.get('content', '').lower()
                                f1_pred = f1.get('predicate', '').lower()
                                f2_pred = f2.get('predicate', '').lower()
                                
                                # Simple similarity: check if predicates are similar and content has common keywords
                                pred_similar = (f1_pred == f2_pred or 
                                              any(p in f1_pred and p in f2_pred for p in ['lost', 'start', 'open', 'won', 'achieved']))
                                
                                if pred_similar:
                                    # Extract keywords from content
                                    f1_keywords = set(w for w in f1_content.split() if len(w) > 4)
                                    f2_keywords = set(w for w in f2_content.split() if len(w) > 4)
                                    common_keywords = f1_keywords & f2_keywords
                                    
                                    # If predicates match and have common keywords, consider it a match
                                    if len(common_keywords) >= 2 or f1_pred == f2_pred:
                                        similarity = len(common_keywords) / max(len(f1_keywords), len(f2_keywords), 1)
                                        if similarity > 0.3 or f1_pred == f2_pred:
                                            similar_fact_pairs.append({
                                                'type': 'common_fact_pattern',
                                                'fact1': f1,
                                                'fact2': f2,
                                                'similarity': similarity,
                                                'common_pattern': f1_pred or 'attribute'
                                            })
                    
                    # Add similar fact pairs to common_nodes
                    if similar_fact_pairs:
                        # Sort by similarity and take top matches
                        similar_fact_pairs.sort(key=lambda x: x['similarity'], reverse=True)
                        for pair in similar_fact_pairs[:3]:  # Top 3 matches
                            # Format as a common pattern description
                            pattern_desc = f"Common Pattern: {pair['common_pattern']} (Similarity: {pair['similarity']:.2f})"
                            pattern_node = {
                                'node_type': 'pattern',
                                'content': pattern_desc,
                                'fact1_content': pair['fact1'].get('content', ''),
                                'fact2_content': pair['fact2'].get('content', ''),
                                'supporting_node_ids': [pair['fact1']['id'], pair['fact2']['id']],
                            }
                            common_nodes.append(pattern_node)

                    shared_terms = self._extract_shared_fact_terms([facts_by_entity[eid] for eid in entity_list], focus_terms)
                    for term, evidence_rows in shared_terms[:3]:
                        pattern_node = {
                            'node_type': 'pattern',
                            'content': f"Shared fact theme: {term}",
                            'fact1_content': evidence_rows[0].get('content', '') if evidence_rows else '',
                            'fact2_content': evidence_rows[1].get('content', '') if len(evidence_rows) > 1 else '',
                            'supporting_node_ids': [row['id'] for row in evidence_rows if row.get('id')],
                        }
                        common_nodes.append(pattern_node)
            except Exception as e:
                raise_if_execution_failure(e)
                # Silently fail if semantic matching fails
                pass
        
        if not common_nodes:
            return "No common ground found."

        common_nodes.sort(key=lambda node: (
            -self._score_common_ground_node(node, focus_terms) if focus_terms else 0,
            str(node.get('id') or '|'.join(sorted(node.get('supporting_node_ids', [])))),
            json.dumps(node, sort_keys=True, default=str),
        ))
        common_nodes = common_nodes[:self.find_common_ground_top_k]
        
        result_lines = []
        hit_ids = []
        for node in common_nodes:
            addresses = node.get('supporting_node_ids', []) if node.get('node_type') == 'pattern' else [node.get('id')]
            hit_ids.extend(address for address in addresses
                           if address and self.graph_service.graph_store.get_node(address))
            if node.get('node_type') == 'pattern':
                # Format pattern nodes specially
                result_lines.append(f"- {node['content']}")
                result_lines.append(f"  * Entity 1: {node['fact1_content']}")
                result_lines.append(f"  * Entity 2: {node['fact2_content']}")
            else:
                result_lines.append(f"- [ID: {node.get('id', '')}] {self._format_node_to_rich_text(node)}")
        self._remember_tool_hits("find_common_ground", hit_ids)
        
        return "Common Nodes Found:\n" + "\n".join(result_lines)

    def _normalize_match_token(self, token: str) -> str:
        token = re.sub(r'[^a-z0-9]+', '', token.lower())
        if len(token) > 5 and token.endswith('ing'):
            token = token[:-3]
        elif len(token) > 4 and token.endswith('ed'):
            token = token[:-2]
        elif len(token) > 4 and token.endswith('es'):
            token = token[:-2]
        elif len(token) > 3 and token.endswith('s'):
            token = token[:-1]
        return token

    def _extract_focus_terms(self, query: str) -> set[str]:
        stopwords = {
            "what", "when", "where", "which", "who", "whom", "does", "did", "have", "has",
            "both", "their", "they", "them", "with", "about", "into", "from", "that", "this",
            "were", "been", "being", "would", "could", "should"
        }
        terms = set()
        for token in re.findall(r"[A-Za-z0-9_]+", query.lower()):
            normalized = self._normalize_match_token(token)
            if normalized and len(normalized) >= 3 and normalized not in stopwords:
                terms.add(normalized)
        return terms

    def _score_query_node(self, node: Dict[str, Any], focus_terms: set[str]) -> float:
        text_parts = [
            str(node.get("name", "")),
            str(node.get("title", "")),
            str(node.get("content", "")),
            str(node.get("description", "")),
            str(node.get("fact1_content", "")),
            str(node.get("fact2_content", "")),
        ]
        haystack = " ".join(text_parts).lower()
        score = 0.0
        for term in focus_terms:
            if term in self._normalize_match_token(haystack) or term in haystack:
                score += 3.0
        return score

    def _score_common_ground_node(self, node: Dict[str, Any], focus_terms: set[str]) -> float:
        score = self._score_query_node(node, focus_terms)
        if node.get("node_type") == "pattern":
            score += 4.0
        elif node.get("node_type") == "concept":
            score += 2.0
        return score

    def _extract_shared_fact_terms(self, facts_groups: List[List[Dict[str, Any]]], focus_terms: set[str]) -> List[tuple[str, List[Dict[str, Any]]]]:
        if len(facts_groups) < 2:
            return []
        generic_terms = {
            "like", "create", "have", "use", "make", "find", "fact",
            "person", "subject", "object", "and"
        }
        per_group_terms = []
        examples_by_term: Dict[str, List[Optional[Dict[str, Any]]]] = {}
        for group_idx, facts in enumerate(facts_groups):
            group_terms = set()
            for fact in facts:
                text = " ".join([
                    str(fact.get("content", "")),
                    str(fact.get("predicate", "")),
                    str(fact.get("object_id", "")),
                    str(fact.get("purpose_id", "")),
                ]).lower().replace("_", " ")
                for token in re.findall(r"[A-Za-z0-9]+", text):
                    normalized = self._normalize_match_token(token)
                    if len(normalized) < 3 or normalized in generic_terms:
                        continue
                    group_terms.add(normalized)
                    examples_by_term.setdefault(normalized, [None] * len(facts_groups))
                    if not examples_by_term[normalized][group_idx]:
                        examples_by_term[normalized][group_idx] = fact
            per_group_terms.append(group_terms)
        shared = set.intersection(*per_group_terms) if per_group_terms else set()
        ranked = []
        for term in shared:
            score = 1
            if term in focus_terms:
                score += 4
            examples = [example for example in examples_by_term.get(term, []) if example]
            ranked.append((score, term, examples[:2]))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [(term, examples) for _, term, examples in ranked]

    def _is_low_signal_answer(self, answer: str) -> bool:
        if not answer:
            return True
        # This is the explicit failure value required by the reader contract.
        # It remains a valid final abstention, but cannot certify early success.
        if answer.strip().rstrip('.').casefold() == "i don't have enough information":
            return True
        lowered = answer.lower()
        markers = [
            "thought:",
            "action:",
            "observation:",
            "tool:",
            "args:",
            "not enough evidence",
            "there is no specific",
            "semantic search results",
            "claim lookup results",
            "common nodes found",
            "reasoning limit reached",
            "no common ground found",
            "timeline (",
            "supporting evidence:",
        ]
        return any(marker in lowered for marker in markers)

    def _sanitize_final_answer_candidate(self, answer: str) -> str:
        candidate = (answer or "").strip()
        if not candidate:
            return ""

        candidate = re.sub(r"^FINAL_ANSWER\s*:\s*", "", candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"\n+Supporting evidence:.*$", "", candidate, flags=re.IGNORECASE | re.DOTALL).strip()

        if self._looks_like_tool_protocol(candidate):
            return ""

        if re.match(r"^(Thought|Action|ARGS|Observation)\s*:", candidate, flags=re.IGNORECASE):
            return ""

        if candidate.startswith("Timeline (") or candidate.startswith("Semantic Search Results"):
            return ""

        return candidate

    def _postprocess_final_answer(self, query, answer, confidence, history):
        if self.require_evidence_for_answer and not self._has_raw_evidence(history):
            return "I don't have enough information.", "low"
        synthesized = self._synthesize_answer_with_llm(query, "", history)
        if synthesized:
            return self._sanitize_final_answer_candidate(synthesized), confidence
        # A failed consumer cannot fall back to a controller's graph-based draft.
        return "I don't have enough information.", "low"

    def _build_synthesis_evidence_package(self, history: List[ThoughtActionObservation], max_items: int = 5) -> Dict[str, Any]:
        if self.require_evidence_for_answer:
            history = [step for step in history
                       if step.action == "get_evidence" and self._has_raw_evidence([step])]
        prioritized_actions = [
            "get_evidence",
            "claim_lookup",
            "fact_lookup",
            "timeline_query",
            "find_common_ground",
            "semantic_search",
            "find_anchors",
        ]
        ranked_steps: List[ThoughtActionObservation] = []
        seen = set()
        for action in prioritized_actions:
            for step in history:
                if step.action != action:
                    continue
                obs = (step.observation or "").strip()
                if not obs or obs in seen:
                    continue
                seen.add(obs)
                ranked_steps.append(step)
                if len(ranked_steps) >= max_items:
                    break
            if len(ranked_steps) >= max_items:
                break

        if not ranked_steps:
            ranked_steps = [step for step in history if step.action != "find_anchors"][-max_items:]
        if not ranked_steps:
            ranked_steps = history[-max_items:]

        query_entities = self._extract_query_entities(self._current_query or "")
        evidence_package = {
            "question_type": self._current_question_type,
            "query_entities": query_entities,
            "shared_evidence": [],
            "entity_evidence": {entity: [] for entity in query_entities},
            "general_evidence": [],
            "steps": [],
        }
        for step in ranked_steps[-max_items:]:
            obs = (step.observation or "").strip()
            if not obs:
                continue
            evidence_item = {
                "step": step.step,
                "action": step.action,
                "action_input": step.action_input,
                "evidence": self._structured_observation_payload(step),
            }
            evidence_package["steps"].append(evidence_item)
            matched_entities = self._infer_evidence_entities(evidence_item["evidence"], query_entities)
            if step.action == "find_common_ground" or len(matched_entities) >= 2:
                evidence_package["shared_evidence"].append(evidence_item)
            elif len(matched_entities) == 1:
                evidence_package["entity_evidence"][matched_entities[0]].append(evidence_item)
            else:
                evidence_package["general_evidence"].append(evidence_item)
        return evidence_package

    def _build_synthesis_evidence(self, history: List[ThoughtActionObservation], max_items: int = 5) -> str:
        return json.dumps(
            self._build_synthesis_evidence_package(history, max_items=max_items),
            ensure_ascii=False,
            indent=2,
        )

    def _synthesize_answer_with_llm(
        self,
        question: str,
        draft_answer: str,
        history: List[ThoughtActionObservation],
    ) -> Optional[str]:
        evidence = self._build_synthesis_evidence(history)
        if not evidence:
            return None
        try:
            refined = self._refine_academic(
                question=question,
                raw_answer=draft_answer or "",
                evidence=evidence,
                is_category3=(self._current_question_type == "single_hop"),
            )
            if refined and not self._is_low_signal_answer(refined):
                return refined
        except Exception as execution_error:
            raise_if_execution_failure(execution_error)
            return None
        return None

    def _structured_observation_payload(self, step: ThoughtActionObservation) -> Any:
        action = step.action
        observation = (step.observation or "").strip()
        if not observation:
            return ""

        def _extract_inline_id(text: str) -> tuple[str, str]:
            match = re.search(r"\[ID:\s*([^\]]+)\]\s*(.+)$", text)
            if not match:
                return "", text
            return match.group(1).strip(), match.group(2).strip()

        if action == "claim_lookup":
            claims = []
            pattern = re.compile(
                r"^\d+\.\s+(?:\[ID:\s*(?P<id>[^\]]+)\]\s+)?Claim:\s+(?P<source>.+?)\s+about\s+(?P<target>.+?):\s+(?P<content>.+?)(?:\s+\|\s+Sentiment:\s+(?P<sentiment>\w+))?(?:\s+\|\s+Session:\s+(?P<session>.+))?$",
                re.IGNORECASE,
            )
            for line in observation.splitlines():
                match = pattern.match(line.strip())
                if not match:
                    continue
                claims.append({
                    "id": (match.group("id") or "").strip(),
                    "source": match.group("source").strip(),
                    "target": match.group("target").strip(),
                    "content": match.group("content").strip(),
                    "sentiment": (match.group("sentiment") or "").strip(),
                    "session": (match.group("session") or "").strip(),
                })
            return {"type": "claim_list", "items": claims or observation[:1800]}

        if action == "semantic_search":
            items = []
            pattern = re.compile(
                r"^\[Score:\s*(?P<score>[0-9.]+)\]\s+(?:\[ID:\s*(?P<id>[^\]]+)\]\s+)?(?P<content>.+)$",
                re.IGNORECASE,
            )
            for line in observation.splitlines():
                match = pattern.match(line.strip())
                if not match:
                    continue
                content = match.group("content").strip()
                item_type = content.split(":", 1)[0].strip() if ":" in content else "unknown"
                items.append({
                    "id": (match.group("id") or "").strip(),
                    "score": float(match.group("score")),
                    "item_type": item_type,
                    "content": content,
                })
            return {"type": "ranked_results", "items": items or observation[:1800]}

        if action == "find_common_ground":
            items = []
            for line in observation.splitlines():
                stripped = line.strip()
                if stripped.startswith("- "):
                    items.append(stripped[2:])
            return {"type": "common_ground", "items": items or observation[:1800]}

        if action == "find_anchors":
            items = []
            for line in observation.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("HINT:"):
                    continue
                _, content = _extract_inline_id(stripped)
                prefix = content.split(":", 1)[0] if ":" in content else ""
                if prefix in {"Person", "Concept", "Event", "Fact", "Claim"}:
                    items.append(stripped)
            return {"type": action, "items": items[:8] or observation[:400]}

        if action in {"timeline_query", "fact_lookup"}:
            items = []
            for line in observation.splitlines():
                stripped = line.strip()
                if re.match(r"^\d+\.\s+", stripped):
                    entry = re.sub(r"^\d+\.\s+", "", stripped)
                    node_id, content = _extract_inline_id(entry)
                    items.append({"id": node_id, "content": content})
                elif stripped and not stripped.endswith(":") and "Results" not in stripped and "Timeline (" not in stripped:
                    node_id, content = _extract_inline_id(stripped)
                    items.append({"id": node_id, "content": content})
            return {"type": action, "items": items[:12] or observation[:1800]}

        if action == "get_evidence":
            items = []
            pattern = re.compile(
                r"^\[Evidence\s+\d+\][ \t]*(?P<content>.*?)(?=^\[Evidence\s+\d+\]|\Z)",
                re.IGNORECASE | re.MULTILINE | re.DOTALL,
            )
            for match in pattern.finditer(observation):
                if match:
                    content = match.group("content").strip()
                    session_match = re.search(r"\[Session Date:\s*([^\]]+)\]", content, re.IGNORECASE)
                    session_date = ""
                    if session_match:
                        raw_session = session_match.group(1).strip()
                        session_date = self._normalize_tool_time_query(raw_session.split(" on ")[-1] if " on " in raw_session else raw_session)
                    items.append({
                        "session_date": session_date,
                        "content": content,
                    })
            return {"type": "evidence_chunks", "items": items[:6] or observation[:1800]}

        return {"type": action, "text": observation[:1800]}

    def _extract_query_entities(self, query: str) -> List[str]:
        entities = []
        for token in re.findall(r"\b[A-Z][a-z]+\b", query):
            lower = token.lower()
            if lower in {"what", "when", "where", "which", "who", "how"}:
                continue
            if token not in entities:
                entities.append(token)
        return entities

    def _infer_evidence_entities(self, evidence_payload: Any, query_entities: List[str]) -> List[str]:
        if not query_entities:
            return []
        if isinstance(evidence_payload, str):
            haystack = evidence_payload.lower()
        else:
            haystack = json.dumps(evidence_payload, ensure_ascii=False).lower()
        matched = []
        for entity in query_entities:
            if entity.lower() in haystack and entity not in matched:
                matched.append(entity)
        return matched

    def _tool_get_evidence(self, entity_id: Optional[str] = None, limit: int = 3, query: str = None, entity_ids: Optional[List[str]] = None) -> str:
        """Resolve graph addresses to original sources, then apply one shared read limit."""
        if not self.evidence_store:
            return "EvidenceStore not available. Cannot retrieve original text evidence."
        if entity_ids is not None and (not isinstance(entity_ids,list) or any(not isinstance(node,str) or not node for node in entity_ids)):
            return "Error retrieving evidence: entity_ids must be a list of nonempty node IDs."
        if entity_id is not None and (not isinstance(entity_id,str) or not entity_id):
            return "Error retrieving evidence: entity_id must be a nonempty node ID."
        requested=list(dict.fromkeys(([entity_id] if entity_id else [])+(entity_ids or [])))
        if not requested:
            return "No graph node IDs provided for original source retrieval."
        try:
            limit=max(1,int(limit))
            pending=deque(requested);visited=set();scope=[]
            graph_store=getattr(self.graph_service,'graph_store',None)
            network=getattr(graph_store,'_g',None)
            while pending:
                node=pending.popleft()
                if node in visited:continue
                visited.add(node)
                direct=self.evidence_store.get_chunks(node,limit=None)
                scope.extend(direct)
                chunk_store = getattr(self.graph_service, 'chunk_store', None)
                if chunk_store is not None:
                    original = chunk_store.get_chunk(node)
                    if original is not None:
                        scope.append(original.text)
                # Cached sources and explicit provenance are complementary. A
                # partial direct cache must not conceal other declared supports.
                # Follow only provenance edges; never search unrelated raw history.
                if network is not None and node in network:
                    pending.extend(target for _,target,data in network.out_edges(node,data=True)
                        if (data.get('rel_type') or data.get('relation_type'))=='SUPPORTED_BY' and target not in visited)
            scope=list(dict.fromkeys(scope))
            if query:
                chunks=self.evidence_store.rank_chunks(scope,query,limit=limit,
                    embedding_encoder=getattr(self.graph_service,'embedding_encoder',None),
                    chunk_store=getattr(self.graph_service,'chunk_store',None))
            else:
                chunks=scope[:limit]
        except Exception as error:
            raise_if_execution_failure(error)
            return f"Error retrieving evidence: {str(error)}"
        if not chunks:
            return f"No original text evidence found for graph nodes: {requested}"
        address=requested[0] if len(requested)==1 else json.dumps(requested,ensure_ascii=False)
        evidence_text="\n".join(f"[Evidence {i+1}] {chunk}" for i,chunk in enumerate(chunks))
        return f"Original Text Evidence for {address}:\n{evidence_text}"

    def _robust_find_anchors(self, query: str) -> Dict[str, List[str]]:
        """轻量锚点解析：提供候选 ID 供 LLM 提示"""
        anchors = self.graph_service.find_anchors(query, top_k=self.find_anchors_top_k)
        ids = [a.node_id for a in anchors]
        
        # 尝试小写匹配
        q_lower = query.lower()
        if q_lower != query:
            anchors_lower = self.graph_service.find_anchors(q_lower, top_k=self.find_anchors_top_k)
            for a in anchors_lower:
                if a.node_id not in ids:
                    ids.append(a.node_id)
        
        return {"entities": ids[:8]} if ids else {}
