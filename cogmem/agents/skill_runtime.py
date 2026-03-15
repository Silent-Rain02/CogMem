"""Skill runtime: skill triggers, switches, cache, and trace."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .runtime_schema import (
    RuntimeTraceEntry,
    SkillContext,
    SkillResult,
    ToolResult,
)
from .tool_registry import ToolRegistry

class SkillRuntime:
    """Manages skill execution: routing, temporal, evidence, common_ground."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        *,
        enable_routing_skill: bool = True,
        enable_evidence_skill: bool = True,
        enable_temporal_skill: bool = True,
        enable_common_ground_skill: bool = True,
        enable_trace: bool = True,
        enable_runtime_cache: bool = True,
    ):
        self.tool_registry = tool_registry
        self.enable_routing_skill = enable_routing_skill
        self.enable_evidence_skill = enable_evidence_skill
        self.enable_temporal_skill = enable_temporal_skill
        self.enable_common_ground_skill = enable_common_ground_skill
        self.enable_trace = enable_trace
        self.enable_runtime_cache = enable_runtime_cache
        self._trace: List[RuntimeTraceEntry] = []

    def execute_skill(self, name: str, context: SkillContext) -> SkillResult:
        """Execute a skill by name. Returns SkillResult with decision and optional tool_calls/results."""
        if name == "routing_skill":
            result = self._run_routing_skill(context)
        elif name == "temporal_grounding_skill":
            result = self._run_temporal_grounding_skill(context)
        elif name == "evidence_retrieval_skill":
            result = self._run_evidence_retrieval_skill(context)
        elif name == "common_ground_skill":
            result = self._run_common_ground_skill(context)
        else:
            result = SkillResult(skill_name=name, triggered=False, decision=f"Unknown skill: {name}")

        if self.enable_trace:
            self._trace.append(RuntimeTraceEntry(
                tool_name=f"skill:{result.skill_name}",
                skill_name=result.skill_name,
                args=result.output,
                latency_ms=0.0,
                cache_hit=False,
                result_count=0,
                question_type=context.question_type,
                used_supported_by_fallback=False,
                used_time_resolution=bool(result.output.get("time_query")) if isinstance(result.output, dict) else False,
            ))
        return result

    def _extract_time_query(self, query: str) -> str:
        if not query:
            return ""
        time_patterns = [
            r"\b\d{4}(?:-\d{2})?(?:-\d{2})?\b",
            r"\b(last year|last month|last week|next month|next week|yesterday|today|tomorrow)\b",
            r"\b(before|after|during|since|until|ago|recently|recent|latest|earliest)\b",
            r"\b(summer|spring|fall|winter|weekend)\b",
        ]
        matches = []
        for pattern in time_patterns:
            for match in re.findall(pattern, query, re.IGNORECASE):
                if isinstance(match, tuple):
                    match = next((part for part in match if part), "")
                if match:
                    matches.append(str(match))
        deduped = []
        for item in matches:
            lowered = item.lower()
            if lowered not in deduped:
                deduped.append(lowered)
        return " ".join(deduped[:3]).strip()

    def _build_routing_output(self, context: SkillContext) -> Dict[str, Any]:
        query = (context.query or "").lower()
        question_type = context.question_type or "single_hop"

        output: Dict[str, Any] = {
            "question_type": question_type,
            "query_conditioned_evidence": True,
            "candidate_actions": [],
            "semantic_node_types": ["fact", "event"],
        }

        if question_type == "multi_hop":
            output.update({
                "primary_action": "find_common_ground",
                "candidate_actions": ["find_common_ground", "semantic_search"],
                "semantic_node_types": ["fact", "event", "concept"],
                "require_shared_evidence": True,
            })
            return output

        if question_type == "temporal":
            output.update({
                "primary_action": "timeline_query",
                "candidate_actions": ["timeline_query", "semantic_search", "get_evidence"],
                "semantic_node_types": ["event", "fact"],
                "time_query": self._extract_time_query(context.query),
            })
            return output

        if any(token in query for token in ["think about", "feel about", "opinion", "believe", "judg", "trait"]):
            output.update({
                "primary_action": "claim_lookup",
                "candidate_actions": ["claim_lookup", "semantic_search", "get_evidence"],
                "semantic_node_types": ["claim", "fact"],
            })
            return output

        if any(token in query for token in ["like", "prefer", "enjoy", "habit", "hobby", "what does", "what did", "what has"]):
            output.update({
                "primary_action": "fact_lookup",
                "candidate_actions": ["fact_lookup", "semantic_search", "get_evidence"],
                "semantic_node_types": ["fact", "event"],
            })
            return output

        output.update({
            "primary_action": "semantic_search",
            "candidate_actions": ["semantic_search", "get_evidence"],
        })
        return output

    def _run_routing_skill(self, context: SkillContext) -> SkillResult:
        """Routing: question_type and config overrides (no tool call)."""
        if not self.enable_routing_skill:
            return SkillResult(skill_name="routing_skill", triggered=False, decision="routing_skill disabled")
        # Routing only sets context; actual classification is done by caller (SearchAgent / config.classify_question_type)
        return SkillResult(
            skill_name="routing_skill",
            triggered=True,
            decision="apply_question_type_config",
            output={
                "enabled_skills": context.enabled_skills,
                **self._build_routing_output(context),
            },
        )

    def _run_temporal_grounding_skill(self, context: SkillContext) -> SkillResult:
        """Temporal: timeline filtering / time resolution (delegates to tool when needed)."""
        if not self.enable_temporal_skill:
            return SkillResult(skill_name="temporal_grounding_skill", triggered=False, decision="temporal_skill disabled")
        return SkillResult(
            skill_name="temporal_grounding_skill",
            triggered=True,
            decision="use_timeline_filter",
            output={
                "preferred_action": "timeline_query",
                "time_query": self._extract_time_query(context.query),
                "query_conditioned_evidence": True,
            },
        )

    def _run_evidence_retrieval_skill(self, context: SkillContext) -> SkillResult:
        """Evidence: entity/fact -> chunk -> SUPPORTED_BY fallback (delegates to get_evidence tool)."""
        if not self.enable_evidence_skill:
            return SkillResult(skill_name="evidence_retrieval_skill", triggered=False, decision="evidence_skill disabled")
        return SkillResult(
            skill_name="evidence_retrieval_skill",
            triggered=True,
            decision="use_get_evidence",
            output={
                "query_conditioned": True,
                "prefer_supported_by": True,
                "query": context.query,
            },
        )

    def _run_common_ground_skill(self, context: SkillContext) -> SkillResult:
        """Common ground: multi-entity common nodes + semantic fact pattern fallback."""
        if not self.enable_common_ground_skill:
            return SkillResult(skill_name="common_ground_skill", triggered=False, decision="common_ground_skill disabled")
        return SkillResult(
            skill_name="common_ground_skill",
            triggered=True,
            decision="use_find_common_ground",
            output={
                "preferred_action": "find_common_ground",
                "require_shared_evidence": True,
                "node_types": ["concept", "event", "fact"],
            },
        )

    def execute_tool(
        self,
        name: str,
        args: Dict[str, Any],
        context: Optional[SkillContext] = None,
        *,
        skill_name: Optional[str] = None,
        query_id: str = "",
    ) -> ToolResult:
        """Execute a tool via registry with optional cache/trace. Returns ToolResult."""
        trace_id = None
        if context and self.enable_trace:
            trace_id = getattr(context, "trace_id", None) or query_id
        on_trace = None
        if self.enable_trace:
            ctx = context
            def _on_trace(entry: RuntimeTraceEntry):
                if ctx and hasattr(ctx, "question_type"):
                    entry.question_type = ctx.question_type
                self._trace.append(entry)
            on_trace = _on_trace
        return self.tool_registry.execute_tool(
            name,
            args,
            query_id=query_id,
            skill_name=skill_name,
            trace_id=trace_id,
            on_trace=on_trace,
        )

    def skill_flags(self) -> Dict[str, bool]:
        """Return current skill switches for eval/ablation."""
        return {
            "enable_routing_skill": self.enable_routing_skill,
            "enable_evidence_skill": self.enable_evidence_skill,
            "enable_temporal_skill": self.enable_temporal_skill,
            "enable_common_ground_skill": self.enable_common_ground_skill,
            "enable_trace": self.enable_trace,
            "enable_runtime_cache": self.enable_runtime_cache,
        }

    def enabled_skill_names(self) -> List[str]:
        """Return list of enabled skill names."""
        skills = []
        if self.enable_routing_skill:
            skills.append("routing_skill")
        if self.enable_evidence_skill:
            skills.append("evidence_retrieval_skill")
        if self.enable_temporal_skill:
            skills.append("temporal_grounding_skill")
        if self.enable_common_ground_skill:
            skills.append("common_ground_skill")
        return skills

    def get_trace(self) -> List[RuntimeTraceEntry]:
        """Return trace entries for this runtime."""
        return list(self._trace)

    def clear_trace(self) -> None:
        """Clear trace and registry trace."""
        self._trace.clear()
        self.tool_registry.clear_trace()
