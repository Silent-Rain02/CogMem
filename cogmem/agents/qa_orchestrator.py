"""Coordinate graph retrieval, cognitive tools, and answer generation."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from ..storage.graph_service import GraphService
from .search_agent import SearchAgent
from .skill_runtime import SkillRuntime
from .tool_registry import ToolRegistry

if TYPE_CHECKING:
    from ..cogmem import CogMem


class QAOrchestrator:
    """Build and run the query-time cognitive retrieval stack."""

    def __init__(self, cogmem: CogMem) -> None:
        self.cogmem = cogmem
        agent_cfg = cogmem.cfg.agent
        self.graph_service = GraphService(
            graph_store=cogmem.graph,
            vector_store=cogmem.vstore,
            embedding_encoder=cogmem._embed,
            use_vector_search=cogmem.cfg.faiss.use_vector_search,
            chunk_store=cogmem.chunk_store,
        )
        self.tool_registry = ToolRegistry(enable_trace=agent_cfg.enable_trace)
        self.search_agent = SearchAgent(
            graph_service=self.graph_service,
            llm=cogmem.llm,
            max_iterations=agent_cfg.max_iterations,
            config=cogmem.cfg,
            evidence_store=cogmem.indexer.evidence,
            tool_registry=self.tool_registry,
        )
        self.skill_runtime = SkillRuntime(
            self.tool_registry,
            enable_routing_skill=agent_cfg.enable_routing_skill,
            enable_evidence_skill=agent_cfg.enable_evidence_skill,
            enable_temporal_skill=agent_cfg.enable_temporal_skill,
            enable_common_ground_skill=agent_cfg.enable_common_ground_skill,
            enable_trace=agent_cfg.enable_trace,
            enable_runtime_cache=agent_cfg.enable_runtime_cache,
        )
        self.search_agent.skill_runtime = self.skill_runtime

    def answer(self, query: str, verbose: bool = False) -> str:
        """Answer a query and retain the complete trace on ``search_agent``."""
        result = self.search_agent.run(query, verbose=verbose)
        self.search_agent.last_result = result

        if verbose:
            self._print_trace(query, result)

        return self._clean_answer(result.get("answer", "I could not generate an answer."))

    @staticmethod
    def _clean_answer(answer: Any) -> str:
        """Extract a final answer from plain text or a model-produced JSON object."""
        if not isinstance(answer, str):
            return str(answer)

        candidate = answer.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = None
            extracted = QAOrchestrator._find_answer_field(parsed)
            if extracted is not None:
                return extracted

        return re.sub(
            r"^(Thought|Action|Observation|Reflection):\s*",
            "",
            candidate,
            flags=re.IGNORECASE | re.MULTILINE,
        ).strip()

    @staticmethod
    def _find_answer_field(value: Any) -> str | None:
        if not isinstance(value, dict):
            return None
        for field in ("final_answer", "answer", "response", "message", "text", "content"):
            item = value.get(field)
            if isinstance(item, str):
                return item
            if isinstance(item, dict):
                nested = QAOrchestrator._find_answer_field(item)
                if nested is not None:
                    return nested
            if isinstance(item, list) and item:
                first = item[0]
                if isinstance(first, str):
                    return first
                if isinstance(first, dict):
                    nested = QAOrchestrator._find_answer_field(first)
                    if nested is not None:
                        return nested
        return None

    @staticmethod
    def _print_trace(query: str, result: dict[str, Any]) -> None:
        print("\n" + "=" * 80)
        print("[QAOrchestrator] SearchAgent completed")
        print(f"  Query: {query}")
        print(f"  Confidence: {result.get('confidence', 'unknown')}")
        print("  Reasoning path:")
        for step in result.get("reasoning_path", []):
            if isinstance(step, dict):
                print(f"    Step {step.get('step', '?')}: {step.get('action', 'unknown')}")
            else:
                print(f"    Step {step.step}: {step.action}")
        print("=" * 80 + "\n")
