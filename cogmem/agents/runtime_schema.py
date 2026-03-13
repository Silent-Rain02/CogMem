"""Runtime schema for tool/skill execution: trace, metrics, and ablation-friendly output.

Used by ToolRegistry and SkillRuntime; kept separate from storage/schema.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


Stage = Literal["ingestion", "retrieval", "consolidation"]


class ToolCall(BaseModel):
    """Record of a single tool invocation."""
    name: str
    stage: Stage = "retrieval"
    args: Dict[str, Any] = Field(default_factory=dict)
    query_id: str = ""
    skill_name: Optional[str] = None
    trace_id: Optional[str] = None


class ToolMetrics(BaseModel):
    """Per-call metrics for a tool."""
    latency_ms: float = 0.0
    cache_hit: bool = False
    result_count: int = 0


class ToolResult(BaseModel):
    """Structured result of a tool execution."""
    ok: bool = True
    tool_name: str = ""
    items: List[Dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
    metrics: ToolMetrics = Field(default_factory=ToolMetrics)
    warnings: List[str] = Field(default_factory=list)


class SkillContext(BaseModel):
    """Context passed into a skill (query, stage, enabled skills, budgets)."""
    query: str = ""
    question_type: Optional[str] = None
    stage: Stage = "retrieval"
    enabled_skills: List[str] = Field(default_factory=list)
    budgets: Dict[str, Any] = Field(default_factory=dict)
    flags: Dict[str, bool] = Field(default_factory=dict)


class SkillResult(BaseModel):
    """Output of a skill execution (triggered, decision, tool_calls, output)."""
    skill_name: str = ""
    triggered: bool = False
    decision: str = ""
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_results: List[ToolResult] = Field(default_factory=list)
    output: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = None


class RuntimeTraceEntry(BaseModel):
    """Single trace entry for tool/skill (for eval and debugging)."""
    tool_name: str = ""
    skill_name: Optional[str] = None
    args: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    cache_hit: bool = False
    result_count: int = 0
    question_type: Optional[str] = None
    used_supported_by_fallback: bool = False
    used_time_resolution: bool = False