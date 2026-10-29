"""Tool registry: register and execute atomic tools with optional trace."""

from __future__ import annotations
from cogmem.models.execution_errors import raise_if_execution_failure

import time
from typing import Any, Callable, Dict, Optional

from .runtime_schema import ToolCall, ToolMetrics, ToolResult, RuntimeTraceEntry

class ToolRegistry:
    """Registry of atomic tools. Executes by name and returns ToolResult with metrics."""

    def __init__(self, enable_trace: bool = True):
        self._tools: Dict[str, Callable[..., str]] = {}
        self.enable_trace = enable_trace
        self._trace_entries: list[RuntimeTraceEntry] = []

    def register(self, name: str, fn: Callable[..., str]) -> None:
        """Register a tool. Callable must return observation string."""
        self._tools[name] = fn

    def execute_tool(
        self,
        name: str,
        args: Dict[str, Any],
        *,
        query_id: str = "",
        skill_name: Optional[str] = None,
        trace_id: Optional[str] = None,
        stage: str = "retrieval",
        on_trace: Optional[Callable[[RuntimeTraceEntry], None]] = None,
    ) -> ToolResult:
        """Execute a registered tool and return ToolResult with metrics."""
        if name not in self._tools:
            return ToolResult(
                ok=False,
                tool_name=name,
                summary=f"Error: Tool {name} not found.",
                warnings=["tool_not_found"],
            )
        start = time.perf_counter()
        try:
            out = self._tools[name](**args)
            latency_ms = (time.perf_counter() - start) * 1000
            result_count = 0
            if isinstance(out, str) and ("\n" in out or " " in out):
                result_count = max(1, out.count("\n") + (1 if "Results" in out or "Found" in out else 0))
            metrics = ToolMetrics(latency_ms=latency_ms, cache_hit=False, result_count=result_count)
            result = ToolResult(ok=True, tool_name=name, summary=out, metrics=metrics)

            if self.enable_trace or on_trace:
                entry = RuntimeTraceEntry(
                    tool_name=name,
                    skill_name=skill_name,
                    args=args,
                    latency_ms=latency_ms,
                    cache_hit=False,
                    result_count=result_count,
                )
                self._trace_entries.append(entry)
                if on_trace:
                    on_trace(entry)
            return result
        except Exception as e:
            raise_if_execution_failure(e)
            latency_ms = (time.perf_counter() - start) * 1000
            result = ToolResult(
                ok=False,
                tool_name=name,
                summary=f"Tool Error: {str(e)}",
                metrics=ToolMetrics(latency_ms=latency_ms),
                warnings=[str(e)],
            )
            if self.enable_trace or on_trace:
                entry = RuntimeTraceEntry(tool_name=name, skill_name=skill_name, args=args, latency_ms=latency_ms)
                self._trace_entries.append(entry)
                if on_trace:
                    on_trace(entry)
            return result

    def list_tools(self):
        """Return registered tool names."""
        return list(self._tools.keys())

    def clear_trace(self) -> None:
        """Clear accumulated trace entries."""
        self._trace_entries.clear()

    def get_trace(self) -> list[RuntimeTraceEntry]:
        """Return current trace entries (for eval/debug)."""
        return list(self._trace_entries)
