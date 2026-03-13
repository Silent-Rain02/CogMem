from __future__ import annotations

from typing import Dict, List, Any, Optional
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field

@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievalMetrics:
    """Metrics for retrieval operations."""
    query: str
    total_time: float
    step_times: Dict[str, float] = field(default_factory=dict)
    retrieval_path: List[str] = field(default_factory=list)
    entities_found: int = 0
    candidates_generated: int = 0
    evidence_returned: int = 0
    confidence_score: float = 0.0
    fallback_used: bool = False
    error_occurred: bool = False
    error_details: Optional[str] = None

class SystemMonitor:
    """System-wide monitoring and performance tracking."""

    def __init__(self):
        self._active_operations: Dict[str, OperationMetrics] = {}
        self._completed_operations: List[OperationMetrics] = []
        self._retrieval_history: List[RetrievalMetrics] = []
        self._lock = threading.Lock()

    @contextmanager
    def track_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager to track operation timing and results."""
        operation_id = f"{operation_name}_{threading.current_thread().ident}_{time.time()}"

        with self._lock:
            self._active_operations[operation_id] = OperationMetrics(
                operation_name=operation_name,
                start_time=time.time(),
                metadata=metadata or {}
            )

        try:
            yield operation_id
            with self._lock:
                if operation_id in self._active_operations:
                    op = self._active_operations[operation_id]
                    op.end_time = time.time()
                    op.duration = op.end_time - op.start_time
                    self._completed_operations.append(op)
                    del self._active_operations[operation_id]

        except Exception as e:
            with self._lock:
                if operation_id in self._active_operations:
                    op = self._active_operations[operation_id]
                    op.end_time = time.time()
                    op.duration = op.end_time - op.start_time
                    op.success = False
                    op.error_message = str(e)
                    self._completed_operations.append(op)
                    del self._active_operations[operation_id]
            raise

    def start_retrieval_tracking(self, query: str) -> str:
        """Start tracking a retrieval operation."""
        retrieval_id = f"retrieval_{threading.current_thread().ident}_{time.time()}"

        with self._lock:
            self._active_operations[retrieval_id] = OperationMetrics(
                operation_name="retrieval",
                start_time=time.time(),
                metadata={"query": query, "retrieval_id": retrieval_id}
            )

        return retrieval_id

    def update_retrieval_metrics(self, retrieval_id: str, **updates):
        """Update metrics for an ongoing retrieval operation."""
        with self._lock:
            if retrieval_id in self._active_operations:
                op = self._active_operations[retrieval_id]
                op.metadata.update(updates)

    def complete_retrieval_tracking(self, retrieval_id: str, retrieval_path: List[str],
                                   entities_count: int, candidates_count: int,
                                   evidence_count: int, confidence_score: float,
                                   fallback_used: bool = False, error: Optional[str] = None):
        """Complete tracking of a retrieval operation."""
        with self._lock:
            if retrieval_id in self._active_operations:
                op = self._active_operations[retrieval_id]
                op.end_time = time.time()
                op.duration = op.end_time - op.start_time

                # Calculate step times from metadata
                step_times = {}
                step_start_times = op.metadata.get("step_start_times", {})

                for step_name, start_time in step_start_times.items():
                    if step_name in op.metadata.get("step_end_times", {}):
                        end_time = op.metadata["step_end_times"][step_name]
                        step_times[step_name] = end_time - start_time

                # Create retrieval metrics
                retrieval_metrics = RetrievalMetrics(
                    query=op.metadata.get("query", ""),
                    total_time=op.duration,
                    step_times=step_times,
                    retrieval_path=retrieval_path,
                    entities_found=entities_count,
                    candidates_generated=candidates_count,
                    evidence_returned=evidence_count,
                    confidence_score=confidence_score,
                    fallback_used=fallback_used,
                    error_occurred=error is not None,
                    error_details=error
                )

                self._retrieval_history.append(retrieval_metrics)
                self._completed_operations.append(op)
                del self._active_operations[retrieval_id]

    def get_performance_summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Get performance summary statistics."""
        with self._lock:
            operations = self._completed_operations
            retrievals = self._retrieval_history

            if last_n:
                operations = operations[-last_n:]
                retrievals = retrievals[-last_n:]

            if not operations:
                return {"message": "No operations recorded yet"}

            # Overall stats
            total_operations = len(operations)
            successful_operations = sum(1 for op in operations if op.success)
            failed_operations = total_operations - successful_operations

            operation_durations = [op.duration for op in operations if op.duration is not None]
            avg_duration = sum(operation_durations) / len(operation_durations) if operation_durations else 0

            # Retrieval-specific stats
            if retrievals:
                avg_retrieval_time = sum(r.total_time for r in retrievals) / len(retrievals)
                avg_entities_found = sum(r.entities_found for r in retrievals) / len(retrievals)
                avg_candidates = sum(r.candidates_generated for r in retrievals) / len(retrievals)
                avg_evidence = sum(r.evidence_returned for r in retrievals) / len(retrievals)
                avg_confidence = sum(r.confidence_score for r in retrievals) / len(retrievals)
                fallback_rate = sum(1 for r in retrievals if r.fallback_used) / len(retrievals)
                error_rate = sum(1 for r in retrievals if r.error_occurred) / len(retrievals)

                # Step time breakdown
                step_times = {}
                for retrieval in retrievals:
                    for step, duration in retrieval.step_times.items():
                        if step not in step_times:
                            step_times[step] = []
                        step_times[step].append(duration)

                avg_step_times = {step: sum(times) / len(times) for step, times in step_times.items()}

                return {
                    "overall": {
                        "total_operations": total_operations,
                        "success_rate": successful_operations / total_operations,
                        "average_duration": avg_duration,
                        "failed_operations": failed_operations
                    },
                    "retrieval": {
                        "total_retrievals": len(retrievals),
                        "average_time": avg_retrieval_time,
                        "average_entities_found": avg_entities_found,
                        "average_candidates_generated": avg_candidates,
                        "average_evidence_returned": avg_evidence,
                        "average_confidence_score": avg_confidence,
                        "fallback_rate": fallback_rate,
                        "error_rate": error_rate,
                        "step_breakdown": avg_step_times
                    }
                }

            return {
                "overall": {
                    "total_operations": total_operations,
                    "success_rate": successful_operations / total_operations,
                    "average_duration": avg_duration,
                    "failed_operations": failed_operations
                }
            }

    def get_recent_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get details of recent operations."""
        with self._lock:
            recent_ops = self._completed_operations[-limit:]
            return [
                {
                    "operation": op.operation_name,
                    "duration": op.duration,
                    "success": op.success,
                    "error": op.error_message,
                    "metadata": op.metadata
                }
                for op in recent_ops
            ]

    def clear_history(self):
        """Clear operation history."""
        with self._lock:
            self._completed_operations.clear()
            self._retrieval_history.clear()
            self._active_operations.clear()

# Global monitor instance
monitor = SystemMonitor()
