"""Profiling and tracing utilities for the NCE pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


class Profiler:
    """Tracks wall-clock time per pipeline stage and aggregate counts."""

    def __init__(self) -> None:
        # stage_name -> cumulative seconds
        self._timings: Dict[str, float] = {}
        # stage_name -> start timestamp (while running)
        self._starts: Dict[str, float] = {}
        # aggregate counters
        self.activated_nodes: int = 0
        self.traversed_edges: int = 0
        self.steps_executed: int = 0

    def start_stage(self, name: str) -> None:
        """Mark the beginning of a named pipeline stage."""
        self._starts[name] = time.perf_counter()

    def end_stage(self, name: str) -> None:
        """Mark the end of a named pipeline stage and accumulate elapsed time."""
        if name in self._starts:
            elapsed = time.perf_counter() - self._starts.pop(name)
            self._timings[name] = self._timings.get(name, 0.0) + elapsed

    def report(self) -> Dict[str, object]:
        """Return a dict summarising timings (ms) and counts."""
        return {
            "timings_ms": {k: round(v * 1000, 4) for k, v in self._timings.items()},
            "activated_nodes": self.activated_nodes,
            "traversed_edges": self.traversed_edges,
            "steps_executed": self.steps_executed,
        }

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._timings.clear()
        self._starts.clear()
        self.activated_nodes = 0
        self.traversed_edges = 0
        self.steps_executed = 0


@dataclass
class ThoughtTrace:
    """Records the step-by-step activation history of a single turn."""

    # (step, node_id, activation_value)
    node_activations: List[Tuple[int, str, float]] = field(default_factory=list)
    # (step, edge_src, edge_dst, weight)
    edge_traversals: List[Tuple[int, str, str, float]] = field(default_factory=list)
    # concept ids that were finally selected
    final_concepts: List[str] = field(default_factory=list)
    # the intent chosen for the response
    response_intent: str = ""

    def record_node(self, step: int, node_id: str, activation: float) -> None:
        """Log a node activation at a given spreading step."""
        self.node_activations.append((step, node_id, round(activation, 4)))

    def record_edge(self, step: int, src: str, dst: str, weight: float) -> None:
        """Log an edge traversal at a given spreading step."""
        self.edge_traversals.append((step, src, dst, round(weight, 4)))

    def pretty_print(self) -> str:
        """Return a human-readable multi-line representation."""
        lines: List[str] = ["─── Thought Trace ───"]

        # Group node activations by step
        steps_seen: Dict[int, List[Tuple[str, float]]] = {}
        for step, nid, val in self.node_activations:
            steps_seen.setdefault(step, []).append((nid, val))

        for step in sorted(steps_seen):
            lines.append(f"  Step {step}:")
            for nid, val in steps_seen[step]:
                bar = "█" * int(val * 20)
                lines.append(f"    {nid:<20s} act={val:.4f}  {bar}")

        if self.edge_traversals:
            lines.append("  Edges traversed:")
            for step, src, dst, w in self.edge_traversals:
                lines.append(f"    step {step}: {src} ──({w:.2f})──▶ {dst}")

        lines.append(f"  Final concepts : {', '.join(self.final_concepts) if self.final_concepts else '(none)'}")
        lines.append(f"  Response intent: {self.response_intent or '(none)'}")
        lines.append("─────────────────────")
        return "\n".join(lines)
