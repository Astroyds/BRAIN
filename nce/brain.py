"""Parser and data structures for the .brain graph format.

The .brain format is a UTF-8 text file with three sections:
  @section nodes      — concept / emotion / drive / response nodes
  @section edges      — weighted, typed connections between nodes
  @section responses  — response selection rules
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Node:
    """A single node in the brain graph."""
    id: str
    type: str                       # concept | emotion | drive | response
    label: str = ""
    base_activation: float = 0.0
    activation: float = 0.0         # mutable: current activation level


@dataclass
class Edge:
    """A directed, weighted edge."""
    source: str
    target: str
    weight: float = 0.5
    edge_type: str = "excitatory"   # excitatory | inhibitory


@dataclass
class ResponseRule:
    """Maps a set of trigger concepts to an intent for response selection."""
    id: str
    trigger_concepts: List[str] = field(default_factory=list)
    intent: str = ""
    priority: int = 0


class BrainGraph:
    """In-memory representation of a parsed .brain file."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        # pre-built adjacency list: source_id -> [Edge, …]
        self.adjacency: Dict[str, List[Edge]] = {}
        self.responses: List[ResponseRule] = []

    # ── public query helpers ─────────────────────────────────────────────

    def get_node(self, node_id: str) -> Optional[Node]:
        """Return the Node with *node_id*, or None."""
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str) -> List[str]:
        """Return ids of all direct neighbours reachable from *node_id*."""
        return [e.target for e in self.adjacency.get(node_id, [])]

    def get_edges_from(self, node_id: str) -> List[Edge]:
        """Return all outgoing edges from *node_id*."""
        return self.adjacency.get(node_id, [])

    # ── graph building helpers (used by parser) ──────────────────────────

    def add_node(self, node: Node) -> None:
        """Insert a node and initialise its adjacency bucket."""
        self.nodes[node.id] = node
        self.adjacency.setdefault(node.id, [])

    def add_edge(self, edge: Edge) -> None:
        """Insert an edge and update the adjacency index."""
        self.edges.append(edge)
        self.adjacency.setdefault(edge.source, []).append(edge)

    def add_response(self, rule: ResponseRule) -> None:
        """Register a response rule."""
        self.responses.append(rule)

    def reset_activations(self) -> None:
        """Set every node's current activation back to its base value."""
        for node in self.nodes.values():
            node.activation = node.base_activation


# ── Parser ───────────────────────────────────────────────────────────────────

class BrainParser:
    """Reads a .brain file and produces a BrainGraph."""

    def parse(self, filepath: str) -> BrainGraph:
        """Parse *filepath* and return a populated BrainGraph."""
        graph = BrainGraph()
        current_section: str = ""

        with open(filepath, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("@section"):
                    current_section = line.split(maxsplit=1)[1].strip().lower()
                    continue

                if current_section == "nodes":
                    self._parse_node(line, graph)
                elif current_section == "edges":
                    self._parse_edge(line, graph)
                elif current_section == "responses":
                    self._parse_response(line, graph)

        return graph

    # ── private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _parse_node(line: str, graph: BrainGraph) -> None:
        """Parse a node line.

        Format: node_id | type:T | label:L | base_activation:F
        """
        parts = [p.strip() for p in line.split("|")]
        if not parts:
            return
        node_id = parts[0]
        ntype = "concept"
        label = node_id
        base_act = 0.0

        for part in parts[1:]:
            if part.startswith("type:"):
                ntype = part[len("type:"):].strip()
            elif part.startswith("label:"):
                label = part[len("label:"):].strip()
            elif part.startswith("base_activation:"):
                try:
                    base_act = float(part[len("base_activation:"):].strip())
                except ValueError:
                    base_act = 0.0

        graph.add_node(Node(
            id=node_id,
            type=ntype,
            label=label,
            base_activation=base_act,
            activation=base_act,
        ))

    @staticmethod
    def _parse_edge(line: str, graph: BrainGraph) -> None:
        """Parse an edge line.

        Format: source_id -> target_id | weight:F | type:T
        """
        # Split on first '|' to isolate the arrow part
        segments = [s.strip() for s in line.split("|")]
        if not segments:
            return

        arrow_part = segments[0]
        if "->" not in arrow_part:
            return
        src, tgt = [s.strip() for s in arrow_part.split("->", maxsplit=1)]

        weight = 0.5
        etype = "excitatory"
        for seg in segments[1:]:
            if seg.startswith("weight:"):
                try:
                    weight = float(seg[len("weight:"):].strip())
                except ValueError:
                    weight = 0.5
            elif seg.startswith("type:"):
                etype = seg[len("type:"):].strip()

        graph.add_edge(Edge(source=src, target=tgt, weight=weight, edge_type=etype))

    @staticmethod
    def _parse_response(line: str, graph: BrainGraph) -> None:
        """Parse a response rule line.

        Format: response_id | trigger_concepts:c1,c2 | intent:I | priority:N
        """
        parts = [p.strip() for p in line.split("|")]
        if not parts:
            return
        rid = parts[0]
        triggers: List[str] = []
        intent = ""
        priority = 0

        for part in parts[1:]:
            if part.startswith("trigger_concepts:"):
                triggers = [t.strip() for t in part[len("trigger_concepts:"):].split(",") if t.strip()]
            elif part.startswith("intent:"):
                intent = part[len("intent:"):].strip()
            elif part.startswith("priority:"):
                try:
                    priority = int(part[len("priority:"):].strip())
                except ValueError:
                    priority = 0

        graph.add_response(ResponseRule(id=rid, trigger_concepts=triggers, intent=intent, priority=priority))
