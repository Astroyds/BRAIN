"""Core activation engine — runs the full NCE pipeline for each turn."""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from nce.brain import BrainGraph, ResponseRule
from nce.memory import EpisodicMemory, ShortTermMemory
from nce.nol import NolData
from nce.realize import Realizer
from nce.utils import Profiler, ThoughtTrace


# Simple regex: split on whitespace and common punctuation
_TOKEN_RE = re.compile(r"[a-z0-9']+")


class NCEEngine:
    """Orchestrates tokenisation, activation spreading, response selection,
    and surface realisation for each conversational turn.
    """

    # Activation threshold — nodes below this are considered inactive
    ACTIVATION_THRESHOLD: float = 0.05

    def __init__(
        self,
        graph: BrainGraph,
        nol: NolData,
        stm: ShortTermMemory,
        episodic: EpisodicMemory,
        profiler: Profiler,
    ) -> None:
        self.graph: BrainGraph = graph
        self.nol: NolData = nol
        self.stm: ShortTermMemory = stm
        self.episodic: EpisodicMemory = episodic
        self.profiler: Profiler = profiler
        self.realizer: Realizer = Realizer()

        # Modulators scale the spreading activation (default neutral)
        self.modulators: Dict[str, float] = {
            "curiosity": 1.0,
            "urgency": 1.0,
        }

        self._turn: int = 0

    # ──────────────────────────────────────────────────────────────────────
    # Pipeline stages
    # ──────────────────────────────────────────────────────────────────────

    def tokenize(self, input_text: str) -> List[str]:
        """Stage 1 — whitespace + punctuation split, lowercased."""
        self.profiler.start_stage("tokenize")
        tokens = _TOKEN_RE.findall(input_text.lower())
        self.profiler.end_stage("tokenize")
        return tokens

    def map_to_concepts(self, tokens: List[str]) -> List[str]:
        """Stage 2 — look up tokens (and synonyms) in the NolData vocab."""
        self.profiler.start_stage("activate")
        concept_ids: List[str] = []
        seen: Set[str] = set()
        for token in tokens:
            entry = self.nol.vocab.get(token)
            if entry and entry.concept_id not in seen:
                concept_ids.append(entry.concept_id)
                seen.add(entry.concept_id)
        self.profiler.end_stage("activate")
        return concept_ids

    def inject_activation(self, concept_ids: List[str]) -> None:
        """Stage 3 — seed matched concepts with activation 1.0 and apply STM priming."""
        for cid in concept_ids:
            node = self.graph.get_node(cid)
            if node:
                node.activation = 1.0
                self.profiler.activated_nodes += 1

        # Priming boost from short-term memory
        primed = self.stm.get_primed_concepts()
        for pid in primed:
            node = self.graph.get_node(pid)
            if node and node.activation < 1.0:
                node.activation = min(1.0, node.activation + 0.3)

    def spread_activation(
        self,
        trace: ThoughtTrace,
        steps: int = 3,
        decay: float = 0.8,
    ) -> None:
        """Stage 4 — iterative spreading activation across the graph."""
        self.profiler.start_stage("spread")

        # Compute a combined modulator multiplier
        mod_factor: float = 1.0
        for val in self.modulators.values():
            mod_factor *= val

        for step in range(steps):
            # Pre-compute decay factor once per step to avoid redundant exponentiation
            decay_factor: float = decay ** step

            # Snapshot current activations so updates don't feed forward within a step
            updates: Dict[str, float] = {}

            for node in self.graph.nodes.values():
                if node.activation <= self.ACTIVATION_THRESHOLD:
                    continue

                # Record active node in trace
                trace.record_node(step, node.id, node.activation)

                for edge in self.graph.get_edges_from(node.id):
                    target = self.graph.get_node(edge.target)
                    if target is None:
                        continue

                    self.profiler.traversed_edges += 1
                    trace.record_edge(step, edge.source, edge.target, edge.weight)

                    delta: float
                    if edge.edge_type == "excitatory":
                        delta = node.activation * edge.weight * decay_factor
                    else:  # inhibitory
                        delta = -(node.activation * abs(edge.weight) * 0.5)

                    # Apply modulator scaling
                    delta *= mod_factor
                    updates[edge.target] = updates.get(edge.target, 0.0) + delta

            # Apply accumulated deltas
            for nid, delta in updates.items():
                target = self.graph.get_node(nid)
                if target:
                    target.activation = max(0.0, min(1.0, target.activation + delta))

            self.profiler.steps_executed += 1

        self.profiler.end_stage("spread")

    def select_response(self, trace: ThoughtTrace) -> ResponseRule:
        """Stage 5 — score response rules and pick the best one."""
        self.profiler.start_stage("plan")

        best_rule: ResponseRule | None = None
        best_score: float = -1.0

        for rule in self.graph.responses:
            score = sum(
                (self.graph.get_node(c).activation if self.graph.get_node(c) else 0.0)
                for c in rule.trigger_concepts
            )
            # Weight by priority
            score += rule.priority * 0.01
            if score > best_score:
                best_score = score
                best_rule = rule

        # Fallback rule when nothing fires
        if best_rule is None or best_score < self.ACTIVATION_THRESHOLD:
            best_rule = ResponseRule(id="r_fallback", trigger_concepts=[], intent="unknown", priority=0)

        trace.response_intent = best_rule.intent
        self.profiler.end_stage("plan")
        return best_rule

    def plan_response(self, rule: ResponseRule) -> Tuple[str, str]:
        """Stage 6 — map a ResponseRule to (intent, template_text)."""
        self.profiler.start_stage("realize")

        # Gather active concepts sorted by activation (descending)
        active: List[Tuple[float, str]] = [
            (n.activation, n.id)
            for n in self.graph.nodes.values()
            if n.activation > self.ACTIVATION_THRESHOLD
        ]
        active.sort(reverse=True)
        active_ids = [cid for _, cid in active]

        # Build label map
        labels: Dict[str, str] = {
            n.id: n.label for n in self.graph.nodes.values()
        }

        text = self.realizer.realize(rule.intent, active_ids, self.nol, labels)
        self.profiler.end_stage("realize")
        return rule.intent, text

    # ──────────────────────────────────────────────────────────────────────
    # Full turn orchestration
    # ──────────────────────────────────────────────────────────────────────

    def run_turn(self, input_text: str) -> Dict[str, object]:
        """Execute the complete NCE pipeline for one user turn.

        Returns a dict with keys:
          response_text, thought_trace, profiling_data
        """
        self.profiler.reset()
        self.graph.reset_activations()
        trace = ThoughtTrace()
        self._turn += 1

        # 1. Tokenize
        tokens = self.tokenize(input_text)

        # 2. Map tokens to concepts
        concept_ids = self.map_to_concepts(tokens)

        # 3. Inject activation
        self.inject_activation(concept_ids)

        # 4. Spread activation
        self.spread_activation(trace, steps=3, decay=0.8)

        # 5. Select response rule
        rule = self.select_response(trace)

        # 6. Plan & realise response
        intent, response_text = self.plan_response(rule)

        # Record final concepts (those still above threshold)
        final_concepts = [
            n.id for n in self.graph.nodes.values()
            if n.activation > self.ACTIVATION_THRESHOLD
        ]
        trace.final_concepts = sorted(final_concepts)

        # Persist to short-term memory
        self.stm.add_turn(self._turn, input_text, concept_ids, response_text)

        # Persist to episodic memory
        self.episodic.store_episode(
            turn=self._turn,
            context_concepts=set(concept_ids),
            outcome_concepts=set(final_concepts),
            reinforcement=0.0,
        )

        return {
            "response_text": response_text,
            "thought_trace": trace,
            "profiling_data": self.profiler.report(),
        }
