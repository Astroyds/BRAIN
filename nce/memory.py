"""Short-term and episodic memory systems for the NCE."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Set, Tuple


# ── Short-Term Memory ────────────────────────────────────────────────────────

@dataclass
class STMEntry:
    """A single conversational turn stored in short-term memory."""
    turn_number: int
    input_text: str
    active_concepts: List[str]
    response_text: str


class ShortTermMemory:
    """Fixed-size buffer of the most recent conversational turns.

    Concepts that appeared recently receive a small activation boost
    (priming) in the next turn.
    """

    def __init__(self, capacity: int = 5) -> None:
        self._capacity: int = capacity
        self._buffer: Deque[STMEntry] = deque(maxlen=capacity)

    def add_turn(
        self,
        turn_number: int,
        input_text: str,
        active_concepts: List[str],
        response_text: str,
    ) -> None:
        """Record a completed turn."""
        self._buffer.append(STMEntry(
            turn_number=turn_number,
            input_text=input_text,
            active_concepts=list(active_concepts),
            response_text=response_text,
        ))

    def get_recent(self, n: int = 5) -> List[STMEntry]:
        """Return the *n* most recent entries (oldest first)."""
        items = list(self._buffer)
        return items[-n:]

    def get_primed_concepts(self) -> Set[str]:
        """Return the union of active concepts across recent turns."""
        primed: Set[str] = set()
        for entry in self._buffer:
            primed.update(entry.active_concepts)
        return primed


# ── Episodic Memory ──────────────────────────────────────────────────────────

@dataclass
class Episode:
    """A single episodic memory record."""
    turn: int
    context_concepts: Set[str]
    outcome_concepts: Set[str]
    reinforcement: float = 0.0


class EpisodicMemory:
    """Long-ish-term store of concept episodes with Jaccard-based recall.

    Oldest episodes are evicted once *max_episodes* is reached.
    """

    def __init__(self, max_episodes: int = 100) -> None:
        self._max: int = max_episodes
        self._episodes: List[Episode] = []

    def store_episode(
        self,
        turn: int,
        context_concepts: Set[str],
        outcome_concepts: Set[str],
        reinforcement: float = 0.0,
    ) -> None:
        """Persist a new episode, evicting the oldest if at capacity."""
        if len(self._episodes) >= self._max:
            self._episodes.pop(0)
        self._episodes.append(Episode(
            turn=turn,
            context_concepts=set(context_concepts),
            outcome_concepts=set(outcome_concepts),
            reinforcement=reinforcement,
        ))

    def recall_similar(
        self,
        current_concepts: Set[str],
        top_k: int = 3,
    ) -> List[Episode]:
        """Return the *top_k* episodes most similar to *current_concepts*.

        Similarity is measured with Jaccard index over context_concepts.
        """
        if not current_concepts or not self._episodes:
            return []

        scored: List[Tuple[float, Episode]] = []
        for ep in self._episodes:
            intersection = current_concepts & ep.context_concepts
            union = current_concepts | ep.context_concepts
            jaccard = len(intersection) / len(union) if union else 0.0
            scored.append((jaccard, ep))

        # Sort descending by similarity
        scored.sort(key=lambda t: t[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]
