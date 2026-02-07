#!/usr/bin/env python3
"""Neuron Conversation Engine — REPL entry point.

Usage:
    python -m nce.main [--nol PATH] [--brain PATH]
    python nce/main.py [--nol PATH] [--brain PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

# Ensure the project root is on sys.path so `python nce/main.py` works
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from nce.brain import BrainParser
from nce.engine import NCEEngine
from nce.memory import EpisodicMemory, ShortTermMemory
from nce.nol import NolParser
from nce.utils import Profiler, ThoughtTrace


# ── Helpers ──────────────────────────────────────────────────────────────────

_BANNER = r"""
╔══════════════════════════════════════════════╗
║   Neuron Conversation Engine  (NCE)  v0.1    ║
║   Type 'quit' or 'exit' to leave.            ║
╚══════════════════════════════════════════════╝
"""


def _format_profiling(data: Dict[str, object]) -> str:
    """Pretty-print profiling data returned by the engine."""
    lines = ["── Profiling ──"]
    timings = data.get("timings_ms", {})
    if isinstance(timings, dict):
        for stage, ms in timings.items():
            lines.append(f"  {stage:<12s} {ms:>8.4f} ms")
    lines.append(f"  {'nodes':.<12s} {data.get('activated_nodes', 0)}")
    lines.append(f"  {'edges':.<12s} {data.get('traversed_edges', 0)}")
    lines.append(f"  {'steps':.<12s} {data.get('steps_executed', 0)}")
    lines.append("───────────────")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse arguments, load data files, and run the interactive REPL."""
    parser = argparse.ArgumentParser(description="Neuron Conversation Engine REPL")
    parser.add_argument("--nol", default=os.path.join(_PROJECT_ROOT, "example.nol"),
                        help="Path to .nol vocabulary/template file")
    parser.add_argument("--brain", default=os.path.join(_PROJECT_ROOT, "example.brain"),
                        help="Path to .brain graph file")
    args = parser.parse_args()

    # Load data files
    try:
        nol_data = NolParser().parse(args.nol)
        brain_graph = BrainParser().parse(args.brain)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Failed to parse data files: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(nol_data.vocab)} vocab entries, "
          f"{len(nol_data.templates)} template intents.")
    print(f"Loaded {len(brain_graph.nodes)} nodes, "
          f"{len(brain_graph.edges)} edges, "
          f"{len(brain_graph.responses)} response rules.")

    # Bootstrap engine components
    stm = ShortTermMemory(capacity=5)
    episodic = EpisodicMemory(max_episodes=100)
    profiler = Profiler()
    engine = NCEEngine(brain_graph, nol_data, stm, episodic, profiler)

    print(_BANNER)

    # REPL loop
    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        result = engine.run_turn(user_input)

        # Display response
        print(f"\nNCE> {result['response_text']}\n")

        # Display thought trace
        trace: ThoughtTrace = result["thought_trace"]
        print(trace.pretty_print())

        # Display profiling stats
        print(_format_profiling(result["profiling_data"]))
        print()


if __name__ == "__main__":
    main()
