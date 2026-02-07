"""Parser for the .nol vocabulary / template format.

The .nol format is a UTF-8 text file with two sections:
  @section vocabulary   — word-to-concept mappings
  @section templates    — intent-keyed response templates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ConceptEntry:
    """A single vocabulary entry mapping a surface word to a concept."""
    concept_id: str
    synonyms: List[str] = field(default_factory=list)
    category: str = ""
    sentiment: float = 0.0


@dataclass
class NolData:
    """Parsed contents of a .nol file."""
    # surface_word (lowercased) -> ConceptEntry
    vocab: Dict[str, ConceptEntry] = field(default_factory=dict)
    # intent_name -> list of template strings
    templates: Dict[str, List[str]] = field(default_factory=dict)


# ── Parser ───────────────────────────────────────────────────────────────────

class NolParser:
    """Reads a .nol file and produces a NolData instance."""

    def parse(self, filepath: str) -> NolData:
        """Parse *filepath* and return structured NolData."""
        data = NolData()
        current_section: str = ""

        with open(filepath, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()

                # Skip blanks and comments
                if not line or line.startswith("#"):
                    continue

                # Detect section headers
                if line.startswith("@section"):
                    current_section = line.split(maxsplit=1)[1].strip().lower()
                    continue

                # Dispatch to section-specific handler
                if current_section == "vocabulary":
                    self._parse_vocab_line(line, data)
                elif current_section == "templates":
                    self._parse_template_line(line, data)

        return data

    # ── private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _parse_vocab_line(line: str, data: NolData) -> None:
        """Parse a single vocabulary line into *data*.

        Format: word_or_phrase | concept_id | synonyms:s1,s2 | category:cat | sentiment:val
        """
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            return  # malformed

        surface_word = parts[0].lower()
        concept_id = parts[1]
        synonyms: List[str] = []
        category: str = ""
        sentiment: float = 0.0

        for part in parts[2:]:
            if part.startswith("synonyms:"):
                synonyms = [s.strip() for s in part[len("synonyms:"):].split(",") if s.strip()]
            elif part.startswith("category:"):
                category = part[len("category:"):].strip()
            elif part.startswith("sentiment:"):
                try:
                    sentiment = float(part[len("sentiment:"):].strip())
                except ValueError:
                    sentiment = 0.0

        entry = ConceptEntry(
            concept_id=concept_id,
            synonyms=synonyms,
            category=category,
            sentiment=sentiment,
        )

        # Index the primary surface word
        data.vocab[surface_word] = entry
        # Also index every synonym so lookup works both ways
        for syn in synonyms:
            data.vocab[syn.lower()] = entry

    @staticmethod
    def _parse_template_line(line: str, data: NolData) -> None:
        """Parse a single template line.

        Format: intent_name | template_string
        """
        parts = [p.strip() for p in line.split("|", maxsplit=1)]
        if len(parts) < 2:
            return
        intent = parts[0]
        template = parts[1]
        data.templates.setdefault(intent, []).append(template)
