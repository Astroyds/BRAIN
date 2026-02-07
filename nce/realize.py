"""Surface realizer: selects and fills response templates."""

from __future__ import annotations

from typing import Dict, List

from nce.nol import NolData


class Realizer:
    """Converts an intent + active concepts into a natural-language response.

    Template placeholders of the form ``{concept}`` are replaced with the
    label of the most activated matching concept.
    """

    # Fallback when no template matches the intent
    _FALLBACK = "I'm not sure what to say about that."

    def realize(
        self,
        intent: str,
        active_concepts: List[str],
        nol: NolData,
        concept_labels: Dict[str, str] | None = None,
    ) -> str:
        """Produce a surface string for *intent*.

        Parameters
        ----------
        intent:
            The selected response intent (e.g. ``"greeting"``).
        active_concepts:
            Concept ids ordered by descending activation.
        nol:
            The loaded NolData (contains templates and vocab).
        concept_labels:
            Optional mapping concept_id -> human-readable label.
            Used for ``{concept}`` placeholder substitution.

        Returns
        -------
        str
            The final response text.
        """
        templates = nol.templates.get(intent)
        if not templates:
            return self._FALLBACK

        # Pick the first available template for this intent
        template = templates[0]

        # If the template contains {concept}, substitute with best label
        if "{concept}" in template:
            label = self._best_label(active_concepts, concept_labels)
            template = template.replace("{concept}", label)

        return template

    # ── private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _best_label(
        active_concepts: List[str],
        concept_labels: Dict[str, str] | None,
    ) -> str:
        """Return a human-readable label for the top active concept."""
        if not active_concepts:
            return "something"
        top = active_concepts[0]
        if concept_labels and top in concept_labels:
            return concept_labels[top]
        # Strip the c_ prefix as a rough fallback
        return top.replace("c_", "").replace("_", " ")
