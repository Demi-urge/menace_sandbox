from __future__ import annotations

from typing import Any, List, Dict, Optional

from vector_service.retriever import Retriever, FallbackResult


class PromptEngine:
    """Construct prompts from historical patch examples."""

    @staticmethod
    def _fetch_patches(query: str, top_n: int) -> List[Dict[str, Any]]:
        """Return patch records for ``query`` using the vector service."""
        retriever = Retriever()
        results = retriever.search(query, top_k=top_n, dbs=["patch"])
        if isinstance(results, FallbackResult):
            return list(results)
        return list(results)

    @staticmethod
    def _format_snippet(meta: Dict[str, Any]) -> str:
        """Compress metadata into a human friendly snippet block."""
        summary = str(meta.get("summary", "")).strip()
        diff = str(meta.get("diff", "")).strip()
        outcome = str(meta.get("outcome", "")).strip()
        tests_passed = meta.get("tests_passed")

        status: str = ""
        if tests_passed is True:
            status = "tests passed"
        elif tests_passed is False:
            status = "tests failed"

        if outcome and status:
            outcome_text = f"{outcome} ({status})"
        elif outcome:
            outcome_text = outcome
        else:
            outcome_text = status

        parts = []
        if summary:
            parts.append(f"Code summary: {summary}")
        if diff:
            parts.append(f"Diff summary: {diff}")
        if outcome_text:
            parts.append(f"Outcome: {outcome_text}")

        if not parts:
            return ""

        first, *rest = parts
        snippet = f"- {first}"
        for p in rest:
            snippet += f"\n  {p}"
        return snippet

    @classmethod
    def construct_prompt(
        cls, description: str, retry_trace: Optional[str] = None, top_n: int = 5
    ) -> str:
        """Return a structured prompt with patch examples for ``description``."""
        records = cls._fetch_patches(description, top_n)
        positives: List[str] = []
        negatives: List[str] = []
        for rec in records:
            meta = rec.get("metadata", {}) if isinstance(rec, dict) else {}
            snippet = cls._format_snippet(meta)
            if not snippet:
                continue
            if meta.get("tests_passed"):
                positives.append(snippet)
            else:
                negatives.append(snippet)

        sections: List[str] = []
        if positives:
            sections.append("Given the following pattern...\n")
            sections.append("\n".join(positives))
        if negatives:
            if sections:
                sections.append("\n")
            sections.append("Avoid...\n")
            sections.append("\n".join(negatives))
        if retry_trace:
            if sections:
                sections.append("\n")
            sections.append(f"{retry_trace}\nPlease try a different approach.")
        return "".join(sections).strip()
