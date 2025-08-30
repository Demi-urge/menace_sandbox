from __future__ import annotations

import logging
from typing import Any, List, Dict, Optional, Tuple

from vector_service.retriever import Retriever, FallbackResult


logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE = "No relevant patch examples available."
CONFIDENCE_THRESHOLD = 0.2


class PromptEngine:
    """Construct prompts from historical patch examples."""

    @staticmethod
    def _fetch_patches(query: str, top_n: int) -> Tuple[List[Dict[str, Any]], float]:
        """Return patch records and confidence for ``query``."""
        retriever = Retriever()
        results = retriever.search(query, top_k=top_n, dbs=["patch"])
        confidence = 1.0
        if isinstance(results, FallbackResult):
            confidence = results.confidence
            return list(results), confidence
        try:
            confidence = max(r.get("score", 0.0) for r in results) if results else 0.0
        except Exception:
            confidence = 0.0
        return list(results), confidence

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
        records, confidence = cls._fetch_patches(description, top_n)
        if confidence < CONFIDENCE_THRESHOLD:
            logger.info(
                "PromptEngine falling back to default template: confidence %.3f",
                confidence,
            )
            if retry_trace:
                return f"{DEFAULT_TEMPLATE}\n{retry_trace}"
            return DEFAULT_TEMPLATE

        # Rank records by ROI delta when available, otherwise by recency
        def _score(rec: Dict[str, Any]) -> Tuple[int, float]:
            meta = rec.get("metadata", {}) if isinstance(rec, dict) else {}
            roi = meta.get("roi_delta")
            ts = (
                meta.get("ts")
                or meta.get("timestamp")
                or meta.get("created_at")
                or 0.0
            )
            try:
                if roi is not None:
                    return 1, float(roi)
                return 0, float(ts)
            except Exception:
                return 0, 0.0

        records.sort(key=_score, reverse=True)
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
        if sections:
            if retry_trace:
                sections.append("\n" if sections else "")
                sections.append(f"{retry_trace}\nPlease try a different approach.")
            return "".join(sections).strip()

        logger.info("PromptEngine falling back to default template: no snippets")
        if retry_trace:
            return f"{DEFAULT_TEMPLATE}\n{retry_trace}"
        return DEFAULT_TEMPLATE
