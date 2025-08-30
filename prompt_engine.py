"""Utilities to assemble prompts for self-coding helpers.

This module queries the vector service for past patches, compresses the
returned metadata and builds a structured prompt that guides the language
model towards the current enhancement goal.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Iterable, List, Tuple

try:  # pragma: no cover - optional at runtime
    from .roi_tracker import ROITracker  # type: ignore
except Exception:  # pragma: no cover - fallback when running in isolation
    ROITracker = None  # type: ignore

try:  # pragma: no cover - optional dependency used for diff compression
    from .micro_models.diff_summarizer import summarize_diff  # type: ignore
except Exception:  # pragma: no cover - summariser may be missing
    def summarize_diff(before: str, after: str, max_new_tokens: int = 128) -> str:  # type: ignore
        return (before + "\n" + after).strip()

DEFAULT_TEMPLATE = (
    "No relevant patches were found. Proceed with a fresh implementation."
)


@dataclass
class PromptEngine:
    """Build prompts for the :class:`SelfCodingEngine`.

    Parameters
    ----------
    roi_tracker:
        Optional :class:`ROITracker` instance used to calculate risk adjusted
        ROI values when ``roi_delta`` metadata is available.
    """

    roi_tracker: ROITracker | None = None

    # ------------------------------------------------------------------
    @staticmethod
    def _fetch_patches(goal: str, top_n: int) -> Tuple[List[Dict[str, Any]], float]:
        """Return ``(records, confidence)`` for ``goal``.

        The default implementation queries :class:`PatchRetriever` from the
        vector service.  The confidence score is derived from the best hit
        similarity and normalised to the ``[0, 1]`` range.  Test suites monkey
        patch this helper to isolate the prompt construction logic.
        """

        try:  # pragma: no cover - heavy dependency optional in tests
            from .vector_service.retriever import PatchRetriever  # type: ignore
        except Exception:  # pragma: no cover - vector service unavailable
            return [], 0.0

        retriever = PatchRetriever(top_k=top_n)
        records = retriever.search(goal, top_k=top_n) or []
        confidence = 0.0
        for rec in records:
            score = rec.get("score") or rec.get("similarity") or 0.0
            confidence = max(confidence, float(score))
        return records, confidence

    # ------------------------------------------------------------------
    @staticmethod
    def _compress(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Return a condensed copy of ``meta`` using micro-models.

        Only a handful of fields are kept to keep the final prompt lean.  The
        function uses the diff summariser when ``before``/``after`` snippets are
        present and truncates long code blocks or logs.
        """

        out: Dict[str, Any] = {}
        out["summary"] = meta.get("summary") or meta.get("description")

        before = meta.get("before")
        after = meta.get("after")
        diff = meta.get("diff")
        if before or after:
            try:
                out["diff"] = summarize_diff(before or "", after or "")
            except Exception:  # pragma: no cover - defensive
                out["diff"] = diff
        elif diff:
            out["diff"] = diff

        snippet = meta.get("snippet") or meta.get("code")
        if isinstance(snippet, str):
            out["snippet"] = snippet if len(snippet) < 200 else snippet[:197] + "..."

        log = meta.get("test_log")
        if isinstance(log, str):
            out["test_log"] = log if len(log) < 200 else log[:197] + "..."

        out["outcome"] = meta.get("outcome")
        out["tests_passed"] = meta.get("tests_passed")
        out["roi_delta"] = meta.get("roi_delta")
        out["ts"] = meta.get("ts")
        out["context"] = meta.get("context") or meta.get("retrieval_context")
        return out

    # ------------------------------------------------------------------
    def _format_record(self, meta: Dict[str, Any]) -> List[str]:
        """Return human readable lines for a single patch ``meta``."""

        if self.roi_tracker and meta.get("roi_delta") is not None:
            try:  # pragma: no cover - best effort integration
                _, raroi, _ = self.roi_tracker.calculate_raroi(float(meta["roi_delta"]))
                meta["raroi"] = raroi
            except Exception:
                pass

        lines: List[str] = []
        if meta.get("summary"):
            lines.append(f"Code summary: {meta['summary']}")
        if meta.get("diff"):
            lines.append(f"Diff summary: {meta['diff']}")
        if meta.get("outcome"):
            status = "tests passed" if meta.get("tests_passed") else "tests failed"
            lines.append(f"Outcome: {meta['outcome']} ({status})")
        if meta.get("raroi") is not None:
            lines.append(f"Risk-adjusted ROI: {meta['raroi']:.2f}")
        elif meta.get("roi_delta") is not None:
            lines.append(f"ROI delta: {meta['roi_delta']}")
        if meta.get("context"):
            lines.append(f"Context: {meta['context']}")
        return lines

    # ------------------------------------------------------------------
    def build_prompt(
        self,
        goal: str,
        retry_trace: str | None = None,
        *,
        top_n: int = 5,
        confidence_threshold: float = 0.3,
    ) -> str:
        """Assemble a prompt for ``goal``.

        When the retrieval confidence falls below ``confidence_threshold`` the
        function returns :data:`DEFAULT_TEMPLATE` instead of an unhelpful prompt.
        """

        records, confidence = self._fetch_patches(goal, top_n)
        if not records or confidence < confidence_threshold:
            logging.info(
                "Retrieval confidence %.2f below threshold; falling back", confidence
            )
            return DEFAULT_TEMPLATE

        successes: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        for rec in records:
            meta = self._compress(rec.get("metadata", {}))
            if meta.get("tests_passed"):
                successes.append(meta)
            else:
                failures.append(meta)

        successes.sort(key=lambda m: m.get("roi_delta") or 0.0, reverse=True)
        failures.sort(key=lambda m: m.get("ts") or 0, reverse=True)

        lines = [f"Enhancement goal: {goal}", ""]
        if successes:
            lines.append("Here's a successful example:")
            for meta in successes:
                lines.extend(self._format_record(meta))
                lines.append("")
        if failures:
            lines.append("Avoid these pitfalls:")
            for meta in failures:
                lines.extend(self._format_record(meta))
                lines.append("")
        if retry_trace:
            lines.append(retry_trace)
            lines.append("Please try a different approach.")
        return "\n".join(l for l in lines if l)

    # Backwards compatibility -------------------------------------------------
    @classmethod
    def construct_prompt(
        cls,
        goal: str,
        retry_trace: str | None = None,
        *,
        top_n: int = 5,
        confidence_threshold: float = 0.3,
        roi_tracker: ROITracker | None = None,
    ) -> str:
        """Class method wrapper used by existing tests and callers."""

        engine = cls(roi_tracker=roi_tracker)
        return engine.build_prompt(
            goal,
            retry_trace,
            top_n=top_n,
            confidence_threshold=confidence_threshold,
        )


def build_prompt(goal: str, retry_trace: str | None = None, *, top_n: int = 5) -> str:
    """Convenience function mirroring :meth:`PromptEngine.construct_prompt`."""

    return PromptEngine.construct_prompt(goal, retry_trace, top_n=top_n)


__all__ = ["PromptEngine", "build_prompt", "DEFAULT_TEMPLATE"]
