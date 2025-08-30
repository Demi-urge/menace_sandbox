"""Utilities to assemble prompts for self-coding helpers.

This module queries the vector service for past patches, compresses the
returned metadata and builds a structured prompt that guides the language
model towards the current enhancement goal.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:  # pragma: no cover - optional at runtime
    from .roi_tracker import ROITracker  # type: ignore
except Exception:  # pragma: no cover - fallback when running in isolation
    ROITracker = None  # type: ignore

from snippet_compressor import compress_snippets

try:  # pragma: no cover - optional dependency
    from audit_logger import log_event as audit_log_event  # type: ignore
except Exception:  # pragma: no cover - logging only when available
    def audit_log_event(*_a: Any, **_k: Any) -> None:  # type: ignore
        """Stub when audit logger is unavailable."""
        return

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
    confidence_threshold: float = 0.3

    # ------------------------------------------------------------------
    @staticmethod
    def _fetch_patches(goal: str, top_n: int) -> Tuple[List[Dict[str, Any]], float]:
        """Return ``(records, avg_confidence)`` for ``goal``.

        The default implementation queries :class:`PatchRetriever` from the
        vector service.  The confidence score is the average similarity of the
        retrieved snippets normalised to the ``[0, 1]`` range.  Test suites
        monkey patch this helper to isolate the prompt construction logic.
        """

        try:  # pragma: no cover - heavy dependency optional in tests
            from .vector_service.retriever import PatchRetriever  # type: ignore
        except Exception as exc:  # pragma: no cover - vector service unavailable
            raise RuntimeError("retriever unavailable") from exc

        try:
            retriever = PatchRetriever(top_k=top_n)
            records = retriever.search(goal, top_k=top_n) or []
        except Exception as exc:  # pragma: no cover - retrieval failure
            raise RuntimeError("retriever search failed") from exc

        total = 0.0
        for rec in records:
            score = rec.get("score") or rec.get("similarity") or 0.0
            total += float(score)
        avg_confidence = total / len(records) if records else 0.0
        return records, avg_confidence

    # ------------------------------------------------------------------
    @staticmethod
    def _compress(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Return a condensed copy of ``meta`` using micro-model helpers."""

        out: Dict[str, Any] = {}
        out["summary"] = meta.get("summary") or meta.get("description")

        out.update(compress_snippets(meta))

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
    ) -> str:
        """Assemble a prompt for ``goal``.

        When the average retrieval confidence falls below ``confidence_threshold``
        or snippet retrieval fails, the function falls back to a static template
        instead of emitting an unhelpful prompt.
        """

        try:
            records, confidence = self._fetch_patches(goal, top_n)
        except Exception as exc:
            logging.exception("Snippet retrieval failed: %s", exc)
            audit_log_event(
                "prompt_engine_fallback",
                {"goal": goal, "reason": "retrieval_error", "error": str(exc)},
            )
            return self._static_prompt()

        if not records or confidence < self.confidence_threshold:
            logging.info(
                "Retrieval confidence %.2f below threshold; falling back", confidence
            )
            audit_log_event(
                "prompt_engine_fallback",
                {
                    "goal": goal,
                    "reason": "low_confidence",
                    "confidence": confidence,
                },
            )
            return self._static_prompt()

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
            lines.append(
                f"Previous attempt failed with {retry_trace}; seek alternative solution."
            )
        return "\n".join(line for line in lines if line)

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

        engine = cls(
            roi_tracker=roi_tracker, confidence_threshold=confidence_threshold
        )
        return engine.build_prompt(goal, retry_trace, top_n=top_n)


    # ------------------------------------------------------------------
    @staticmethod
    def _static_prompt() -> str:
        """Return a generic prompt template used when retrieval fails."""

        try:  # pragma: no cover - optional heavy dependency
            from bot_development_bot import DEFAULT_TEMPLATE as BOT_DEV_TEMPLATE
            return Path(BOT_DEV_TEMPLATE).read_text(encoding="utf-8")
        except Exception:
            return DEFAULT_TEMPLATE


def build_prompt(goal: str, retry_trace: str | None = None, *, top_n: int = 5) -> str:
    """Convenience function mirroring :meth:`PromptEngine.construct_prompt`."""

    return PromptEngine.construct_prompt(goal, retry_trace, top_n=top_n)


__all__ = ["PromptEngine", "build_prompt", "DEFAULT_TEMPLATE"]
