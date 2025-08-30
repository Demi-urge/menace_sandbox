"""Utilities to assemble prompts for self-coding helpers.

This module now relies on lightweight service objects that can be injected
from the outside.  ``Retriever`` is used to fetch historical patches while
``ContextBuilder`` exposes optional helpers such as ROI tracking.  The
``PromptEngine`` orchestrates snippet retrieval, compression and ranking to
produce a compact prompt for the language model.  When retrieval fails or the
average confidence falls below a configurable threshold a static fallback
template is returned instead of an unhelpful prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

from snippet_compressor import compress_snippets

try:  # pragma: no cover - optional dependency at runtime
    from audit_logger import log_event as audit_log_event  # type: ignore
except Exception:  # pragma: no cover - logging only when available

    def audit_log_event(*_a: Any, **_k: Any) -> None:  # type: ignore
        """Fallback stub used when the audit logger is unavailable."""
        return


DEFAULT_TEMPLATE = "No relevant patches were found. Proceed with a fresh implementation."


try:  # pragma: no cover - optional heavy imports for type checking
    from vector_service.retriever import Retriever  # type: ignore
    from vector_service.context_builder import ContextBuilder  # type: ignore
    from vector_service.roi_tags import RoiTag  # type: ignore
except Exception:  # pragma: no cover - allow running tests without service layer
    Retriever = Any  # type: ignore
    ContextBuilder = Any  # type: ignore

    class RoiTag:  # type: ignore[misc]
        SUCCESS = "success"
        HIGH_ROI = "high-ROI"
        LOW_ROI = "low-ROI"
        BUG_INTRODUCED = "bug-introduced"
        NEEDS_REVIEW = "needs-review"
        BLOCKED = "blocked"

        @classmethod
        def validate(cls, value: Any) -> "RoiTag":
            return cls.SUCCESS


@dataclass
class PromptEngine:
    """Build prompts for the :class:`SelfCodingEngine`.

    Parameters
    ----------
    retriever:
        Object exposing a ``search(query, top_k)`` method returning patch
        records.  The default attempts to instantiate
        :class:`vector_service.retriever.Retriever`.
    context_builder:
        Optional :class:`vector_service.context_builder.ContextBuilder`
        instance.  When supplied the engine will attempt to read the
        ``roi_tracker`` attribute for risk adjusted ROI calculations.
    roi_tracker:
        Optional ROI tracker overriding the tracker from ``context_builder``.
    """

    retriever: Retriever | None = None
    context_builder: ContextBuilder | None = None
    roi_tracker: Any | None = None
    confidence_threshold: float = 0.3
    top_n: int = 5
    roi_weight: float = 1.0
    recency_weight: float = 0.1
    roi_tag_weights: Dict[str, float] = field(
        default_factory=lambda: {
            RoiTag.HIGH_ROI.value: 1.0,
            RoiTag.SUCCESS.value: 0.5,
            RoiTag.LOW_ROI.value: -0.5,
            RoiTag.NEEDS_REVIEW.value: -0.5,
            RoiTag.BUG_INTRODUCED.value: -1.0,
            RoiTag.BLOCKED.value: -1.0,
        }
    )

    def __post_init__(self) -> None:  # pragma: no cover - lightweight setup
        if self.retriever is None:
            try:
                self.retriever = Retriever()
            except Exception:
                self.retriever = None
        if self.context_builder is None:
            try:
                self.context_builder = ContextBuilder(retriever=self.retriever)
            except Exception:
                self.context_builder = None
        if self.roi_tracker is None and self.context_builder is not None:
            self.roi_tracker = getattr(self.context_builder, "roi_tracker", None)

    # ------------------------------------------------------------------
    def build_snippets(self, patches: Iterable[Dict[str, Any]]) -> List[str]:
        """Return formatted snippet lines for pre-ranked *patches*.

        Each element in *patches* is expected to be a mapping containing a
        ``metadata`` field.  The metadata is compressed via
        :func:`snippet_compressor.compress_snippets` and grouped into
        successful or failed examples.  The order of *patches* is preserved so
        callers can provide their own ranking.
        """

        successes: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        for record in patches:
            meta = self._compress(record.get("metadata", {}))
            if meta.get("tests_passed"):
                successes.append(meta)
            else:
                failures.append(meta)

        lines: List[str] = []
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
        return [line for line in lines if line]
    # ------------------------------------------------------------------
    def build_prompt(self, task: str, retry_info: str | None = None) -> str:
        """Return a prompt for *task* using retrieved patch examples.

        ``retry_info`` is appended to the prompt when provided.  When retrieval
        fails or the average confidence of returned patches falls below
        ``confidence_threshold`` a static fallback template is returned.
        """

        if self.retriever is None:
            logging.info("No retriever available; falling back to static template")
            return self._static_prompt()

        try:
            result = self.retriever.search(task, top_k=self.top_n)
        except Exception as exc:  # pragma: no cover - best effort
            logging.exception("Snippet retrieval failed: %s", exc)
            audit_log_event(
                "prompt_engine_fallback",
                {"goal": task, "reason": "retrieval_error", "error": str(exc)},
            )
            return self._static_prompt()

        confidence: float
        if isinstance(result, tuple):
            records, confidence = result  # type: ignore[assignment]
        else:
            records = result
            total = 0.0
            for rec in records:
                total += float(rec.get("score") or rec.get("similarity") or 0.0)
            confidence = total / len(records) if records else 0.0

        ranked = self._rank_records(records)
        if not ranked or confidence < self.confidence_threshold:
            logging.info(
                "Retrieval confidence %.2f below threshold; falling back", confidence
            )
            audit_log_event(
                "prompt_engine_fallback",
                {
                    "goal": task,
                    "reason": "low_confidence",
                    "confidence": confidence,
                },
            )
            return self._static_prompt()

        lines = [f"Enhancement goal: {task}", ""]
        lines.extend(self.build_snippets(ranked))
        if retry_info:
            lines.append(
                f"Previous attempt failed with {retry_info}; seek alternative solution."
            )
        return "\n".join(line for line in lines if line)

    # ------------------------------------------------------------------
    def _rank_records(self, records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return ``records`` ordered by ROI tag and recency."""

        items = list(records)
        if not items:
            return []
        ts_vals = [float(r.get("metadata", {}).get("ts") or 0.0) for r in items]
        min_ts = min(ts_vals) if ts_vals else 0.0
        max_ts = max(ts_vals) if ts_vals else 0.0
        span = max(max_ts - min_ts, 1.0)

        def score(rec: Dict[str, Any]) -> float:
            meta = rec.get("metadata", {})
            tag_val = meta.get("roi_tag") or rec.get("roi_tag")
            tag = RoiTag.validate(tag_val)
            roi_score = self.roi_tag_weights.get(tag.value, 0.0)
            ts = float(meta.get("ts") or 0.0)
            ts_score = (ts - min_ts) / span
            return self.roi_weight * roi_score + self.recency_weight * ts_score

        items.sort(key=score, reverse=True)
        return items[: self.top_n]

    # ------------------------------------------------------------------
    def _compress(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Return a condensed copy of ``meta`` using micro-model helpers."""

        out: Dict[str, Any] = {}
        out["summary"] = meta.get("summary") or meta.get("description")

        out.update(compress_snippets(meta))

        out["outcome"] = meta.get("outcome")
        out["tests_passed"] = meta.get("tests_passed")
        out["roi_delta"] = meta.get("roi_delta")
        out["roi_tag"] = meta.get("roi_tag")
        out["ts"] = meta.get("ts")
        out["context"] = meta.get("context") or meta.get("retrieval_context")
        return out

    # ------------------------------------------------------------------
    def _format_record(self, meta: Dict[str, Any]) -> List[str]:
        """Return human readable lines for a single patch ``meta``."""

        if self.roi_tracker and meta.get("roi_delta") is not None:
            try:  # pragma: no cover - best effort integration
                _, raroi, _ = self.roi_tracker.calculate_raroi(
                    float(meta["roi_delta"])
                )
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
        if meta.get("roi_tag"):
            lines.append(f"ROI tag: {meta['roi_tag']}")
        if meta.get("context"):
            lines.append(f"Context: {meta['context']}")
        return lines

    # ------------------------------------------------------------------
    @staticmethod
    def _static_prompt() -> str:
        """Return a generic prompt template used when retrieval fails."""

        try:  # pragma: no cover - optional heavy dependency
            from bot_development_bot import (
                DEFAULT_TEMPLATE as BOT_DEV_TEMPLATE,  # type: ignore
            )
            return Path(BOT_DEV_TEMPLATE).read_text(encoding="utf-8")
        except Exception:
            return DEFAULT_TEMPLATE

    # Backwards compatibility -------------------------------------------------
    @classmethod
    def construct_prompt(
        cls,
        goal: str,
        retry_trace: str | None = None,
        *,
        top_n: int = 5,
        confidence_threshold: float = 0.3,
        retriever: Retriever | None = None,
        context_builder: ContextBuilder | None = None,
        roi_tracker: Any | None = None,
    ) -> str:
        """Class method wrapper used by existing callers and tests."""

        engine = cls(
            retriever=retriever,
            context_builder=context_builder,
            roi_tracker=roi_tracker,
            confidence_threshold=confidence_threshold,
            top_n=top_n,
        )
        return engine.build_prompt(goal, retry_trace)


def build_prompt(goal: str, retry_trace: str | None = None, *, top_n: int = 5) -> str:
    """Convenience function mirroring :meth:`PromptEngine.construct_prompt`."""

    return PromptEngine.construct_prompt(goal, retry_trace, top_n=top_n)


__all__ = ["PromptEngine", "build_prompt", "DEFAULT_TEMPLATE"]
