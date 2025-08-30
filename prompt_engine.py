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
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from snippet_compressor import compress_snippets

try:  # pragma: no cover - optional dependency at runtime
    from audit_logger import log_event as audit_log_event  # type: ignore
except Exception:  # pragma: no cover - logging only when available

    def audit_log_event(*_a: Any, **_k: Any) -> None:  # type: ignore
        """Fallback stub used when the audit logger is unavailable."""
        return


logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional precise tokenizer
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - dependency missing or failed
    tiktoken = None  # type: ignore
    _ENCODER = None


DEFAULT_TEMPLATE = "No relevant patches were found. Proceed with a fresh implementation."


try:  # pragma: no cover - optional heavy imports for type checking
    from vector_service.retriever import (
        Retriever,
        PatchRetriever,
        FallbackResult,
    )  # type: ignore
    from vector_service.context_builder import ContextBuilder  # type: ignore
    from vector_service.roi_tags import RoiTag  # type: ignore
except Exception:  # pragma: no cover - allow running tests without service layer
    Retriever = Any  # type: ignore
    PatchRetriever = Any  # type: ignore
    ContextBuilder = Any  # type: ignore

    class FallbackResult:  # type: ignore[too-many-ancestors]
        reason = ""
        confidence = 0.0

        def __iter__(self):
            return iter([])

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
    patch_retriever:
        Optional :class:`vector_service.retriever.PatchRetriever` instance.
        When omitted the *retriever* is used for patch lookups.
    context_builder:
        Optional :class:`vector_service.context_builder.ContextBuilder`
        instance.  When supplied the engine will attempt to read the
        ``roi_tracker`` attribute for risk adjusted ROI calculations.
    roi_tracker:
        Optional ROI tracker overriding the tracker from ``context_builder``.
    template_path:
        Path to a JSON file containing static prompt templates used when
        retrieval confidence is too low.
    template_sections:
        Names of sections from ``template_path`` to include in fallback
        prompts.
    max_tokens:
        Maximum number of tokens allowed for each snippet block.
    success_header:
        Header inserted before successful examples. Defaults to
        ``"Successful example:"``.
    failure_header:
        Header inserted before failed examples. Defaults to
        ``"Avoid pattern:"``.
    """

    retriever: Retriever | None = None
    patch_retriever: PatchRetriever | None = None
    context_builder: ContextBuilder | None = None
    roi_tracker: Any | None = None
    confidence_threshold: float = 0.3
    top_n: int = 5
    max_tokens: int = 200
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
    template_path: Path = Path(
        os.getenv(
            "PROMPT_TEMPLATES_PATH",
            Path(__file__).resolve().parent / "config" / "prompt_templates.v1.json",
        )
    )
    template_sections: List[str] = field(
        default_factory=lambda: os.getenv(
            "PROMPT_TEMPLATE_SECTIONS",
            "coding_standards;repository_layout;metadata;version_control;testing",
        ).split(";"),
    )
    success_header: str = "Successful example:"
    failure_header: str = "Avoid pattern:"

    def __post_init__(self) -> None:  # pragma: no cover - lightweight setup
        if self.retriever is None:
            try:
                self.retriever = Retriever()
            except Exception:
                self.retriever = None
        if self.patch_retriever is None:
            if self.retriever is not None:
                self.patch_retriever = self.retriever  # type: ignore[assignment]
            else:
                try:
                    self.patch_retriever = PatchRetriever()
                except Exception:
                    self.patch_retriever = None
        if self.context_builder is None:
            try:
                self.context_builder = ContextBuilder(retriever=self.retriever)
            except Exception:
                self.context_builder = None
        if self.roi_tracker is None and self.context_builder is not None:
            self.roi_tracker = getattr(self.context_builder, "roi_tracker", None)

    # ------------------------------------------------------------------
    def build_snippets(self, patches: Iterable[Dict[str, Any]]) -> List[str]:
        """Return formatted snippet lines sorted by ROI tag and age.

        Each element in *patches* is expected to be a mapping containing a
        ``metadata`` field.  The metadata is compressed via
        :func:`snippet_compressor.compress_snippets` and grouped into
        successful or failed examples.  The groups are sorted by the internal
        :meth:`_score_snippet` helper so callers receive the most relevant
        examples first.
        """

        records = list(patches)
        metas = [r.get("metadata", {}) for r in records]
        ts_vals = [float(m.get("ts") or 0.0) for m in metas]
        min_ts = min(ts_vals) if ts_vals else 0.0
        max_ts = max(ts_vals) if ts_vals else 0.0
        span = max(max_ts - min_ts, 1.0)

        successes: List[tuple[float, str]] = []
        failures: List[tuple[float, str]] = []
        for rec, meta in zip(records, metas):
            score = rec.get("weighted_score")
            if score is None:
                score = self._score_snippet(meta, min_ts=min_ts, span=span)
            if score < self.confidence_threshold:
                continue
            snippet, passed = self._compress_patch(meta, max_tokens=self.max_tokens)
            if not snippet:
                continue
            if passed:
                successes.append((score, snippet))
            else:
                failures.append((score, snippet))

        successes.sort(key=lambda x: x[0], reverse=True)
        failures.sort(key=lambda x: x[0], reverse=True)

        lines: List[str] = []
        if successes:
            lines.append(self.success_header)
            for _, text in successes:
                lines.extend(text.splitlines())
                lines.append("")
        if failures:
            lines.append(self.failure_header)
            for _, text in failures:
                lines.extend(text.splitlines())
                lines.append("")
        return [line for line in lines if line]

    # ------------------------------------------------------------------
    def build_prompt(
        self,
        task: str,
        *,
        context: str | None = None,
        retrieval_context: str | None = None,
        retry_trace: str | None = None,
    ) -> str:
        """Return a prompt for *task* using retrieved patch examples.

        ``context`` and ``retrieval_context`` allow callers to prepend
        additional information such as the snippet body, repository layout or
        metadata from vector retrieval.  ``retry_trace`` may contain failure
        logs or tracebacks from a prior attempt.  When supplied a "Previous
        failure" section is appended and the details are de-duplicated so
        repeated retries do not accumulate duplicate traces.  When retrieval
        fails or the average confidence of returned patches falls below
        ``confidence_threshold`` a static fallback template is returned.
        """

        retriever = self.patch_retriever or self.retriever
        if retriever is None:
            logging.info("No retriever available; falling back to static template")
            return self._static_prompt()

        try:
            result = retriever.search(task, top_k=self.top_n)
        except Exception as exc:  # pragma: no cover - best effort
            logging.exception("Snippet retrieval failed: %s", exc)
            audit_log_event(
                "prompt_engine_fallback",
                {"goal": task, "reason": "retrieval_error", "error": str(exc)},
            )
            return self._static_prompt()

        if isinstance(result, FallbackResult):
            logging.info(
                "Retriever returned fallback (%s); using static template",
                result.reason,
            )
            audit_log_event(
                "prompt_engine_fallback",
                {
                    "goal": task,
                    "reason": result.reason,
                    "confidence": result.confidence,
                },
            )
            return self._static_prompt()

        if isinstance(result, tuple):
            records = result[0]
        else:
            records = result
        scores = [float(rec.get("score") or 0.0) for rec in records]
        confidence = sum(scores) / len(scores) if scores else 0.0

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

        lines: List[str] = []
        if retrieval_context:
            lines.append(retrieval_context.strip())
            lines.append("")
        if context:
            lines.append(context.strip())
            lines.append("")
        lines.append(f"Enhancement goal: {task}")
        lines.append("")
        lines.extend(self.build_snippets(ranked))
        if retry_trace:
            lines.extend(self._format_retry_trace(retry_trace))
        return "\n".join(line for line in lines if line)

    # ------------------------------------------------------------------
    def _format_retry_trace(self, retry_trace: str) -> List[str]:
        """Return formatted ``retry_trace`` lines without duplicates.

        The helper removes any existing "Previous failure" headers or concluding
        guidance so that repeated invocations remain idempotent.
        """

        skip_prefix = "previous failure:"
        skip_suffixes = {"please attempt a different solution."}
        cleaned: List[str] = []
        for line in retry_trace.strip().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if lower.startswith(skip_prefix):
                continue
            if lower in skip_suffixes:
                continue
            cleaned.append(line)
        if not cleaned:
            return []
        return [
            "Previous failure:",
            *cleaned,
            "Please attempt a different solution.",
        ]

    # ------------------------------------------------------------------
    def _rank_records(self, records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return ``records`` ordered by the weighted scoring heuristic."""

        items = list(records)
        if not items:
            return []
        ts_vals = [float(r.get("metadata", {}).get("ts") or 0.0) for r in items]
        min_ts = min(ts_vals) if ts_vals else 0.0
        max_ts = max(ts_vals) if ts_vals else 0.0
        span = max(max_ts - min_ts, 1.0)

        scored: List[tuple[float, Dict[str, Any]]] = []
        for rec in items:
            meta = rec.get("metadata", {})
            score = self._score_snippet(meta, min_ts=min_ts, span=span)
            if score < self.confidence_threshold:
                continue
            rec["weighted_score"] = score
            scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in scored[: self.top_n]]

    # ------------------------------------------------------------------
    def _score_snippet(
        self, meta: Dict[str, Any], *, min_ts: float = 0.0, span: float = 1.0
    ) -> float:
        """Return a weighted ranking score for a snippet ``meta``.

        The score combines three components:

        * ``raroi`` (risk-adjusted ROI) scaled by :attr:`roi_weight`.
        * Recency normalised to ``[0, 1]`` and scaled by :attr:`recency_weight`.
        * An adjustment based on ``roi_tag`` from :attr:`roi_tag_weights`.

        When ``raroi`` is absent but ``roi_delta`` is available and an ROI
        tracker is configured, the value is converted using
        ``roi_tracker.calculate_raroi``.  Missing components simply contribute
        ``0`` to the final score.
        """

        roi_val = meta.get("raroi")
        if roi_val is None and meta.get("roi_delta") is not None:
            try:  # pragma: no cover - best effort
                _, roi_val, _ = (
                    self.roi_tracker.calculate_raroi(float(meta["roi_delta"]))
                    if self.roi_tracker
                    else (None, None, None)
                )
            except Exception:
                roi_val = None

        try:
            raroi = float(roi_val) if roi_val is not None else 0.0
        except Exception:  # pragma: no cover - defensive
            raroi = 0.0

        ts = float(meta.get("ts") or 0.0)
        normalised_ts = (ts - min_ts) / span if span else 0.0

        score = (
            self.roi_weight * raroi + self.recency_weight * normalised_ts
        )

        tag = meta.get("roi_tag")
        score += float(self.roi_tag_weights.get(tag, 0.0))

        return score

    # ------------------------------------------------------------------
    def _trim_tokens(self, text: str, limit: int) -> str:
        """Trim ``text`` to ``limit`` tokens using available tokenizers."""

        counter = None
        if self.context_builder is not None:
            counter = getattr(self.context_builder, "_count_tokens", None)
        if counter is None:
            if _ENCODER is not None:
                def _token_counter(s: Any) -> int:
                    return len(_ENCODER.encode(str(s)))

                counter = _token_counter
            else:
                logger.warning(
                    "precise token counting requires the 'tiktoken' package"
                )

                def _default_counter(s: Any) -> int:
                    return len(str(s).split())

                counter = _default_counter

        tokens = counter(text)
        if tokens <= limit:
            return text

        ratio = limit / max(tokens, 1)
        char_limit = max(1, int(len(text) * ratio))
        trimmed = text[:char_limit].rstrip()
        if trimmed != text:
            trimmed += "..."
        return trimmed

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
    def _compress_patch(
        self, patch: Dict[str, Any], *, max_tokens: int | None = None
    ) -> tuple[str, bool]:
        """Return a compact textual representation of ``patch``.

        The function extracts description, diff, code snippet and test logs and
        trims the final snippet to ``max_tokens`` tokens using the
        :class:`ContextBuilder`'s token counting utilities when available.
        ``True`` is returned alongside the snippet when tests passed for the
        patch.
        """

        limit = self.max_tokens if max_tokens is None else max_tokens
        meta = self._compress(patch)
        lines = self._format_record(meta)
        snippet = "\n".join(lines)
        snippet = self._trim_tokens(snippet, limit)
        return snippet, bool(meta.get("tests_passed"))

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
        if meta.get("snippet"):
            lines.append(f"Code:\n{meta['snippet']}")
        if meta.get("test_log"):
            lines.append(f"Test log:\n{meta['test_log']}")
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
    def _static_prompt(self) -> str:
        """Return a generic prompt assembled from static templates."""

        try:
            if self.template_path.suffix in {".yaml", ".yml"}:
                import yaml  # type: ignore

                with open(self.template_path, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                tmpl = data.get("default_template") or data.get("templates", {})
                lines: List[str] = []
                if isinstance(tmpl, dict) and self.template_sections:
                    for section in self.template_sections:
                        lines.extend(tmpl.get(section, []))
                elif isinstance(tmpl, list):
                    lines = [str(x) for x in tmpl]
                elif isinstance(tmpl, str):
                    lines = [tmpl]
            else:
                with open(self.template_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                templates = data.get("templates", data)
                lines = []
                for section in self.template_sections:
                    lines.extend(templates.get(section, []))
            return "\n".join(lines) if lines else DEFAULT_TEMPLATE
        except Exception:
            return DEFAULT_TEMPLATE

    # Backwards compatibility -------------------------------------------------
    @classmethod
    def construct_prompt(
        cls,
        goal: str,
        retry_trace: str | None = None,
        *,
        context: str | None = None,
        retrieval_context: str | None = None,
        top_n: int = 5,
        confidence_threshold: float = 0.3,
        retriever: Retriever | None = None,
        context_builder: ContextBuilder | None = None,
        roi_tracker: Any | None = None,
        success_header: str = "Successful example:",
        failure_header: str = "Avoid pattern:",
    ) -> str:
        """Class method wrapper used by existing callers and tests."""

        engine = cls(
            retriever=retriever,
            context_builder=context_builder,
            roi_tracker=roi_tracker,
            confidence_threshold=confidence_threshold,
            top_n=top_n,
            success_header=success_header,
            failure_header=failure_header,
        )
        return engine.build_prompt(
            goal,
            context=context,
            retrieval_context=retrieval_context,
            retry_trace=retry_trace,
        )


def build_prompt(
    goal: str,
    retry_trace: str | None = None,
    *,
    context: str | None = None,
    retrieval_context: str | None = None,
    top_n: int = 5,
    success_header: str = "Successful example:",
    failure_header: str = "Avoid pattern:",
) -> str:
    """Convenience wrapper mirroring :meth:`PromptEngine.construct_prompt`."""

    return PromptEngine.construct_prompt(
        goal,
        retry_trace,
        context=context,
        retrieval_context=retrieval_context,
        top_n=top_n,
        success_header=success_header,
        failure_header=failure_header,
    )


__all__ = ["PromptEngine", "build_prompt", "DEFAULT_TEMPLATE"]
