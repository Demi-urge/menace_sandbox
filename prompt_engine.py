"""Utilities to assemble prompts for self-coding helpers.

This module now relies on lightweight service objects that can be injected
from the outside.  ``Retriever`` is used to fetch historical patches while
``ContextBuilder`` exposes optional helpers such as ROI tracking.  The
``PromptEngine`` orchestrates snippet retrieval, compression and ranking to
produce a compact :class:`llm_interface.Prompt` object for the language model.
When retrieval fails or the average confidence falls below a configurable
threshold a static fallback template is returned instead of an unhelpful
prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dynamic_path_router import resolve_path, path_for_prompt

from llm_interface import Prompt, LLMClient
from snippet_compressor import compress_snippets
from chunking import split_into_chunks, summarize_code
from target_region import TargetRegion, extract_target_region
from billing.prompt_notice import prepend_payment_notice
try:  # pragma: no cover - optional billing integration
    import stripe_billing_router  # noqa: F401
except Exception:  # pragma: no cover - ignore missing dependencies
    stripe_billing_router = None  # type: ignore


SYSTEM_NOTICE = prepend_payment_notice([])[0]["content"]

try:  # pragma: no cover - optional settings dependency
    from sandbox_settings import SandboxSettings  # type: ignore
except Exception:  # pragma: no cover - allow running without settings
    SandboxSettings = None  # type: ignore

_SETTINGS = SandboxSettings() if SandboxSettings else None

try:  # pragma: no cover - optional runtime dependency
    from prompt_optimizer import (
        PromptOptimizer,
        select_format as _select_optimizer_format,
    )  # type: ignore
except Exception:  # pragma: no cover - degrade gracefully
    PromptOptimizer = None  # type: ignore

    def _select_optimizer_format(*_a: Any, **_k: Any) -> Dict[str, Any]:  # type: ignore
        return {}

try:  # pragma: no cover - optional runtime dependency
    from prompt_memory_trainer import PromptMemoryTrainer  # type: ignore
except Exception:  # pragma: no cover - degrade gracefully
    PromptMemoryTrainer = None  # type: ignore

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

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - degrade gracefully when unavailable
    yaml = None  # type: ignore


DEFAULT_TEMPLATE = "No relevant patches were found. Proceed with a fresh implementation."

# Strategy templates live in ``templates/prompt_strategies.yaml``.  The helper
# falls back to an empty mapping when the file is missing or cannot be parsed.
_STRATEGY_TEMPLATE_PATH = resolve_path("templates/prompt_strategies.yaml")
_STRATEGY_TEMPLATES: Dict[str, str] | None = None


def _load_strategy_templates() -> Dict[str, str]:
    """Return cached strategy templates loaded from ``prompt_strategies.yaml``."""

    global _STRATEGY_TEMPLATES
    if _STRATEGY_TEMPLATES is None:
        templates: Dict[str, str] = {}
        if yaml is not None:
            try:  # pragma: no cover - best effort
                if _STRATEGY_TEMPLATE_PATH.exists():
                    data = yaml.safe_load(
                        _STRATEGY_TEMPLATE_PATH.read_text(encoding="utf-8")
                    )
                    if isinstance(data, dict):
                        # Support both a top-level mapping and the legacy
                        # structure using a ``templates`` key.
                        raw = data.get("templates", data)
                        if isinstance(raw, dict):
                            templates = {
                                str(k): str(v)
                                for k, v in raw.items()
                                if isinstance(k, str)
                            }
            except Exception:
                templates = {}
        _STRATEGY_TEMPLATES = templates
    return _STRATEGY_TEMPLATES


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


def diff_within_target_region(
    original: List[str], modified: List[str], region: TargetRegion
) -> bool:
    """Return ``True`` if changes between ``original`` and ``modified`` stay inside ``region``.

    ``original`` and ``modified`` are lists of file lines.  The check iterates
    over both sequences in lock step and ensures that any changed line falls
    within the ``TargetRegion`` boundaries.
    """

    for idx, (a, b) in enumerate(zip_longest(original, modified)):
        if (a or "") != (b or ""):
            line_no = idx + 1
            if line_no < region.start_line or line_no > region.end_line:
                return False
    return True


@dataclass
class PromptEngine:
    """Build prompts for the :class:`SelfCodingEngine`.

    Retrieved patches are ranked by a weighted heuristic combining
    risk-adjusted ROI, recency and ROI tag weights.  Snippets are then
    grouped into successful or failed examples before being assembled into
    the final prompt.  When no snippet clears ``confidence_threshold`` the
    engine falls back to a static template instead of emitting a low
    confidence prompt.

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
        :class:`vector_service.context_builder.ContextBuilder` instance used
        for token counting and ROI tracking. The engine relies on this object
        and does not create one automatically.
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
    token_threshold:
        Maximum number of tokens permitted for the assembled prompt. Text
        exceeding this limit is trimmed and long files are summarised when
        included as context.
    success_header:
        Header inserted before successful examples. Defaults to
        ``"Given the following pattern:"``.
    failure_header:
        Template for headers inserted before failed examples. Supports
        ``{summary}`` and ``{outcome}`` placeholders and defaults to
        ``"Avoid {summary} because it caused {outcome}:"``.
    optimizer:
        Optional :class:`prompt_optimizer.PromptOptimizer` used to suggest
        tone and structural preferences.
    optimizer_refresh_interval:
        Refresh optimiser statistics after this many prompts when set.
    """

    context_builder: ContextBuilder
    retriever: Retriever | None = None
    patch_retriever: PatchRetriever | None = None
    roi_tracker: Any | None = None
    confidence_threshold: float = 0.3
    top_n: int = 5
    max_tokens: int = 200
    token_threshold: int = 3500
    chunk_token_threshold: int = field(
        default_factory=lambda: _SETTINGS.prompt_chunk_token_threshold
        if _SETTINGS
        else 3500
    )
    chunk_summary_cache_dir: Path = field(
        default_factory=lambda: resolve_path(_SETTINGS.chunk_summary_cache_dir)
        if _SETTINGS
        else resolve_path("chunk_summary_cache")
    )
    llm: LLMClient | None = None
    roi_weight: float = 1.0
    recency_weight: float = 0.1
    roi_tag_weights: Dict[str, float] = field(
        default_factory=lambda: {
            getattr(RoiTag.HIGH_ROI, "value", RoiTag.HIGH_ROI): 1.0,
            getattr(RoiTag.SUCCESS, "value", RoiTag.SUCCESS): 0.5,
            getattr(RoiTag.LOW_ROI, "value", RoiTag.LOW_ROI): -0.5,
            getattr(RoiTag.NEEDS_REVIEW, "value", RoiTag.NEEDS_REVIEW): -0.5,
            getattr(RoiTag.BUG_INTRODUCED, "value", RoiTag.BUG_INTRODUCED): -1.0,
            getattr(RoiTag.BLOCKED, "value", RoiTag.BLOCKED): -1.0,
        }
    )
    template_path: Path = resolve_path(
        os.getenv(
            "PROMPT_TEMPLATES_PATH",
            "config/prompt_templates.v2.json",
        )
    )
    template_sections: List[str] = field(
        default_factory=lambda: os.getenv(
            "PROMPT_TEMPLATE_SECTIONS",
            "coding_standards;repository_layout;metadata;version_control;testing",
        ).split(";"),
    )
    weights_path: Path = resolve_path(
        os.getenv(
            "PROMPT_STYLE_WEIGHTS_PATH",
            "prompt_style_weights.json",
        )
    )
    trainer: PromptMemoryTrainer | None = None
    optimizer: PromptOptimizer | None = None
    optimizer_refresh_interval: int | None = None
    success_header: str = "Given the following pattern:"
    failure_header: str = "Avoid {summary} because it caused {outcome}:"
    tone: str = "neutral"
    last_metadata: Dict[str, Any] = field(default_factory=dict, init=False)
    trained_headers: List[str] | None = field(default=None, init=False)
    trained_example_order: List[str] | None = field(default=None, init=False)
    trained_structured_sections: List[str] | None = field(default=None, init=False)
    trained_example_count: int | None = field(default=None, init=False)
    trained_example_placement: str | None = field(default=None, init=False)
    trained_length: str | None = field(default=None, init=False)
    style_version: int = field(default=0, init=False)
    _optimizer_counter: int = field(default=0, init=False)
    _optimizer_applied: bool = field(default=False, init=False)

    @property
    def prompt_chunk_cache_dir(self) -> Path:  # pragma: no cover - backward compat
        """Alias for :attr:`chunk_summary_cache_dir`.

        The prompt engine historically exposed ``prompt_chunk_cache_dir``.  The
        attribute is retained for compatibility but proxies to the new
        :attr:`chunk_summary_cache_dir` setting.
        """

        return self.chunk_summary_cache_dir

    def __post_init__(self) -> None:  # pragma: no cover - lightweight setup
        if self.context_builder is None:
            raise ValueError("PromptEngine requires a ContextBuilder instance")
        try:
            self.template_path = resolve_path(self.template_path)
        except Exception:
            pass
        try:
            self.weights_path = resolve_path(self.weights_path)
        except Exception:
            pass
        if self.retriever is None:
            try:
                self.retriever = Retriever(context_builder=self.context_builder)
            except Exception:
                self.retriever = None
        if self.patch_retriever is None:
            if self.retriever is not None:
                self.patch_retriever = self.retriever  # type: ignore[assignment]
            else:
                try:
                    self.patch_retriever = PatchRetriever(
                        context_builder=self.context_builder
                    )
                except Exception:
                    self.patch_retriever = None
        if self.roi_tracker is None:
            self.roi_tracker = getattr(self.context_builder, "roi_tracker", None)
        if self.trainer is None and PromptMemoryTrainer is not None:
            try:
                self.trainer = PromptMemoryTrainer()
            except Exception:
                self.trainer = None
        if self.optimizer is None and PromptOptimizer is not None and SandboxSettings:
            try:
                _s = SandboxSettings()
                self.optimizer = PromptOptimizer(
                    _s.prompt_success_log_path,
                    _s.prompt_failure_log_path,
                )
            except Exception:
                self.optimizer = None
        self._load_trained_config()
        # Apply persisted optimiser preferences
        self.apply_optimizer_format(__name__, "build_prompt")
        try:
            import chunking as _pc
            from chunk_summary_cache import ChunkSummaryCache

            self.chunk_summary_cache_dir = resolve_path(self.chunk_summary_cache_dir)
            cache = getattr(_pc, "CHUNK_CACHE", None)
            if not cache or getattr(cache, "cache_dir", None) != self.chunk_summary_cache_dir:
                _pc.CHUNK_CACHE = ChunkSummaryCache(self.chunk_summary_cache_dir)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _load_trained_config(
        self,
        summary: Dict[str, Dict[str, float]] | None = None,
        *,
        version: int | None = None,
    ) -> None:
        """Load formatting preferences from :class:`PromptMemoryTrainer`.

        The trainer aggregates success rates for previously used prompt
        headers and example orderings.  When the best observed configuration
        meets :attr:`confidence_threshold` it is cached for future prompts.
        Missing trainers or training data leave the default formatting
        untouched.
        """

        if PromptMemoryTrainer is None:
            return
        if summary is None:
            if self.trainer and getattr(self.trainer, "style_weights", None):
                summary = self.trainer.style_weights
                version = getattr(self.trainer, "STYLE_VERSION", 0)
            if (not summary or version is None) and self.weights_path.exists():
                try:
                    version, summary = PromptMemoryTrainer.load_weights(
                        self.weights_path
                    )  # type: ignore[attr-defined]
                except Exception:
                    audit_log_event(
                        "prompt_style_reverted",
                        {"reason": "load_failed", "path": str(self.weights_path)},
                    )
                    summary = None
            if summary is None and self.trainer is not None:
                try:  # pragma: no cover - best effort training lookup
                    summary = self.trainer.train()
                    version = getattr(self.trainer, "STYLE_VERSION", 0)
                    try:
                        self.trainer.save_weights(self.weights_path)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                except Exception:
                    return
        else:
            if version is None:
                version = getattr(PromptMemoryTrainer, "STYLE_VERSION", 0)
        if not summary:
            audit_log_event("prompt_style_reverted", {"reason": "missing_data"})
            return
        expected = getattr(PromptMemoryTrainer, "STYLE_VERSION", 0)
        if version != expected:
            audit_log_event(
                "prompt_style_reverted",
                {"reason": "incompatible_version", "expected": expected, "found": version},
            )
            return
        self.style_version = version

        # Determine preferred headers (first entry is success header)
        headers_summary = summary.get("headers", {}) or {}
        best_headers: List[str] | None = None
        best_score = 0.0
        for raw, score in headers_summary.items():
            try:
                hdrs = [str(h) for h in json.loads(raw)]
                sc = float(score or 0.0)
            except Exception:
                continue
            if sc > best_score:
                best_headers, best_score = hdrs, sc
        if best_headers and best_score >= self.confidence_threshold:
            self.trained_headers = best_headers

        # Determine preferred example order
        order_summary = summary.get("example_order", {}) or {}
        best_order: List[str] | None = None
        best_order_score = 0.0
        for raw, score in order_summary.items():
            try:
                order = [str(x) for x in json.loads(raw)]
                sc = float(score or 0.0)
            except Exception:
                continue
            if sc > best_order_score:
                best_order, best_order_score = order, sc
        if best_order and best_order_score >= self.confidence_threshold:
            self.trained_example_order = best_order

        # Preferred tone
        tone_summary = summary.get("tone", {}) or {}
        if tone_summary:
            val, score = max(tone_summary.items(), key=lambda kv: kv[1])
            try:
                sc = float(score)
            except Exception:
                sc = 0.0
            if sc >= self.confidence_threshold:
                self.tone = str(val)

        # Structured sections
        sect_summary = summary.get("structured_sections", {}) or {}
        best_sects: List[str] | None = None
        best_sect_score = 0.0
        for raw, score in sect_summary.items():
            try:
                sects = [str(s) for s in json.loads(raw)]
                sc = float(score or 0.0)
            except Exception:
                continue
            if sc > best_sect_score:
                best_sects, best_sect_score = sects, sc
        if best_sects and best_sect_score >= self.confidence_threshold:
            self.trained_structured_sections = best_sects

        # Example count
        count_summary = summary.get("example_count", {}) or {}
        if count_summary:
            val, score = max(count_summary.items(), key=lambda kv: kv[1])
            if score >= self.confidence_threshold:
                try:
                    self.trained_example_count = int(float(val))
                except Exception:
                    pass

        # Example placement
        place_summary = summary.get("example_placement", {}) or {}
        if place_summary:
            val, score = max(place_summary.items(), key=lambda kv: kv[1])
            if score >= self.confidence_threshold:
                self.trained_example_placement = val

        # Length / verbosity
        length_summary = summary.get("length", {}) or {}
        if length_summary:
            val, score = max(length_summary.items(), key=lambda kv: kv[1])
            if score >= self.confidence_threshold:
                self.trained_length = val
                if val == "short":
                    self.max_tokens = min(self.max_tokens, 100)
                elif val == "long":
                    self.max_tokens = max(self.max_tokens, 300)

    # ------------------------------------------------------------------
    def apply_optimizer_format(self, module: str, action: str) -> Dict[str, Any]:
        """Fetch optimiser preferences and apply them to the engine."""

        prefs: Dict[str, Any] = {}
        if self.optimizer and hasattr(self.optimizer, "select_format"):
            try:  # pragma: no cover - best effort
                prefs = self.optimizer.select_format(module, action)
            except Exception:
                prefs = {}
        else:
            try:  # pragma: no cover - best effort
                prefs = _select_optimizer_format()
                if prefs.get("headers") and "structured_sections" not in prefs:
                    prefs["structured_sections"] = prefs.pop("headers")
            except Exception:
                prefs = {}
        if prefs:
            self._apply_optimizer_preferences(prefs)
            self.last_metadata = {
                "tone": self.tone,
                "headers": self.trained_headers or [],
                "structured_sections": self.trained_structured_sections or [],
                "example_order": self.trained_example_order or [],
                "example_placement": self.trained_example_placement or "end",
            }
            self._optimizer_applied = True
        return prefs

    # ------------------------------------------------------------------
    def _apply_optimizer_preferences(self, prefs: Dict[str, Any]) -> None:
        """Update formatting parameters from optimiser ``prefs``.

        The mapping may contain ``tone``, ``headers`` or ``structured_sections``,
        ``example_order`` and ``example_placement`` keys. Missing keys leave the
        existing configuration untouched.
        """

        tone = prefs.get("tone")
        if isinstance(tone, str):
            self.tone = tone
        headers = prefs.get("headers")
        if isinstance(headers, list) and headers:
            self.trained_headers = [str(h) for h in headers]
        structured = prefs.get("structured_sections")
        if isinstance(structured, list) and structured:
            self.trained_structured_sections = [str(s) for s in structured]
        order = prefs.get("example_order")
        if isinstance(order, list) and order:
            self.trained_example_order = [str(o) for o in order]
        placement = prefs.get("example_placement")
        if isinstance(placement, str):
            self.trained_example_placement = placement

    # ------------------------------------------------------------------
    def after_patch_cycle(self) -> None:
        """Backward compatible wrapper for :meth:`refresh_trained_config`."""

        self.refresh_trained_config()

    # ------------------------------------------------------------------
    def refresh_trained_config(self) -> None:
        """Re-run trainer and update cached formatting preferences."""

        if not self.trainer:
            return
        try:  # pragma: no cover - best effort retraining
            summary = self.trainer.train()
        except Exception:
            return
        try:
            self.trainer.save_weights(self.weights_path)  # type: ignore[attr-defined]
        except Exception:
            pass
        self._load_trained_config(
            summary, version=getattr(self.trainer, "STYLE_VERSION", 0)
        )
        self.apply_optimizer_format(__name__, "build_prompt")

    # ------------------------------------------------------------------
    def refresh_optimizer(self) -> None:
        """Recompute optimiser statistics if available."""

        if not self.optimizer:
            return
        func = getattr(self.optimizer, "refresh", None)
        if func is None:
            func = getattr(self.optimizer, "aggregate", None)
        if not func:
            return
        try:  # pragma: no cover - best effort
            func()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def after_log_append(self) -> None:
        """Hook invoked after new log entries are written."""

        self.refresh_optimizer()

    # ------------------------------------------------------------------
    def _maybe_refresh_optimizer(self) -> None:
        """Refresh optimiser based on the configured interval."""

        if not self.optimizer or self.optimizer_refresh_interval is None:
            return
        self._optimizer_counter += 1
        if self._optimizer_counter >= self.optimizer_refresh_interval:
            self.refresh_optimizer()
            self._optimizer_counter = 0

    # ------------------------------------------------------------------
    def build_snippets(self, patches: Iterable[Dict[str, Any]]) -> List[str]:
        """Return formatted snippet lines ordered by weighted scoring.

        Each entry in *patches* must provide a ``metadata`` mapping.  The
        metadata is compressed via :func:`snippet_compressor.compress_snippets`
        before snippets are grouped into successful and failed examples.  The
        groups are sorted by :meth:`_score_snippet`, which combines
        risk-adjusted ROI, recency and ROI tag weights.  Records that score
        below ``confidence_threshold`` are ignored so that upstream callers can
        fall back to static templates when no confident snippets remain.
        """

        records = list(patches)
        metas = [r.get("metadata", {}) for r in records]
        ts_vals = [float(m.get("ts") or 0.0) for m in metas]
        min_ts = min(ts_vals) if ts_vals else 0.0
        max_ts = max(ts_vals) if ts_vals else 0.0
        span = max(max_ts - min_ts, 1.0)

        successes: List[tuple[float, str]] = []
        failures: List[tuple[float, str, str, str]] = []
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
                summary = str(meta.get("summary") or "this pattern")
                outcome = str(meta.get("outcome") or "a failure")
                failures.append((score, summary, outcome, snippet))

        successes.sort(key=lambda x: x[0], reverse=True)
        failures.sort(key=lambda x: x[0], reverse=True)

        lines: List[str] = []
        headers: List[str] = []
        example_order: List[str] = []
        success_header = (
            self.trained_headers[0]
            if self.trained_headers
            else self.success_header
        )
        order_pref = self.trained_example_order or ["success", "failure"]

        # Process groups according to preferred order with optional limit
        remaining_successes = successes
        remaining_failures = failures
        limit = self.trained_example_count
        used = 0
        for group in order_pref:
            if limit is not None and used >= limit:
                break
            if group == "success" and remaining_successes:
                lines.append(success_header)
                headers.append(success_header)
                take = remaining_successes
                if limit is not None:
                    take = take[: max(0, limit - used)]
                for _, text in take:
                    lines.extend(text.splitlines())
                    lines.append("")
                    example_order.append("success")
                used += len(take)
                remaining_successes = []
            elif group == "failure" and remaining_failures:
                take = remaining_failures
                if limit is not None:
                    take = take[: max(0, limit - used)]
                for _, summary, outcome, text in take:
                    header = self.failure_header.format(
                        summary=summary, outcome=outcome
                    )
                    lines.append(header)
                    headers.append(header)
                    lines.extend(text.splitlines())
                    lines.append("")
                    example_order.append("failure")
                used += len(take)
                remaining_failures = []

        # Append any leftover groups not covered by the learned order
        if limit is None or used < limit:
            if remaining_successes:
                lines.append(success_header)
                headers.append(success_header)
                take = remaining_successes
                if limit is not None:
                    take = take[: max(0, limit - used)]
                for _, text in take:
                    lines.extend(text.splitlines())
                    lines.append("")
                    example_order.append("success")
                used += len(take)
            if (limit is None or used < limit) and remaining_failures:
                take = remaining_failures
                if limit is not None:
                    take = take[: max(0, limit - used)]
                for _, summary, outcome, text in take:
                    header = self.failure_header.format(
                        summary=summary, outcome=outcome
                    )
                    lines.append(header)
                    headers.append(header)
                    lines.extend(text.splitlines())
                    lines.append("")
                    example_order.append("failure")
                used += len(take)
        self.last_metadata = {
            "headers": headers,
            "example_order": example_order,
            "tone": self.tone,
            "structured_sections": self.trained_structured_sections or [],
            "example_count": len(example_order),
            "example_placement": self.trained_example_placement or "end",
            "length": self.trained_length or "unknown",
        }
        return [line for line in lines if line]

    # ------------------------------------------------------------------
    def build_prompt(
        self,
        task: str,
        *,
        context: str | None = None,
        summaries: List[str] | None = None,
        retrieval_context: str | None = None,
        retry_trace: str | None = None,
        tone: str | None = None,
        strategy: str | None = None,
        target_region: TargetRegion | None = None,
        context_builder: ContextBuilder,
    ) -> Prompt:
        """Return a :class:`Prompt` for *task* using retrieved patch examples.

        ``context_builder`` supplies token counting and ROI helpers for the
        invocation and must be provided explicitly.

        ``context`` and ``retrieval_context`` allow callers to prepend
        additional information such as the snippet body, repository layout or
        metadata from vector retrieval.  ``summaries`` may contain short
        descriptions of file chunks when the full source is too large to
        include.  ``retry_trace`` may contain failure logs or tracebacks from a
        prior attempt.  ``strategy`` selects an optional template from
        ``prompt_strategies.yaml`` that is inserted near the start of the prompt
        and recorded in the prompt metadata.  Recognised values include
        ``strict_fix``, ``delete_rebuild``, ``comment_refactor`` and
        ``unit_test_rewrite``.  When supplied a "Previous failure" section is
        appended and the details are de-duplicated so repeated retries do not
        accumulate duplicate traces.  When retrieval fails or the average
        confidence of returned patches falls below ``confidence_threshold`` a
        static fallback template is returned.
        """

        if context_builder is None:
            raise ValueError("context_builder is required")
        builder = context_builder

        def _count(text: str) -> int:
            counter = getattr(builder, "_count_tokens", None)
            if counter is None:
                if _ENCODER is not None:
                    def _token_counter(s: Any) -> int:
                        return len(_ENCODER.encode(str(s)))

                    counter = _token_counter
                else:
                    def _default_counter(s: Any) -> int:
                        return len(str(s).split())

                    counter = _default_counter
            return counter(text)

        def _trim(text: str, limit: int) -> str:
            counter = getattr(builder, "_count_tokens", None)
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
        self._maybe_refresh_optimizer()
        if not self._optimizer_applied:
            self.apply_optimizer_format(__name__, "build_prompt")
        if tone is not None:
            self.tone = tone
        retriever = self.patch_retriever or self.retriever
        if retry_trace and target_region is None:
            target_region = extract_target_region(retry_trace)
        if retriever is None:
            logging.info("No retriever available; falling back to static template")
            self._optimizer_applied = False
            return Prompt(system=SYSTEM_NOTICE, user=self._static_prompt())

        try:
            result = retriever.search(task, top_k=self.top_n)
        except Exception as exc:  # pragma: no cover - best effort
            logging.exception("Snippet retrieval failed: %s", exc)
            audit_log_event(
                "prompt_engine_fallback",
                {"goal": task, "reason": "retrieval_error", "error": str(exc)},
            )
            self._optimizer_applied = False
            return Prompt(system=SYSTEM_NOTICE, user=self._static_prompt())

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
            self._optimizer_applied = False
            return Prompt(system=SYSTEM_NOTICE, user=self._static_prompt())

        if isinstance(result, tuple):
            records = result[0]
        else:
            records = result
        if (
            context
            and summaries is None
            and self.chunk_token_threshold
            and _count(context) > self.chunk_token_threshold
            and target_region is None
        ):
            chunks = split_into_chunks(context, self.chunk_token_threshold)
            summaries = []
            for i, ch in enumerate(chunks):
                summary_text = (
                    summarize_code(
                        ch.text, self.llm, context_builder=self.context_builder
                    )
                    if self.llm
                    else ch.text.splitlines()[0][:80]
                )
                summaries.append(f"Chunk {i}: {summary_text}")
            context = None
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
            self._optimizer_applied = False
            return Prompt(system=SYSTEM_NOTICE, user=self._static_prompt())

        examples: List[str] = []
        outcome_tags: List[str] = []
        for rec in ranked:
            meta = rec.get("metadata", {})
            snippet, _ = self._compress_patch(meta, max_tokens=self.max_tokens)
            if not snippet:
                continue
            examples.append(snippet)
            tag = meta.get("roi_tag") or meta.get("outcome")
            if tag:
                outcome_tags.append(str(tag))

        snippet_lines = self.build_snippets(ranked)

        filename = None
        raw_filename = None
        original_lines: List[str] = []
        if target_region is not None:
            raw_filename = target_region.filename or None
            filename = path_for_prompt(raw_filename) if raw_filename else None
            func = target_region.function or None
            instr = f"Modify only lines {target_region.start_line}-{target_region.end_line}"
            if func:
                instr += f" within function {func}"
            if filename:
                instr += f" in {filename}"
            instr += " unless surrounding logic is causally required."

            original_lines = list(getattr(target_region, "original_lines", []) or [])
            if not original_lines and raw_filename:
                try:
                    resolved = resolve_path(raw_filename)
                except Exception:
                    resolved = None
                if resolved and resolved.exists():
                    try:
                        file_lines = resolved.read_text(encoding="utf-8").splitlines()
                        start = max(target_region.start_line - 1, 0)
                        end = target_region.end_line
                        original_lines = file_lines[start:end]
                    except Exception:
                        original_lines = []
        else:
            instr = (
                "Modify only the provided lines unless surrounding logic is causally required."
            )

        lines: List[str] = [instr, ""]

        if strategy:
            tmpl = _load_strategy_templates().get(strategy)
            if tmpl:
                lines.append(str(tmpl).strip())
                lines.append("")

        if summaries:
            summary_text = "\n".join(summaries).strip()
            if summary_text:
                lines.append(summary_text)
                lines.append("")

        for section in self.trained_structured_sections or []:
            lines.append(f"{section.capitalize()}:")
            lines.append("")

        if self.trained_example_placement == "start":
            lines.extend(snippet_lines)
            lines.append("")

        if retrieval_context:
            lines.append(retrieval_context.strip())
            lines.append("")
        if context:
            lines.append(context.strip())
            lines.append("")

        if self.trained_example_placement == "middle":
            lines.extend(snippet_lines)
            lines.append("")

        lines.append(f"Given the following pattern, {task}")
        lines.append("")

        if self.trained_example_placement not in {"start", "middle"}:
            lines.extend(snippet_lines)

        if retry_trace:
            lines.extend(self._format_retry_trace(retry_trace))
        text = "\n".join(line for line in lines if line)
        text = _trim(text, self.token_threshold)
        meta: Dict[str, Any] = {"vector_confidences": scores}
        if strategy:
            meta["strategy"] = strategy
        if target_region is not None:
            region_meta = {
                "filename": filename or "",
                "start_line": target_region.start_line,
                "end_line": target_region.end_line,
                "function": target_region.function,
                "signature": getattr(
                    target_region, "func_signature", getattr(target_region, "signature", "")
                ),
                "original_lines": original_lines,
                "original_snippet": "\n".join(original_lines),
            }
            meta["target_region"] = region_meta
            try:
                self.last_metadata.update({"target_region": region_meta})
            except Exception:
                self.last_metadata = {"target_region": region_meta}
        prompt_obj = Prompt(
            system=SYSTEM_NOTICE,
            user=text,
            examples=examples,
            vector_confidence=confidence,
            vector_confidences=scores,
            tags=outcome_tags,
            metadata=meta,
        )
        self._optimizer_applied = False
        return prompt_obj

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
            roi_component = float(roi_val) if roi_val is not None else 0.0
        except Exception:  # pragma: no cover - defensive
            roi_component = 0.0

        ts = float(meta.get("ts") or 0.0)
        recency_component = (ts - min_ts) / span if span else 0.0

        tag_weight = 0.0
        tag = meta.get("roi_tag")
        if tag is not None:
            try:
                valid = RoiTag.validate(tag)
                key = getattr(valid, "value", valid)
                tag_weight = float(self.roi_tag_weights.get(key, 0.0))
            except Exception:  # pragma: no cover - defensive
                tag_weight = 0.0

        return (
            self.roi_weight * roi_component
            + self.recency_weight * recency_component
            + tag_weight
        )

    # ------------------------------------------------------------------
    def _count_tokens(self, text: str) -> int:
        """Return token count for ``text`` using available tokenizers."""

        counter = getattr(self.context_builder, "_count_tokens", None)
        if counter is None:
            if _ENCODER is not None:
                def _token_counter(s: Any) -> int:
                    return len(_ENCODER.encode(str(s)))

                counter = _token_counter
            else:
                def _default_counter(s: Any) -> int:
                    return len(str(s).split())

                counter = _default_counter
        return counter(text)

    # ------------------------------------------------------------------
    def _trim_tokens(self, text: str, limit: int) -> str:
        """Trim ``text`` to ``limit`` tokens using available tokenizers."""

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
        summaries: List[str] | None = None,
        top_n: int = 5,
        confidence_threshold: float = 0.3,
        retriever: Retriever | None = None,
        context_builder: ContextBuilder,
        roi_tracker: Any | None = None,
        success_header: str = "Given the following pattern:",
        failure_header: str = "Avoid {summary} because it caused {outcome}:",
        tone: str = "neutral",
        trainer: PromptMemoryTrainer | None = None,
        optimizer: PromptOptimizer | None = None,
        optimizer_refresh_interval: int | None = None,
        target_region: TargetRegion | None = None,
        strategy: str | None = None,
    ) -> Prompt:
        """Class method wrapper used by existing callers and tests."""
        if context_builder is None:
            raise ValueError("context_builder is required")

        return build_prompt(
            goal,
            retry_trace=retry_trace,
            context=context,
            retrieval_context=retrieval_context,
            summaries=summaries,
            context_builder=context_builder,
            retriever=retriever,
            roi_tracker=roi_tracker,
            confidence_threshold=confidence_threshold,
            top_n=top_n,
            success_header=success_header,
            failure_header=failure_header,
            tone=tone,
            trainer=trainer,
            optimizer=optimizer,
            optimizer_refresh_interval=optimizer_refresh_interval,
            target_region=target_region,
            strategy=strategy,
        )


def build_prompt(
    goal: str,
    retry_trace: str | None = None,
    *,
    context: str | None = None,
    retrieval_context: str | None = None,
    summaries: List[str] | None = None,
    context_builder: ContextBuilder,
    retriever: Retriever | None = None,
    roi_tracker: Any | None = None,
    confidence_threshold: float = 0.3,
    top_n: int = 5,
    success_header: str = "Given the following pattern:",
    failure_header: str = "Avoid {summary} because it caused {outcome}:",
    tone: str = "neutral",
    trainer: PromptMemoryTrainer | None = None,
    optimizer: PromptOptimizer | None = None,
    optimizer_refresh_interval: int | None = None,
    target_region: TargetRegion | None = None,
    strategy: str | None = None,
) -> Prompt:
    """Convenience helper to build a :class:`Prompt` without instantiating ``PromptEngine``."""
    if context_builder is None:
        raise ValueError("context_builder is required")

    engine = PromptEngine(
        retriever=retriever,
        context_builder=context_builder,
        roi_tracker=roi_tracker,
        confidence_threshold=confidence_threshold,
        top_n=top_n,
        success_header=success_header,
        failure_header=failure_header,
        tone=tone,
        trainer=trainer,
        optimizer=optimizer,
        optimizer_refresh_interval=optimizer_refresh_interval,
    )
    build = engine.build_prompt
    return build(
        goal,
        context=context,
        retrieval_context=retrieval_context,
        retry_trace=retry_trace,
        tone=tone,
        summaries=summaries,
        strategy=strategy,
        target_region=target_region,
        context_builder=context_builder,
    )


__all__ = [
    "PromptEngine",
    "build_prompt",
    "DEFAULT_TEMPLATE",
    "diff_within_target_region",
]
