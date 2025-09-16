"""Utility to summarise code differences using a fine-tuned model."""

from __future__ import annotations

import importlib
import logging
import sys
import types
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, TYPE_CHECKING

from dynamic_path_router import resolve_path
from context_builder_util import ensure_fresh_weights

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from llm_interface import Prompt
    from vector_service.context_builder import ContextBuilder

try:
    _MODEL_PATH = resolve_path("micro_models/diff_summarizer_model")
except FileNotFoundError:  # pragma: no cover - model may be absent in tests
    _MODEL_PATH = Path("micro_models/diff_summarizer_model")
_tokenizer = None
_hf_model = None

LOGGER = logging.getLogger(__name__)


def load_self_coding_engine() -> type:
    """Return the :class:`SelfCodingEngine` class, initialising package hooks."""

    try:
        from self_coding_engine import SelfCodingEngine  # type: ignore
    except ImportError:
        base = Path(__file__).resolve().parent.parent
        pkg = sys.modules.get("menace_sandbox")
        if pkg is None:
            pkg = types.ModuleType("menace_sandbox")
            pkg.__path__ = [str(base)]  # type: ignore[attr-defined]
            sys.modules["menace_sandbox"] = pkg
        try:
            module = importlib.import_module("menace_sandbox.self_coding_engine")
        except Exception:
            module = None

        if module is None:

            class _FallbackEngine:
                """Lightweight shim providing :meth:`build_enriched_prompt`."""

                logger = LOGGER
                _last_retry_trace: str | None = None
                _last_prompt: "Prompt" | None = None
                _last_prompt_metadata: Dict[str, Any]

                def build_enriched_prompt(
                    self,
                    goal: str | Mapping[str, Any],
                    *,
                    intent: Mapping[str, Any] | None = None,
                    error_log: str | None = None,
                    context_builder: "ContextBuilder",
                ) -> "Prompt":
                    if isinstance(goal, Mapping):
                        payload: Dict[str, Any] = dict(goal)
                        if intent:
                            payload.update(intent)
                        query = str(
                            payload.get("query")
                            or payload.get("task")
                            or payload.get("text")
                            or ""
                        ).strip()
                        if error_log is None:
                            error_log = payload.pop("error_log", None)
                    else:
                        query = str(goal).strip()
                        payload = dict(intent or {})
                    if not query:
                        raise ValueError("goal must supply a non-empty query")

                    top_k_val = payload.get("top_k", 5)
                    try:
                        top_k = int(top_k_val)
                    except (TypeError, ValueError):
                        top_k = top_k_val  # type: ignore[assignment]
                    retriever = payload.pop("retriever", None)
                    if retriever is None:
                        retriever = getattr(context_builder, "patch_retriever", None)
                    meta_payload = dict(payload)
                    kwargs: Dict[str, Any] = {"top_k": top_k}
                    if meta_payload:
                        kwargs["intent_metadata"] = meta_payload
                    log_arg = error_log or self._last_retry_trace
                    if log_arg:
                        kwargs["error_log"] = log_arg
                    if retriever is not None:
                        kwargs["retriever"] = retriever
                    try:
                        prompt_obj = context_builder.build_prompt(query, **kwargs)
                    except TypeError:
                        kwargs.pop("retriever", None)
                        kwargs.pop("error_log", None)
                        meta = kwargs.pop("intent_metadata", None)
                        if meta is not None:
                            alt_kwargs = dict(kwargs)
                            alt_kwargs["intent"] = meta
                            try:
                                prompt_obj = context_builder.build_prompt(
                                    query,
                                    **alt_kwargs,
                                )
                            except TypeError:
                                alt_kwargs.pop("intent", None)
                                prompt_obj = context_builder.build_prompt(query, **alt_kwargs)
                        else:
                            prompt_obj = context_builder.build_prompt(query, **kwargs)

                    meta = dict(getattr(prompt_obj, "metadata", {}) or {})
                    intent_meta = meta_payload
                    meta.setdefault("intent", {}).update(intent_meta)
                    prompt_obj.metadata = meta
                    prompt_obj.origin = "self_coding_engine_fallback"
                    self._last_prompt = prompt_obj
                    self._last_prompt_metadata = meta
                    return prompt_obj

            SelfCodingEngine = _FallbackEngine
        else:
            SelfCodingEngine = module.SelfCodingEngine
    return SelfCodingEngine


def _build_summary_prompt(
    request: str,
    payload: Dict[str, Any],
    *,
    context_builder: "ContextBuilder",
) -> "Prompt":
    """Return an enriched prompt for the diff summarisation request."""

    SelfCodingEngine = load_self_coding_engine()

    engine = SelfCodingEngine.__new__(SelfCodingEngine)
    engine.logger = LOGGER
    engine._last_retry_trace = None
    engine._last_prompt = None
    engine._last_prompt_metadata = {}
    builder = context_builder

    class _RetrieverProxy:
        def __init__(self, inner: "ContextBuilder") -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:  # pragma: no cover - simple proxy
            return getattr(self._inner, name)

        def build_prompt(self, query: str, **kwargs: Any) -> "Prompt":
            kwargs.setdefault("retriever", getattr(self._inner, "patch_retriever", None))
            return self._inner.build_prompt(query, **kwargs)

    proxy = _RetrieverProxy(builder)
    return engine.build_enriched_prompt(
        request,
        intent=payload,
        context_builder=proxy,
    )

try:  # pragma: no cover - optional heavy dependency
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except Exception:  # pragma: no cover
    AutoModelForCausalLM = AutoTokenizer = None  # type: ignore
    torch = None  # type: ignore


def _load_model() -> Tuple[object, object] | Tuple[None, None]:
    """Lazy load the tokenizer and model if available."""
    global _tokenizer, _hf_model
    if _hf_model is not None and _tokenizer is not None:
        return _tokenizer, _hf_model
    if AutoModelForCausalLM is None:
        return None, None
    if not _MODEL_PATH.exists():
        return None, None
    _tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_PATH))  # type: ignore[assignment]
    _hf_model = AutoModelForCausalLM.from_pretrained(str(_MODEL_PATH))  # type: ignore[assignment]
    return _tokenizer, _hf_model


def summarize_diff(
    before: str,
    after: str,
    max_new_tokens: int = 128,
    *,
    context_builder: "ContextBuilder",
) -> str:
    """Return a short description of the code change.

    If the fine-tuned model or required libraries are unavailable, an empty string
    is returned so callers may fall back to alternative approaches.
    """

    if context_builder is None:
        raise ValueError("context_builder is required")
    ensure_fresh_weights(context_builder)

    tokenizer, hf_model = _load_model()
    if tokenizer is None or hf_model is None:
        return ""

    diff = before + "\n" + after
    hint = after.splitlines()[0] if after else (before.splitlines()[0] if before else "")
    top_k = 5 if hint else 0
    payload: Dict[str, Any] = {
        "task": "summarize_diff",
        "diff": diff,
        "before": before,
        "after": after,
        "top_k": top_k,
    }
    if hint:
        payload["hint"] = hint

    prompt_obj = _build_summary_prompt(
        "Summarize the code change.\nSummary:",
        payload,
        context_builder=context_builder,
    )

    inputs = tokenizer(str(prompt_obj), return_tensors="pt")  # type: ignore[call-arg]
    if torch is not None:
        inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}  # type: ignore[attr-defined]
    output = hf_model.generate(**inputs, max_new_tokens=max_new_tokens)  # type: ignore[call-arg]
    text = tokenizer.decode(output[0], skip_special_tokens=True)  # type: ignore[call-arg]
    summary = text.split("Summary:")[-1].strip()
    return summary


__all__ = ["summarize_diff"]
