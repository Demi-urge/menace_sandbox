"""Utility to summarise code differences using a fine-tuned model."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, TYPE_CHECKING

from dynamic_path_router import resolve_path
from context_builder_util import ensure_fresh_weights

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from vector_service.context_builder import ContextBuilder

try:
    _MODEL_PATH = resolve_path("micro_models/diff_summarizer_model")
except FileNotFoundError:  # pragma: no cover - model may be absent in tests
    _MODEL_PATH = Path("micro_models/diff_summarizer_model")
_tokenizer = None
_hf_model = None

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
    intent = {"hint": hint} if hint else None
    prompt_obj = context_builder.build_prompt(
        "Summarize the code change.",
        context=diff,
        intent_metadata=intent,
        top_k=top_k,
    )
    prompt_text = ""
    if getattr(prompt_obj, "system", ""):
        prompt_text += prompt_obj.system + "\n"
    prompt_text += prompt_obj.user + "\nSummary:"

    inputs = tokenizer(prompt_text, return_tensors="pt")  # type: ignore[call-arg]
    if torch is not None:
        inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}  # type: ignore[attr-defined]
    output = hf_model.generate(**inputs, max_new_tokens=max_new_tokens)  # type: ignore[call-arg]
    text = tokenizer.decode(output[0], skip_special_tokens=True)  # type: ignore[call-arg]
    summary = text.split("Summary:")[-1].strip()
    return summary


__all__ = ["summarize_diff"]
