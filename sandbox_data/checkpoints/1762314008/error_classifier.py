"""Lightweight stack trace classifier.

The classifier attempts to load a fine-tuned language model from disk.  When
unavailable it falls back to a couple of simple heuristics.  The interface is
kept intentionally small so the surrounding system can optionally make use of
more capable models when present without failing when dependencies are
missing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

from dynamic_path_router import resolve_path

try:  # pragma: no cover - optional heavy dependency
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover - fall back to heuristics
    AutoModelForCausalLM = AutoTokenizer = None  # type: ignore

_MODEL_CACHE: tuple | None = None
try:
    DEFAULT_PATH = resolve_path("micro_models/error_classifier_model")
except FileNotFoundError:  # pragma: no cover - model may be missing in tests
    DEFAULT_PATH = Path("micro_models/error_classifier_model")


def _load_model(path: Path) -> tuple | None:
    global _MODEL_CACHE
    if _MODEL_CACHE is None and AutoModelForCausalLM:
        try:
            tok = AutoTokenizer.from_pretrained(path)  # type: ignore[union-attr]
            mdl = AutoModelForCausalLM.from_pretrained(path)  # type: ignore[union-attr]
            _MODEL_CACHE = (tok, mdl)
        except Exception:  # pragma: no cover - model not available
            _MODEL_CACHE = (None, None)
    return _MODEL_CACHE


_HEURISTICS = [
    (r"KeyError|IndexError|FileNotFoundError|ZeroDivisionError|AttributeError", "RuntimeFault", "Check inputs and object attributes."),
    (r"ModuleNotFoundError|ImportError|PackageNotFoundError|OSError", "DependencyMismatch", "Verify installed packages and versions."),
    (r"AssertionError|NotImplementedError", "LogicMisfire", "Review assertions and implemented interfaces."),
    (r"TypeError|ValueError", "SemanticBug", "Validate data types and values."),
    (r"MemoryError", "ResourceLimit", "Reduce memory usage or increase limits."),
    (r"TimeoutError", "Timeout", "Increase timeout or optimise operations."),
    (r"ConnectionError|HTTPError", "ExternalAPI", "Check external service availability."),
]


def classify_error(trace: str, *, model_path: Path | None = None) -> Tuple[str, str, float]:
    """Return ``(category, fix, confidence)`` for ``trace``.

    The confidence is a coarse estimate: ``0.9`` when a model predicts a
    category or a heuristic rule matches, otherwise ``0``.
    """

    path = model_path or DEFAULT_PATH
    if AutoModelForCausalLM is not None:
        tok, mdl = _load_model(path) or (None, None)
        if tok is not None and mdl is not None:
            try:
                prompt = (
                    "Classify the error and propose a fix.\nStack trace:\n" + trace + "\nResponse:"
                )
                inputs = tok(prompt, return_tensors="pt")  # type: ignore[union-attr]
                out_ids = mdl.generate(**inputs, max_new_tokens=64)  # type: ignore[union-attr]
                text = tok.decode(out_ids[0], skip_special_tokens=True)
                category = ""
                fix = ""
                for line in text.splitlines():
                    if line.lower().startswith("category:"):
                        category = line.split(":", 1)[1].strip()
                    elif line.lower().startswith("fix:"):
                        fix = line.split(":", 1)[1].strip()
                if category:
                    return category, fix, 0.9
            except Exception:  # pragma: no cover - model inference failure
                pass

    low = trace.lower()
    for pattern, category, fix in _HEURISTICS:
        if re.search(pattern, trace, re.IGNORECASE):
            return category, fix, 0.9
    return "", "", 0.0


__all__ = ["classify_error"]
