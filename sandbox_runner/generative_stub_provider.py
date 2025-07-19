from __future__ import annotations

"""Stub provider using a small language model via ``transformers``."""

from typing import Any, Dict, List
import json
import logging
import os
import re

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional dependency
    pipeline = None  # type: ignore

logger = logging.getLogger(__name__)

_GENERATOR = None


def _load_generator():
    """Return a text generation pipeline or ``None`` when unavailable."""
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR
    if pipeline is None:
        return None
    try:
        model = os.getenv("SANDBOX_STUB_MODEL", "distilgpt2")
        _GENERATOR = pipeline("text-generation", model=model)
    except Exception:  # pragma: no cover - model load failures
        logger.exception("failed to load generative model")
        _GENERATOR = None
    return _GENERATOR


def generate_stubs(stubs: List[Dict[str, Any]], ctx: dict) -> List[Dict[str, Any]]:
    """Generate or enhance ``stubs`` using a language model.

    When the backend is unavailable the input ``stubs`` are returned
    unchanged.
    """
    gen = _load_generator()
    if gen is None:
        return stubs

    new_stubs: List[Dict[str, Any]] = []
    for stub in stubs:
        prompt = f"Create a JSON object with fields {list(stub.keys())}:"
        try:
            text = gen(prompt, max_length=64, num_return_sequences=1)[0][
                "generated_text"
            ]
            match = re.search(r"{.*}", text, flags=re.S)
            if match:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    new_stubs.append(data)
                    continue
        except Exception:  # pragma: no cover - generation failures
            logger.exception("stub generation failed")
        new_stubs.append(stub)
    return new_stubs
