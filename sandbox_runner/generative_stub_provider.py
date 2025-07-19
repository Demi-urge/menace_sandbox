from __future__ import annotations

"""Stub provider using a language model via ``transformers``."""

from typing import Any, Dict, List
import json
import logging
import os
import re
from pathlib import Path
from collections import Counter

from .input_history_db import InputHistoryDB

ROOT = Path(__file__).resolve().parents[1]

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
    model = os.getenv("SANDBOX_STUB_MODEL")
    candidates = ["gpt2-large", "distilgpt2"] if model is None else [model]
    for name in candidates:
        try:
            _GENERATOR = pipeline("text-generation", model=name)
            break
        except Exception:  # pragma: no cover - model load failures
            logger.exception("failed to load model %s", name)
            _GENERATOR = None
    return _GENERATOR


def _get_history_db() -> InputHistoryDB:
    path = os.getenv(
        "SANDBOX_INPUT_HISTORY", str(ROOT / "sandbox_data" / "input_history.db")
    )
    return InputHistoryDB(path)


def _aggregate(records: List[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, List[Any]] = {}
    for rec in records:
        for k, v in rec.items():
            stats.setdefault(k, []).append(v)
    result: dict[str, Any] = {}
    for k, vals in stats.items():
        if all(isinstance(v, (int, float)) for v in vals):
            avg = sum(float(v) for v in vals) / len(vals)
            if all(isinstance(v, int) for v in vals):
                avg = int(round(avg))
            result[k] = avg
        else:
            cnt = Counter(vals)
            result[k] = cnt.most_common(1)[0][0]
    return result


def generate_stubs(stubs: List[Dict[str, Any]], ctx: dict) -> List[Dict[str, Any]]:
    """Generate or enhance ``stubs`` using recent history or a language model."""

    strategy = ctx.get("strategy")
    if strategy == "history":
        try:
            records = _get_history_db().recent(50)
        except Exception:
            logger.exception("failed to load input history")
            records = []
        if records:
            hist = _aggregate(records)
            return [dict(hist) for _ in range(max(1, len(stubs)))]
        return stubs

    gen = _load_generator()
    if gen is None:
        return stubs

    new_stubs: List[Dict[str, Any]] = []
    for stub in stubs:
        func = ctx.get("target")
        name = getattr(func, "__name__", "function")
        args = ", ".join(f"{k}={v!r}" for k, v in stub.items())
        prompt = (
            f"Create a JSON object for '{name}' using arguments with example values: "
            f"{args}. Return only the JSON object."
        )
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
