"""Utilities for summarising code snippets with caching.

`summarize_code` delegates summarisation to an ``LLMClient`` instance and caches
results on disk keyed by the SHA256 hash of the snippet.  Summaries are stored
as JSON files inside ``chunk_summary_cache``.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
from typing import Any

from llm_interface import LLMClient, LLMResult
from prompt_types import Prompt

# Directory used to store cached summaries.
CACHE_DIR = Path(__file__).resolve().parent / "chunk_summary_cache"


def _ensure_cache_dir() -> None:
    """Create the cache directory if it does not exist.

    ``exist_ok=True`` makes the call safe under concurrent access.
    """

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        if not CACHE_DIR.is_dir():  # pragma: no cover - defensive programming
            raise


def _hash_code(code: str) -> str:
    """Return the SHA256 hash of ``code``."""

    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def load_summary(digest: str) -> str | None:
    """Return a cached summary for ``digest`` or ``None`` if not available."""

    path = CACHE_DIR / f"{digest}.json"
    try:
        with path.open("r", encoding="utf-8") as fh:
            data: Any = json.load(fh)
    except FileNotFoundError:
        return None
    except Exception:  # pragma: no cover - corrupted cache
        return None
    return data.get("summary")


def store_summary(digest: str, summary: str) -> None:
    """Persist ``summary`` for ``digest`` atomically."""

    _ensure_cache_dir()
    path = CACHE_DIR / f"{digest}.json"
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump({"hash": digest, "summary": summary}, fh)
    os.replace(tmp_path, path)


def summarize_code(code: str, llm: LLMClient) -> str:
    """Return a cached summary for ``code`` using ``llm`` if necessary."""

    code = code.strip()
    if not code:
        return ""

    digest = _hash_code(code)
    cached = load_summary(digest)
    if cached is not None:
        return cached

    prompt = Prompt(text=f"Summarize the following code:\n{code}\nSummary:")
    try:
        result: LLMResult = llm.generate(prompt)
        summary = getattr(result, "text", "").strip()
    except Exception:  # pragma: no cover - defensive against LLM failures
        summary = ""
    if summary:
        store_summary(digest, summary)
    return summary


__all__ = ["summarize_code", "load_summary", "store_summary"]
