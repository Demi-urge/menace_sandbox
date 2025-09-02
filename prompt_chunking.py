from __future__ import annotations

"""Utilities for splitting code into token-limited chunks and caching summaries."""

import ast
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Optional encoder dependency -------------------------------------------------
try:  # pragma: no cover - try to reuse existing encoder
    from prompt_engine import _ENCODER as _PE_ENCODER  # type: ignore
except Exception:  # pragma: no cover - avoid hard dependency
    _PE_ENCODER = None

try:  # pragma: no cover - fallback to direct tiktoken import
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - dependency may be missing
    tiktoken = None  # type: ignore

if _PE_ENCODER is not None:
    _ENCODER = _PE_ENCODER
elif tiktoken is not None:
    _ENCODER = tiktoken.get_encoding("cl100k_base")
else:  # pragma: no cover - final fallback when no tokenizer is available
    _ENCODER = None


def _count_tokens(text: str) -> int:
    """Return the number of tokens in ``text`` using the best available encoder."""

    if _ENCODER is not None:
        try:  # pragma: no cover - defensive
            return len(_ENCODER.encode(text))
        except Exception:
            pass
    return len(text.split())


def _split_to_limit(code: str, token_limit: int) -> List[str]:
    """Yield ``code`` split into chunks under ``token_limit`` tokens."""

    if _count_tokens(code) <= token_limit:
        return [code]
    out: List[str] = []
    current: List[str] = []
    for line in code.splitlines():
        tentative = "\n".join(current + [line])
        if _count_tokens(tentative) <= token_limit or not current:
            current.append(line)
            continue
        out.append("\n".join(current))
        current = [line]
    if current:
        out.append("\n".join(current))
    return out


def chunk_code(path: Path, token_limit: int) -> List[str]:
    """Split ``path`` into top-level function/class chunks ≤ ``token_limit`` tokens."""

    source = path.read_text()
    module = ast.parse(source)
    lines = source.splitlines()
    chunks: List[str] = []
    prev_end = 1
    for node in module.body:
        if hasattr(node, "lineno"):
            start = getattr(node, "lineno")
            end = getattr(node, "end_lineno", start)
            # Include any intervening top-level statements
            if start > prev_end:
                segment = "\n".join(lines[prev_end - 1:start - 1]).rstrip()
                if segment:
                    chunks.extend(_split_to_limit(segment, token_limit))
            block = "\n".join(lines[start - 1:end]).rstrip()
            chunks.extend(_split_to_limit(block, token_limit))
            prev_end = end + 1
    # Trailing statements
    if prev_end <= len(lines):
        segment = "\n".join(lines[prev_end - 1:]).rstrip()
        if segment:
            chunks.extend(_split_to_limit(segment, token_limit))
    logger.debug("split %s into %d chunks", path, len(chunks))
    return chunks


def summarize_code(code: str) -> str:
    """Return a short summary for ``code`` using available micro-models."""

    try:  # pragma: no cover - optional dependency
        from micro_models.diff_summarizer import summarize_diff as _summ
    except Exception:  # pragma: no cover - summariser may be missing
        return code
    try:  # pragma: no cover - defensive
        return _summ("", code) or code
    except Exception:
        return code


CACHE_DIR = Path(__file__).resolve().parent / "chunk_summary_cache"

logger = logging.getLogger(__name__)


def get_chunk_summaries(path: Path, token_limit: int) -> List[Dict[str, str]]:
    """Return cached or freshly generated summaries for ``path`` chunks."""

    CACHE_DIR.mkdir(exist_ok=True)
    out: List[Dict[str, str]] = []
    for chunk in chunk_code(path, token_limit):
        h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        cache_file = CACHE_DIR / f"{h}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                if data.get("code") == chunk:
                    logger.debug("using cached summary for %s", h)
                    out.append(data)
                    continue
            except Exception:
                logger.warning("failed to load cache file %s", cache_file, exc_info=True)
        logger.debug("cache miss for chunk %s", h)
        summary = summarize_code(chunk)
        data = {
            "hash": h,
            "code": chunk,
            "summary": summary,
            "path": str(path),
            "created_at": datetime.utcnow().isoformat(),
            "token_limit": token_limit,
        }
        cache_file.write_text(json.dumps(data, indent=2, sort_keys=True))
        out.append(data)
    return out


__all__ = ["chunk_code", "summarize_code", "get_chunk_summaries"]
