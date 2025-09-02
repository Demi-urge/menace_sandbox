from __future__ import annotations
"""Utilities for splitting code into token-limited chunks respecting AST boundaries."""

from typing import List
import ast
import hashlib
import json
from pathlib import Path

try:  # Optional dependency for accurate token counts
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - tiktoken may be missing
    tiktoken = None  # type: ignore

_encoder = None
if tiktoken is not None:  # pragma: no branch - simple import logic
    try:  # pragma: no cover - defensive
        _encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - encoder creation failed
        _encoder = None


def _count_tokens(text: str) -> int:
    """Return number of tokens in *text* using best available tokenizer."""

    if _encoder is not None:
        try:  # pragma: no cover - defensive
            return len(_encoder.encode(text))
        except Exception:
            pass
    return len(text.split())


def summarize_chunk(code: str) -> str:
    """Return a short summary for ``code``.

    Tries to use the optional micro-model summariser when available. If it is
    missing or fails, the original ``code`` is returned unchanged so callers can
    still operate on the raw chunks.
    """

    try:  # pragma: no cover - optional dependency
        from micro_models.diff_summarizer import summarize_diff as _summ
    except Exception:  # pragma: no cover - summariser may be missing
        return code
    try:  # pragma: no cover - defensive
        return _summ("", code) or code
    except Exception:
        return code


def _split_by_line(code: str, limit: int) -> List[str]:
    """Split *code* by lines ensuring each chunk stays under *limit* tokens."""

    if _count_tokens(code) <= limit:
        return [code]

    out: List[str] = []
    current: List[str] = []
    for line in code.splitlines():
        tentative = "\n".join(current + [line])
        if _count_tokens(tentative) <= limit or not current:
            current.append(line)
        else:
            out.append("\n".join(current))
            current = [line]
    if current:
        out.append("\n".join(current))
    return out


def split_into_chunks(code: str, max_tokens: int) -> List[str]:
    """Split *code* into chunks under *max_tokens* respecting AST boundaries.

    Parsing failures (syntax errors) fall back to simple line-based splitting.
    """

    try:
        module = ast.parse(code)
    except SyntaxError:  # pragma: no cover - error path tested separately
        return _split_by_line(code, max_tokens)

    lines = code.splitlines()
    chunks: List[str] = []
    prev_end = 1
    for node in module.body:
        if not hasattr(node, "lineno"):
            continue
        start = node.lineno
        end = getattr(node, "end_lineno", start)
        if start > prev_end:
            segment = "\n".join(lines[prev_end - 1:start - 1]).rstrip()
            if segment:
                chunks.extend(_split_by_line(segment, max_tokens))
        block = "\n".join(lines[start - 1:end]).rstrip()
        if block:
            chunks.extend(_split_by_line(block, max_tokens))
        prev_end = end + 1
    if prev_end <= len(lines):
        segment = "\n".join(lines[prev_end - 1:]).rstrip()
        if segment:
            chunks.extend(_split_by_line(segment, max_tokens))
    return chunks


# Directory used to store cached summaries. The repository already ships with
# an empty ``chunk_summary_cache`` directory, but ``mkdir`` is cheap and
# idempotent so we ensure it exists on import.
CACHE_DIR = Path("chunk_summary_cache")
CACHE_DIR.mkdir(exist_ok=True)


def get_chunk_summaries(path: Path, threshold: int) -> List[str]:
    """Return summaries for ``path`` split into ``threshold`` token chunks.

    Results are cached in ``chunk_summary_cache/`` using a SHA256 hash of the
    file contents. If the file changes, a new cache entry is written and the old
    one is ignored.
    """

    source = path.read_text()
    file_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"{file_hash}.json"

    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            if data.get("hash") == file_hash:
                chunks = data.get("chunks", [])
                if isinstance(chunks, list):
                    return [str(c) for c in chunks]
        except Exception:  # pragma: no cover - corrupted cache
            pass

    chunks = split_into_chunks(source, threshold)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    cache_file.write_text(
        json.dumps({"hash": file_hash, "chunks": summaries}, indent=2, sort_keys=True)
    )
    return summaries


__all__ = ["split_into_chunks", "get_chunk_summaries"]
