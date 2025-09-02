from __future__ import annotations

"""Utilities for splitting source code into token-limited chunks.

The :func:`split_into_chunks` function analyses the AST of the provided source
and creates chunks that try to keep functions and classes intact while staying
below a token threshold.  Token estimation uses the same helpers as the LLM
interface so the counts roughly match those used when sending prompts.
"""

from dataclasses import dataclass
import ast
import hashlib
import json
from pathlib import Path
from typing import List

from rate_limit import estimate_tokens


@dataclass(slots=True)
class Chunk:
    """Represents a chunk of source code."""

    source: str
    start_line: int
    end_line: int
    token_count: int


def _count_tokens(text: str) -> int:
    """Return the estimated token count for ``text``."""

    return estimate_tokens(text)


def _make_chunk(lines: List[str], start: int) -> Chunk:
    source = "\n".join(lines).rstrip()
    return Chunk(
        source=source,
        start_line=start,
        end_line=start + len(lines) - 1,
        token_count=_count_tokens(source),
    )


def _split_by_lines(lines: List[str], start: int, limit: int) -> List[Chunk]:
    """Split a list of ``lines`` starting at ``start`` respecting ``limit``."""

    out: List[Chunk] = []
    buf: List[str] = []
    buf_start = start
    for line in lines:
        tentative = "\n".join(buf + [line])
        if _count_tokens(tentative) <= limit or not buf:
            buf.append(line)
            continue
        out.append(_make_chunk(buf, buf_start))
        buf_start += len(buf)
        buf = [line]
    if buf:
        out.append(_make_chunk(buf, buf_start))
    return out


def split_into_chunks(code: str, max_tokens: int) -> List[Chunk]:
    """Split ``code`` into :class:`Chunk` objects under ``max_tokens``.

    The splitter keeps whole function and class definitions together when
    possible.  If a single definition exceeds ``max_tokens`` it is further split
    by lines as a fallback to ensure no chunk surpasses the limit.
    ``SyntaxError`` while parsing the code triggers a plain line-based split.
    """

    try:
        module = ast.parse(code)
    except SyntaxError:
        return _split_by_lines(code.splitlines(), 1, max_tokens)

    lines = code.splitlines()
    segments: List[tuple[int, int]] = []
    prev = 1
    for node in module.body:
        if not hasattr(node, "lineno"):
            continue
        start = node.lineno
        end = getattr(node, "end_lineno", start)
        if start > prev:
            segments.append((prev, start - 1))
        segments.append((start, end))
        prev = end + 1
    if prev <= len(lines):
        segments.append((prev, len(lines)))

    chunks: List[Chunk] = []
    current_lines: List[str] = []
    current_start = 0
    for start, end in segments:
        seg_lines = lines[start - 1:end]
        seg_text = "\n".join(seg_lines)
        seg_tokens = _count_tokens(seg_text)
        if seg_tokens > max_tokens:
            if current_lines:
                chunks.append(_make_chunk(current_lines, current_start))
                current_lines = []
            chunks.extend(_split_by_lines(seg_lines, start, max_tokens))
            current_start = 0
            continue

        tentative = ("\n".join(current_lines + seg_lines) if current_lines else seg_text)
        if current_lines and _count_tokens(tentative) > max_tokens:
            chunks.append(_make_chunk(current_lines, current_start))
            current_lines = seg_lines
            current_start = start
        else:
            if not current_lines:
                current_start = start
            current_lines.extend(seg_lines)

    if current_lines:
        chunks.append(_make_chunk(current_lines, current_start))

    return chunks


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
    summaries = [summarize_chunk(chunk.source) for chunk in chunks]
    cache_file.write_text(
        json.dumps({"hash": file_hash, "chunks": summaries}, indent=2, sort_keys=True)
    )
    return summaries


__all__ = ["Chunk", "split_into_chunks", "get_chunk_summaries"]
