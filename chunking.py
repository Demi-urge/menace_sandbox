from __future__ import annotations

"""Code chunking utilities with token-aware grouping and summarisation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

from context_builder import handle_failure, PromptBuildError

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from llm_interface import LLMClient
    from vector_service.context_builder import ContextBuilder
import ast
import hashlib
import json
import os
import threading

try:  # Optional dependency for accurate token counting
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - tiktoken may be missing
    tiktoken = None  # type: ignore

try:  # pragma: no cover - optional settings dependency
    from sandbox_settings import SandboxSettings  # type: ignore
except Exception:  # pragma: no cover - allow running without settings
    SandboxSettings = None  # type: ignore

from dynamic_path_router import resolve_path
from chunk_summary_cache import ChunkSummaryCache

_ENCODER = None
if tiktoken is not None:  # pragma: no branch - simple import logic
    try:  # pragma: no cover - defensive
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - encoder creation failed
        _ENCODER = None

# Directory used for caching summaries of individual code snippets.  Sharing the
# directory with :class:`ChunkSummaryCache` keeps cache files in one place and
# mirrors the behaviour of the removed ``chunk_summarizer`` module.
try:
    SNIPPET_CACHE_DIR = resolve_path("chunk_summary_cache")
except FileNotFoundError:
    from dynamic_path_router import get_project_root

    SNIPPET_CACHE_DIR = get_project_root() / "chunk_summary_cache"


def _ensure_snippet_cache_dir() -> None:
    try:
        SNIPPET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        if not SNIPPET_CACHE_DIR.is_dir():  # pragma: no cover - defensive
            raise


def _hash_snippet(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _load_snippet_summary(digest: str) -> str | None:
    path = SNIPPET_CACHE_DIR / f"{digest}.json"
    try:
        with path.open("r", encoding="utf-8") as fh:
            data: Dict[str, str] = json.load(fh)
    except FileNotFoundError:
        return None
    except Exception:  # pragma: no cover - corrupted cache
        return None
    return data.get("summary")


def _store_snippet_summary(digest: str, summary: str) -> None:
    _ensure_snippet_cache_dir()
    path = SNIPPET_CACHE_DIR / f"{digest}.json"
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump({"hash": digest, "summary": summary}, fh)
    os.replace(tmp_path, path)


def _count_tokens(text: str) -> int:
    """Return number of tokens in *text* using best available tokenizer."""

    if _ENCODER is not None:
        try:  # pragma: no cover - defensive
            return len(_ENCODER.encode(text))
        except Exception:
            pass
    return len(text.split())


@dataclass(slots=True)
class CodeChunk:
    """Representation of a contiguous code block."""

    start_line: int
    end_line: int
    text: str
    hash: str
    token_count: int


def split_into_chunks(
    code: str,
    max_tokens: int,
    *,
    line_ranges: List[tuple[int, int]] | None = None,
) -> List[CodeChunk]:
    """Split ``code`` into :class:`CodeChunk` objects under ``max_tokens`` tokens.

    ``line_ranges`` can be used to force chunk boundaries at specific line
    numbers.  Each ``(start, end)`` pair is treated as an inclusive range that
    should not be merged with surrounding code.  This allows callers to ensure
    that targeted regions map cleanly to chunk boundaries.
    """

    lines = code.splitlines()
    try:
        module = ast.parse(code)
    except SyntaxError:  # pragma: no cover - syntax error fallback
        return _split_by_lines(lines, 1, max_tokens)

    segments: List[tuple[int, int, str]] = []
    prev_end = 1
    for node in module.body:
        if not hasattr(node, "lineno"):
            continue
        start = node.lineno
        end = getattr(node, "end_lineno", start)
        if start > prev_end:
            filler = "\n".join(lines[prev_end - 1:start - 1]).rstrip()
            if filler:
                segments.append((prev_end, start - 1, filler))
        block = "\n".join(lines[start - 1:end]).rstrip()
        segments.append((start, end, block))
        prev_end = end + 1
    if prev_end <= len(lines):
        filler = "\n".join(lines[prev_end - 1:]).rstrip()
        if filler:
            segments.append((prev_end, len(lines), filler))

    # Split segments so that explicit line range boundaries align with segment
    # edges.  This makes it possible to treat the specified ranges as atomic
    # units during chunk assembly.
    if line_ranges:
        boundaries = sorted({b for s, e in line_ranges for b in (s, e + 1)})
        adjusted: List[tuple[int, int, str]] = []
        for start, end, text in segments:
            curr_start = start
            curr_lines = text.splitlines()
            for b in (b for b in boundaries if start < b <= end):
                rel = b - curr_start
                part = "\n".join(curr_lines[:rel]).rstrip()
                if part:
                    adjusted.append((curr_start, b - 1, part))
                curr_lines = curr_lines[rel:]
                curr_start = b
            remaining = "\n".join(curr_lines).rstrip()
            if remaining:
                adjusted.append((curr_start, end, remaining))
        segments = adjusted

    chunks: List[CodeChunk] = []
    current: List[str] = []
    current_start = 1
    current_end = 1
    token_total = 0
    current_range: tuple[int, int] | None = None

    def _range_for(seg_start: int, seg_end: int) -> tuple[int, int] | None:
        if not line_ranges:
            return None
        for s, e in line_ranges:
            if s <= seg_start and seg_end <= e:
                return s, e
        return None

    for start, end, text in segments:
        count = _count_tokens(text)
        seg_range = _range_for(start, end)
        if current and (current_range != seg_range or token_total + count > max_tokens):
            chunk_text = "\n".join(current).rstrip()
            h = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            t_count = _count_tokens(chunk_text)
            chunks.append(CodeChunk(current_start, current_end, chunk_text, h, t_count))
            current = []
            token_total = 0
        if count > max_tokens:
            # Large segment: split by lines and treat each portion separately.
            chunks.extend(_split_by_lines(text.splitlines(), start, max_tokens))
            current_start = end + 1
            current_end = end
            current_range = None
            continue
        if not current:
            current_start = start
            current_range = seg_range
        current.append(text)
        current_end = end
        token_total += count

    if current:
        chunk_text = "\n".join(current).rstrip()
        h = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        t_count = _count_tokens(chunk_text)
        chunks.append(CodeChunk(current_start, current_end, chunk_text, h, t_count))
    return chunks


def chunk_file(
    path: Path,
    max_tokens: int,
    *,
    line_ranges: List[tuple[int, int]] | None = None,
) -> List[CodeChunk]:
    """Return token limited ``CodeChunk`` objects for ``path``.

    ``line_ranges`` are forwarded to :func:`split_into_chunks` to allow callers to
    align chunks with specific line boundaries.
    """

    source = path.read_text()
    return split_into_chunks(source, max_tokens, line_ranges=line_ranges)


def _split_by_lines(lines: List[str], start: int, limit: int) -> List[CodeChunk]:
    out: List[CodeChunk] = []
    buf: List[str] = []
    buf_start = start
    for line in lines:
        tentative = "\n".join(buf + [line])
        if _count_tokens(tentative) <= limit or not buf:
            buf.append(line)
            continue
        text = "\n".join(buf).rstrip()
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        out.append(CodeChunk(buf_start, buf_start + len(buf) - 1, text, h, _count_tokens(text)))
        buf_start += len(buf)
        buf = [line]
    if buf:
        text = "\n".join(buf).rstrip()
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        out.append(CodeChunk(buf_start, buf_start + len(buf) - 1, text, h, _count_tokens(text)))
    return out


def summarize_snippet(
    text: str,
    llm: LLMClient | None = None,
    *,
    context_builder: "ContextBuilder",
) -> str:
    """Return a short summary for ``text`` using available helpers with caching."""

    if context_builder is None:
        raise ValueError("context_builder is required")

    text = text.strip()
    if not text:
        return ""

    digest = _hash_snippet(text)
    cached = _load_snippet_summary(digest)
    if cached is not None:
        return cached

    summary = ""
    try:  # pragma: no cover - optional dependency
        from micro_models.diff_summarizer import summarize_diff as _summ

        result = _summ("", text)
        if result:
            summary = result
    except Exception:
        pass

    if not summary and llm is not None:
        try:
            prompt = context_builder.build_prompt(
                text,
                intent={
                    "instruction": "Summarise the following code snippet in one sentence.",
                },
            )
        except Exception as exc:
            if isinstance(exc, PromptBuildError):
                raise
            handle_failure("failed to build snippet summary prompt", exc)
        else:
            try:
                result = llm.generate(prompt, context_builder=context_builder)
                if getattr(result, "text", "").strip():
                    summary = result.text.strip()
            except Exception:
                pass

    if not summary:
        try:
            import ast
            import io
            import tokenize

            tree = ast.parse(text)
            lines = text.splitlines()
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    summary = lines[node.lineno - 1].strip()[:80]
                    break
        except Exception:
            try:
                for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                    if tok.type == tokenize.NAME and tok.string in {"def", "class"}:
                        summary = tok.line.strip()[:80]
                        break
            except Exception:
                pass

    if not summary:
        for line in text.splitlines():
            line = line.strip()
            if line:
                summary = line[:80]
                break

    if summary:
        _store_snippet_summary(digest, summary)
    return summary


# Backwards compatibility alias
summarize_code = summarize_snippet


_SETTINGS = SandboxSettings() if SandboxSettings else None

# Global cache instance used by :func:`get_chunk_summaries`.  The directory can
# be overridden by reassigning ``CHUNK_CACHE`` or by providing a custom cache
# instance to :func:`get_chunk_summaries`.
CHUNK_CACHE = ChunkSummaryCache(
    (_SETTINGS.chunk_summary_cache_dir if _SETTINGS else SNIPPET_CACHE_DIR)
)

# Per-path locks to avoid duplicate work when multiple threads request summaries
# simultaneously for the same file.
_CACHE_LOCKS: Dict[str, threading.Lock] = {}


def get_chunk_summaries(
    path: Path,
    max_tokens: int,
    llm: LLMClient | None = None,
    *,
    cache: ChunkSummaryCache | None = None,
    context_builder: "ContextBuilder",
) -> List[Dict[str, str]]:
    """Return cached summaries for ``path`` split into ``max_tokens`` chunks."""

    cache_obj = cache or CHUNK_CACHE
    path_hash = cache_obj.hash_path(path)
    cached = cache_obj.get(path_hash)
    if cached:
        return list(cached.get("summaries", []))

    lock = _CACHE_LOCKS.setdefault(path_hash, threading.Lock())
    with lock:
        cached = cache_obj.get(path_hash)
        if cached:
            return list(cached.get("summaries", []))

        chunks = chunk_file(path, max_tokens)
        summaries: List[Dict[str, str]] = []
        for ch in chunks:
            summary = summarize_code(ch.text, llm, context_builder=context_builder)
            summaries.append(
                {
                    "start_line": ch.start_line,
                    "end_line": ch.end_line,
                    "hash": ch.hash,
                    "summary": summary,
                }
            )

        cache_obj.set(path_hash, summaries)
        return summaries


__all__ = [
    "CodeChunk",
    "split_into_chunks",
    "chunk_file",
    "summarize_snippet",
    "summarize_code",
    "get_chunk_summaries",
]
