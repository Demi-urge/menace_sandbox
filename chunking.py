from __future__ import annotations

"""Code chunking utilities with token-aware grouping and summarisation."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, TYPE_CHECKING, Dict

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from llm_interface import LLMClient
import ast
import hashlib
import json

try:  # Optional dependency for accurate token counting
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - tiktoken may be missing
    tiktoken = None  # type: ignore

_ENCODER = None
if tiktoken is not None:  # pragma: no branch - simple import logic
    try:  # pragma: no cover - defensive
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - encoder creation failed
        _ENCODER = None


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


def split_into_chunks(code: str, max_tokens: int) -> List[CodeChunk]:
    """Split ``code`` into :class:`CodeChunk` objects under ``max_tokens`` tokens."""

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

    chunks: List[CodeChunk] = []
    current: List[str] = []
    current_start = 1
    current_end = 1
    token_total = 0

    for start, end, text in segments:
        count = _count_tokens(text)
        if count > max_tokens:
            if current:
                chunk_text = "\n".join(current).rstrip()
                h = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                t_count = _count_tokens(chunk_text)
                chunks.append(
                    CodeChunk(current_start, current_end, chunk_text, h, t_count)
                )
                current = []
                token_total = 0
            chunks.extend(
                _split_by_lines(text.splitlines(), start, max_tokens)
            )
            current_start = end + 1
            current_end = end
            continue
        if current and token_total + count > max_tokens:
            chunk_text = "\n".join(current).rstrip()
            h = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            t_count = _count_tokens(chunk_text)
            chunks.append(CodeChunk(current_start, current_end, chunk_text, h, t_count))
            current = [text]
            current_start = start
            current_end = end
            token_total = count
        else:
            if not current:
                current_start = start
            current.append(text)
            current_end = end
            token_total += count

    if current:
        chunk_text = "\n".join(current).rstrip()
        h = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        t_count = _count_tokens(chunk_text)
        chunks.append(CodeChunk(current_start, current_end, chunk_text, h, t_count))
    return chunks


def chunk_file(path: Path, max_tokens: int) -> List[CodeChunk]:
    """Return token limited ``CodeChunk`` objects for ``path``."""

    source = path.read_text()
    return split_into_chunks(source, max_tokens)


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


def summarize_code(text: str, llm: LLMClient | None = None) -> str:
    """Return a short summary for ``text`` using available helpers."""

    try:  # pragma: no cover - optional dependency
        from micro_models.diff_summarizer import summarize_diff as _summ

        result = _summ("", text)
        if result:
            return result
    except Exception:
        pass

    if llm is not None:
        from prompt_types import Prompt

        prompt = Prompt(
            system="Summarise the following code snippet in one sentence.",
            user=text,
        )
        try:  # pragma: no cover - llm failures
            result = llm.generate(prompt)
            if result.text.strip():
                return result.text.strip()
        except Exception:
            pass

    try:
        import ast
        import io
        import tokenize

        tree = ast.parse(text)
        lines = text.splitlines()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return lines[node.lineno - 1].strip()[:80]
    except Exception:
        try:
            for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                if tok.type == tokenize.NAME and tok.string in {"def", "class"}:
                    return tok.line.strip()[:80]
        except Exception:
            pass

    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            return line[:80]
    return ""


CACHE_DIR = Path("chunk_summary_cache")
CACHE_DIR.mkdir(exist_ok=True)


def get_chunk_summaries(
    path: Path, max_tokens: int, llm: LLMClient | None = None
) -> List[Dict[str, str]]:
    """Return cached summaries for ``path`` split into ``max_tokens`` chunks."""

    source = path.read_text()
    file_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"{file_hash}.json"

    cached: Dict[str, Dict[str, str]] = {}
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            if data.get("hash") == file_hash:
                for c in data.get("chunks", []):
                    if isinstance(c, dict) and "hash" in c:
                        cached[c["hash"]] = c
        except Exception:  # pragma: no cover - corrupted cache
            pass

    chunks = chunk_file(path, max_tokens)
    summaries: List[Dict[str, str]] = []
    updated = False
    for ch in chunks:
        summary = summarize_code(ch.text, llm)
        entry = cached.get(ch.hash)
        if entry is None or entry.get("summary") != summary:
            entry = {"hash": ch.hash, "summary": summary}
            cached[ch.hash] = entry
            updated = True
        summaries.append({"hash": ch.hash, "summary": summary})

    if updated or not cache_file.exists():
        cache_file.write_text(
            json.dumps({"hash": file_hash, "chunks": summaries}, indent=2, sort_keys=True)
        )
    return summaries


__all__ = [
    "CodeChunk",
    "split_into_chunks",
    "chunk_file",
    "summarize_code",
    "get_chunk_summaries",
]
