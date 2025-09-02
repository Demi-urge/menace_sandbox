from __future__ import annotations

"""Code chunking utilities with token-aware grouping and summarisation."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_interface import LLMClient
import ast
import hashlib

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


@dataclass
class CodeChunk:
    """Representation of a contiguous code block."""

    start_line: int
    end_line: int
    text: str
    hash: str


def chunk_file(path: Path, max_tokens: int) -> List[CodeChunk]:
    """Return token limited ``CodeChunk`` objects for ``path``.

    The file is parsed using :mod:`ast` and split at top-level ``FunctionDef``
    and ``ClassDef`` boundaries. Nodes are grouped together while ensuring the
    accumulated token count does not exceed ``max_tokens``. Individual nodes
    larger than ``max_tokens`` are emitted as their own chunk.
    """

    source = path.read_text()
    lines = source.splitlines()
    try:
        module = ast.parse(source)
    except SyntaxError:  # pragma: no cover - syntax error fallback
        text = source.rstrip()
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return [CodeChunk(1, len(lines), text, h)]

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
        if current and token_total + count > max_tokens:
            chunk_text = "\n".join(current).rstrip()
            h = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            chunks.append(CodeChunk(current_start, current_end, chunk_text, h))
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
        chunks.append(CodeChunk(current_start, current_end, chunk_text, h))
    return chunks


def summarize_code(text: str, llm: LLMClient | None) -> str:
    """Return a short summary for ``text`` using ``llm`` when available."""

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
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            return line[:80]
    return ""


__all__ = ["CodeChunk", "chunk_file", "summarize_code"]
