from __future__ import annotations

"""Utilities for splitting code into token-limited chunks respecting AST boundaries."""

from typing import List
import ast

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
            segment = "\n".join(lines[prev_end - 1 : start - 1]).rstrip()
            if segment:
                chunks.extend(_split_by_line(segment, max_tokens))
        block = "\n".join(lines[start - 1 : end]).rstrip()
        if block:
            chunks.extend(_split_by_line(block, max_tokens))
        prev_end = end + 1
    if prev_end <= len(lines):
        segment = "\n".join(lines[prev_end - 1 :]).rstrip()
        if segment:
            chunks.extend(_split_by_line(segment, max_tokens))
    return chunks


__all__ = ["split_into_chunks"]
