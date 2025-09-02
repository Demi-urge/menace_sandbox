from __future__ import annotations

"""Utilities for logging and querying failure fingerprints."""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from time import time
from typing import Iterable, List

from error_vectorizer import ErrorVectorizer
from vector_utils import persist_embedding, cosine_similarity


_VECTOR = ErrorVectorizer()


@dataclass
class FailureFingerprint:
    """Represents a unique failure event for later similarity search."""

    filename: str
    function_name: str
    stack_trace: str
    error_message: str
    prompt_text: str
    timestamp: float = field(default_factory=lambda: time())
    embedding: List[float] = field(default_factory=list)

    @classmethod
    def from_failure(
        cls,
        filename: str,
        function_name: str,
        stack_trace: str,
        error_message: str,
        prompt_text: str,
    ) -> "FailureFingerprint":
        """Create a fingerprint with an embedding derived from ``stack_trace``."""

        vec = _VECTOR.transform({"stack_trace": stack_trace})
        return cls(
            filename=filename,
            function_name=function_name,
            stack_trace=stack_trace,
            error_message=error_message,
            prompt_text=prompt_text,
            embedding=vec,
        )


def _embedding_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".embeddings.jsonl")


def log_fingerprint(
    fingerprint: FailureFingerprint,
    path: str | Path = "failure_fingerprints.jsonl",
) -> None:
    """Append ``fingerprint`` to the fingerprint log and persist its embedding."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(fingerprint)) + "\n")
    record_id = f"{fingerprint.filename}:{fingerprint.function_name}:{fingerprint.timestamp}"
    persist_embedding(
        "failure_fingerprint",
        record_id,
        fingerprint.embedding,
        path=_embedding_path(file_path),
        metadata={"filename": fingerprint.filename, "function": fingerprint.function_name},
    )


def find_similar(
    trace_embedding: Iterable[float],
    threshold: float,
    *,
    path: str | Path = "failure_fingerprints.jsonl",
) -> List[FailureFingerprint]:
    """Return fingerprints with cosine similarity >= ``threshold`` to ``trace_embedding``."""

    file_path = Path(path)
    if not file_path.exists():
        return []
    matches: List[tuple[float, FailureFingerprint]] = []
    with file_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            emb = data.get("embedding") or []
            sim = cosine_similarity(trace_embedding, emb)
            if sim >= threshold:
                try:
                    fp = FailureFingerprint(**data)
                except TypeError:
                    continue
                matches.append((sim, fp))
    matches.sort(key=lambda x: x[0], reverse=True)
    return [fp for _, fp in matches]


__all__ = ["FailureFingerprint", "log_fingerprint", "find_similar"]
