from __future__ import annotations

"""Utilities for logging and querying failure fingerprints."""

import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from time import time
from typing import Any, Dict, Iterable, List

from error_vectorizer import ErrorVectorizer
from vector_utils import cosine_similarity
# ``vector_service`` is optional; fall back to a minimal stub when unavailable
try:  # pragma: no cover - optional dependency
    from vector_service import EmbeddableDBMixin  # type: ignore
except Exception:  # pragma: no cover - fallback for tests
    class EmbeddableDBMixin:  # type: ignore
        def __init__(self, *a, **k) -> None:  # noqa: D401 - trivial
            pass

        def try_add_embedding(self, *a, **k) -> None:
            pass

        def search_by_vector(self, *a, **k):  # noqa: D401 - trivial
            return []

        _metadata: Dict[str, Dict[str, Any]] = {}


_VECTOR = ErrorVectorizer()


class _FingerprintIndex(EmbeddableDBMixin):
    """Minimal :class:`EmbeddableDBMixin` implementation for fingerprints."""

    def __init__(self, index_path: Path, metadata_path: Path) -> None:
        EmbeddableDBMixin.__init__(
            self,
            index_path=index_path,
            metadata_path=metadata_path,
        )

    def vector(self, record: Dict[str, Any]) -> List[float]:  # pragma: no cover - simple
        return list(record.get("embedding", []))

    def license_text(self, record: Dict[str, Any]) -> str | None:  # pragma: no cover - simple
        return record.get("stack_trace")


@dataclass
class FailureFingerprint:
    """Represents a unique failure event for later similarity search."""

    filename: str
    function_name: str
    error_message: str
    stack_trace: str
    prompt_text: str
    cluster_id: int | None = None
    hash: str = field(init=False)
    embedding: List[float] = field(default_factory=list)
    embedding_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time())
    count: int = 1
    target_region: str | None = None
    escalation_level: int = 0

    # ``function`` is used by ``FailureFingerprintStore``; keep ``function_name`` for
    # backwards compatibility with existing callers.
    @property
    def function(self) -> str:  # pragma: no cover - trivial
        return self.function_name

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
        fp = cls(
            filename=filename,
            function_name=function_name,
            error_message=error_message,
            stack_trace=stack_trace,
            prompt_text=prompt_text,
            embedding=vec,
        )
        return fp

    def __post_init__(self) -> None:
        # ``hash`` uniquely identifies the stack trace for quick comparisons.
        self.hash = hashlib.sha256(self.stack_trace.encode("utf-8")).hexdigest()


def _index_for(path: Path) -> _FingerprintIndex:
    base = path.with_suffix("")
    index_path = base.with_suffix(path.suffix + ".ann")
    meta_path = base.with_suffix(path.suffix + ".ann.json")
    return _FingerprintIndex(index_path=index_path, metadata_path=meta_path)


def log_fingerprint(
    fingerprint: FailureFingerprint,
    path: str | Path = "failure_fingerprints.jsonl",
) -> None:
    """Append ``fingerprint`` to ``path`` and update the embedding index."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(fingerprint)) + "\n")

    record_id = f"{fingerprint.filename}:{fingerprint.function_name}:{fingerprint.timestamp}"
    index = _index_for(file_path)
    index.try_add_embedding(record_id, asdict(fingerprint), "failure_fingerprint")


def similar_fingerprints(
    fingerprint: FailureFingerprint,
    threshold: float,
    *,
    path: str | Path = "failure_fingerprints.jsonl",
) -> List[FailureFingerprint]:
    """Return fingerprints similar to ``fingerprint`` by cosine similarity."""

    file_path = Path(path)
    if not file_path.exists():
        return []
    index = _index_for(file_path)
    matches: List[tuple[float, FailureFingerprint]] = []
    for record_id, _dist in index.search_by_vector(fingerprint.embedding, top_k=10):
        meta = index._metadata.get(str(record_id), {})
        data = meta.get("record")
        if not data:
            continue
        sim = cosine_similarity(fingerprint.embedding, data.get("embedding", []))
        if sim >= threshold:
            try:
                fp = FailureFingerprint(**data)
            except TypeError:
                continue
            matches.append((sim, fp))
    matches.sort(key=lambda x: x[0], reverse=True)
    return [fp for _, fp in matches]


def find_similar(
    trace_embedding: Iterable[float],
    threshold: float,
    *,
    path: str | Path = "failure_fingerprints.jsonl",
) -> List[FailureFingerprint]:
    """Backward compatible wrapper for :func:`similar_fingerprints`."""

    fp = FailureFingerprint("", "", "", "", "", embedding=list(trace_embedding))
    return similar_fingerprints(fp, threshold, path=path)


__all__ = [
    "FailureFingerprint",
    "log_fingerprint",
    "similar_fingerprints",
    "find_similar",
]
