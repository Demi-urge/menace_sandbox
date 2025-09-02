from __future__ import annotations

"""Persistent store for failure fingerprints with similarity search."""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

from vector_service import SharedVectorService
from vector_utils import cosine_similarity

try:  # pragma: no cover - sandbox settings may be unavailable
    from sandbox_settings import SandboxSettings
except Exception:  # pragma: no cover - fall back to env vars
    SandboxSettings = None  # type: ignore


@dataclass
class FailureFingerprint:
    """Captured details of a failure for later matching."""

    filename: str
    function: str
    error_message: str
    stack_trace: str
    prompt: str
    embedding: List[float] = field(default_factory=list)


class FailureFingerprintStore:
    """Store and query :class:`FailureFingerprint` records."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        similarity_threshold: float | None = None,
        vector_service: SharedVectorService | None = None,
    ) -> None:
        settings = None
        if SandboxSettings is not None:
            try:  # pragma: no cover - settings optional
                settings = SandboxSettings()
            except Exception:  # pragma: no cover
                settings = None
        self.path = Path(
            path
            or (
                settings.failure_fingerprint_path
                if settings
                else os.getenv(
                    "FAILURE_FINGERPRINT_PATH",
                    "failure_fingerprints.jsonl",
                )
            )
        )
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else (
                settings.fingerprint_similarity_threshold
                if settings
                else float(os.getenv("FINGERPRINT_SIMILARITY_THRESHOLD", "0.8"))
            )
        )
        self.vector_service = vector_service or SharedVectorService()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, FailureFingerprint] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:  # pragma: no cover - best effort
                        continue
                    fid = data.pop("id", None)
                    if fid is None:
                        continue
                    try:
                        self._cache[fid] = FailureFingerprint(**data)
                    except TypeError:  # pragma: no cover - malformed entry
                        continue

    # ------------------------------------------------------------------ utils
    def _id_for(self, fingerprint: FailureFingerprint) -> str:
        return f"{fingerprint.filename}:{fingerprint.function}:{hash(fingerprint.stack_trace)}"

    def _ensure_embedding(self, fingerprint: FailureFingerprint) -> None:
        if not fingerprint.embedding:
            fingerprint.embedding = self.vector_service.vectorise(
                "text", {"text": fingerprint.stack_trace}
            )

    # ----------------------------------------------------------------- public
    def log(self, fingerprint: FailureFingerprint) -> None:
        """Append ``fingerprint`` to the log and index its embedding."""

        self._ensure_embedding(fingerprint)
        record_id = self._id_for(fingerprint)
        data = asdict(fingerprint)
        data["id"] = record_id
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(data) + "\n")
        self.vector_service.vector_store.add(
            "failure_fingerprint",
            record_id,
            fingerprint.embedding,
            metadata={"filename": fingerprint.filename, "function": fingerprint.function},
        )
        self._cache[record_id] = fingerprint

    def find_similar(
        self,
        fingerprint: FailureFingerprint,
        threshold: float | None = None,
    ) -> List[FailureFingerprint]:
        """Return fingerprints with cosine similarity >= ``threshold``."""

        self._ensure_embedding(fingerprint)
        thresh = threshold if threshold is not None else self.similarity_threshold
        matches: List[FailureFingerprint] = []
        for record_id, _ in self.vector_service.vector_store.query(
            fingerprint.embedding, top_k=5
        ):
            existing = self._cache.get(record_id)
            if not existing:
                continue
            sim = cosine_similarity(fingerprint.embedding, existing.embedding)
            if sim >= thresh:
                matches.append(existing)
        return matches

    def penalize_prompt(self, prompt: str) -> float:
        """Return the highest similarity of ``prompt`` to any stored failure."""

        embedding = self.vector_service.vectorise("text", {"text": prompt})
        best = 0.0
        for record_id, _ in self.vector_service.vector_store.query(embedding, top_k=5):
            existing = self._cache.get(record_id)
            if not existing:
                continue
            sim = cosine_similarity(embedding, existing.embedding)
            if sim > best:
                best = sim
        return best


__all__ = ["FailureFingerprint", "FailureFingerprintStore"]
