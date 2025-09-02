from __future__ import annotations

"""Persistent store for failure fingerprints with similarity search."""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

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
    embedding_metadata: Dict[str, Any] = field(default_factory=dict)


class FailureFingerprintStore:
    """Store and query :class:`FailureFingerprint` records."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        similarity_threshold: float | None = None,
        vector_service: SharedVectorService | None = None,
        compact_interval: int | None = 100,
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
        self.compact_interval = compact_interval or 0
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
            vec = self.vector_service.vectorise("text", {"text": fingerprint.stack_trace})
            fingerprint.embedding = vec
            embedder = getattr(self.vector_service, "text_embedder", None)
            model_name = None
            if embedder is not None:
                model_name = getattr(embedder, "name_or_path", embedder.__class__.__name__)
            fingerprint.embedding_metadata = {"model": model_name, "dim": len(vec)}

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
            metadata={
                "filename": fingerprint.filename,
                "function": fingerprint.function,
                "embedding_meta": fingerprint.embedding_metadata,
            },
        )
        self._cache[record_id] = fingerprint
        if self.compact_interval and len(self._cache) % self.compact_interval == 0:
            self.compact()

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

    # -------------------------------------------------------------- maintenance
    def compact(self) -> None:
        """Rewrite JSONL and rebuild vector index from current cache."""

        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            for record_id, fp in self._cache.items():
                data = asdict(fp)
                data["id"] = record_id
                fh.write(json.dumps(data) + "\n")
        tmp_path.replace(self.path)

        store = self.vector_service.vector_store
        if store is None:
            return
        dim = 0
        if self._cache:
            dim = len(next(iter(self._cache.values())).embedding)
        try:
            path = Path(getattr(store, "path"))
            meta_path = Path(getattr(store, "meta_path", path.with_suffix(".meta.json")))
            if path.exists():
                path.unlink()
            if meta_path.exists():
                meta_path.unlink()
            kwargs: Dict[str, Any] = {}
            if hasattr(store, "metric"):
                kwargs["metric"] = getattr(store, "metric")
            new_store = type(store)(dim, path, **kwargs)
            self.vector_service.vector_store = new_store
            for record_id, fp in self._cache.items():
                new_store.add(
                    "failure_fingerprint",
                    record_id,
                    fp.embedding,
                    metadata={
                        "filename": fp.filename,
                        "function": fp.function,
                        "embedding_meta": fp.embedding_metadata,
                    },
                )
        except Exception:  # pragma: no cover - best effort
            store.load()
            for record_id, fp in self._cache.items():
                try:
                    store.add(
                        "failure_fingerprint",
                        record_id,
                        fp.embedding,
                        metadata={
                            "filename": fp.filename,
                            "function": fp.function,
                            "embedding_meta": fp.embedding_metadata,
                        },
                    )
                except Exception:
                    continue


__all__ = ["FailureFingerprint", "FailureFingerprintStore"]
