from __future__ import annotations

"""Persistent store for failure fingerprints with similarity search."""

import json
import os
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

from failure_fingerprint import FailureFingerprint
from vector_service import SharedVectorService
from vector_utils import cosine_similarity

try:  # pragma: no cover - sandbox settings may be unavailable
    from sandbox_settings import SandboxSettings
except Exception:  # pragma: no cover - fall back to env vars
    SandboxSettings = None  # type: ignore


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
        self._similarity_history: List[float] = []
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
                    hash_val = data.pop("hash", "")
                    try:
                        fp = FailureFingerprint(**data)
                        fp.hash = hash_val or fp.hash
                        self._cache[fid] = fp
                    except TypeError:  # pragma: no cover - malformed entry
                        continue

        # cluster bookkeeping
        self._clusters: Dict[int, List[str]] = {}
        self._cluster_centroids: Dict[int, List[float]] = {}
        self._next_cluster_id = 1
        if self._cache:
            self.cluster_fingerprints()

    # ------------------------------------------------------------------ utils
    def _id_for(self, fingerprint: FailureFingerprint) -> str:
        return f"{fingerprint.filename}:{fingerprint.function}:{fingerprint.hash}"

    def _ensure_embedding(self, fingerprint: FailureFingerprint) -> None:
        if not fingerprint.embedding:
            try:
                vec = self.vector_service.vectorise(
                    "text", {"text": fingerprint.stack_trace}
                )
            except Exception:  # pragma: no cover - best effort
                vec = []
            if not vec:
                # Simple fallback embedding based on character codes
                vec = [float(ord(c)) for c in fingerprint.stack_trace[:32]]
            fingerprint.embedding = vec
            embedder = getattr(self.vector_service, "text_embedder", None)
            model_name = None
            if embedder is not None:
                model_name = getattr(embedder, "name_or_path", embedder.__class__.__name__)
            fingerprint.embedding_metadata = {"model": model_name, "dim": len(vec)}

    def _assign_cluster(self, record_id: str, fingerprint: FailureFingerprint) -> int:
        """Assign ``fingerprint`` to the closest cluster and update centroids."""

        best_id = None
        best_sim = -1.0
        for cid, centroid in self._cluster_centroids.items():
            sim = cosine_similarity(fingerprint.embedding, centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = cid

        if best_sim >= 0:
            self._similarity_history.append(best_sim)

        thresh = self.similarity_threshold
        if best_id is None or best_sim < thresh:
            cid = self._next_cluster_id
            self._next_cluster_id += 1
            self._clusters[cid] = [record_id]
            self._cluster_centroids[cid] = list(fingerprint.embedding)
            fingerprint.cluster_id = cid
            return cid

        self._clusters.setdefault(best_id, []).append(record_id)
        centroid = self._cluster_centroids[best_id]
        n = len(self._clusters[best_id])
        self._cluster_centroids[best_id] = [
            (c * (n - 1) + v) / n for c, v in zip(centroid, fingerprint.embedding)
        ]
        fingerprint.cluster_id = best_id
        return best_id

    def _persist_update(self, record_id: str, fingerprint: FailureFingerprint) -> None:
        """Rewrite the JSONL entry for ``record_id`` with updated data."""

        tmp_path = self.path.with_suffix(".tmp")
        data = asdict(fingerprint)
        data["id"] = record_id
        new_line = json.dumps(data) + "\n"
        try:
            with self.path.open("r", encoding="utf-8") as src, tmp_path.open(
                "w", encoding="utf-8"
            ) as dst:
                for line in src:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        dst.write(line)
                        continue
                    if obj.get("id") == record_id:
                        dst.write(new_line)
                    else:
                        dst.write(line)
        except FileNotFoundError:  # pragma: no cover - best effort
            with tmp_path.open("w", encoding="utf-8") as dst:
                dst.write(new_line)
        tmp_path.replace(self.path)

    # ----------------------------------------------------------------- public
    def cluster_fingerprints(self) -> None:
        """Cluster all fingerprints currently in the store."""

        self._clusters.clear()
        self._cluster_centroids.clear()
        self._next_cluster_id = 1
        for record_id, fp in self._cache.items():
            self._ensure_embedding(fp)
            self._assign_cluster(record_id, fp)

    def cluster_stats(self) -> Dict[int, Dict[str, Any]]:
        """Return basic statistics about all known clusters.

        The returned mapping contains one entry per cluster ID with the total
        number of fingerprints assigned to that cluster (respecting the
        ``count`` field of each fingerprint) and a representative example
        fingerprint.
        """

        stats: Dict[int, Dict[str, Any]] = {}
        for cid, ids in self._clusters.items():
            example = None
            size = 0
            for rid in ids:
                fp = self._cache.get(rid)
                if fp is None:
                    continue
                if example is None:
                    example = fp
                size += fp.count
            if example is None:
                continue
            stats[cid] = {"size": size, "example": example}
        return stats

    def similarity_stats(self, window: int = 50) -> tuple[float, float]:
        """Return moving average and deviation of recent similarities."""

        hist = self._similarity_history[-window:]
        if not hist:
            return 0.0, 0.0
        if len(hist) == 1:
            return hist[0], 0.0
        avg = mean(hist)
        dev = pstdev(hist)
        return avg, dev

    def adaptive_threshold(self, window: int = 50, multiplier: float = 1.0) -> float:
        """Adaptive similarity threshold based on history."""

        avg, dev = self.similarity_stats(window)
        if avg == 0.0 and dev == 0.0:
            return self.similarity_threshold
        return avg + multiplier * dev

    # ----------------------------------------------------------------- public
    def add(self, fingerprint: FailureFingerprint) -> None:
        """Append ``fingerprint`` to the log and index its embedding."""

        self._ensure_embedding(fingerprint)
        record_id = self._id_for(fingerprint)
        existing = self._cache.get(record_id)
        if existing is not None:
            existing.count += 1
            existing.timestamp = fingerprint.timestamp
            existing.embedding_metadata = fingerprint.embedding_metadata
            self._persist_update(record_id, existing)
            try:
                self.vector_service.vector_store.add(
                    "failure_fingerprint",
                    record_id,
                    existing.embedding,
                    metadata={
                        "filename": existing.filename,
                        "function": existing.function,
                        "embedding_meta": existing.embedding_metadata,
                        "count": existing.count,
                        "target_region": existing.target_region,
                        "escalation_level": existing.escalation_level,
                    },
                )
            except Exception:  # pragma: no cover - best effort
                pass
            return

        self._assign_cluster(record_id, fingerprint)
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
                "count": fingerprint.count,
                "target_region": fingerprint.target_region,
                "escalation_level": fingerprint.escalation_level,
            },
        )
        self._cache[record_id] = fingerprint
        if self.compact_interval and len(self._cache) % self.compact_interval == 0:
            self.compact()

    # Backwards compatibility
    def log(self, fingerprint: FailureFingerprint) -> None:  # pragma: no cover
        self.add(fingerprint)

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

    def get_cluster(self, cluster_id: int) -> List[FailureFingerprint]:
        """Return all fingerprints belonging to ``cluster_id``."""

        ids = self._clusters.get(cluster_id, [])
        return [self._cache[rid] for rid in ids if rid in self._cache]

    def cluster_for(self, fingerprint: FailureFingerprint) -> List[FailureFingerprint]:
        """Return all fingerprints that share a cluster with ``fingerprint``."""

        if fingerprint.cluster_id is None:
            return []
        return self.get_cluster(fingerprint.cluster_id)

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
                        "target_region": fp.target_region,
                        "escalation_level": fp.escalation_level,
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
                            "target_region": fp.target_region,
                            "escalation_level": fp.escalation_level,
                        },
                    )
                except Exception:
                    continue


__all__ = ["FailureFingerprint", "FailureFingerprintStore"]
