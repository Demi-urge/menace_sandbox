"""Utilities for clustering and querying module intents.

This module provides :class:`IntentClusterer` which builds intent vectors for
Python modules, clusters them and exposes a simple similarity search utility.

The implementation purposefully keeps dependencies light so it can operate in
minimal environments.  External libraries such as ``scikit-learn`` are used
when available with graceful fallbacks otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

from intent_db import IntentDB
from intent_vectorizer import IntentVectorizer
from universal_retriever import UniversalRetriever
from vector_service import SharedVectorService

try:  # pragma: no cover - optional heavy dependency
    from sklearn.cluster import KMeans  # type: ignore
except Exception:  # pragma: no cover - fall back to simple implementation
    from knowledge_graph import _SimpleKMeans as KMeans  # type: ignore


@dataclass
class IntentClusterer:
    """Index modules, cluster intents and search related modules."""

    intent_db: IntentDB = field(default_factory=IntentDB)
    vector_service: SharedVectorService = field(default_factory=SharedVectorService)
    retriever: UniversalRetriever = field(default_factory=UniversalRetriever)
    vectorizer: IntentVectorizer = field(default_factory=IntentVectorizer)

    # Local caches
    _vectors: Dict[str, List[float]] = field(default_factory=dict)
    module_ids: Dict[str, int] = field(default_factory=dict)
    clusters: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure the retriever knows about the intent database so that
        # ``retrieve`` calls can access it.  Errors are ignored as callers may
        # pre-register the database on a custom retriever instance.
        try:  # pragma: no cover - best effort
            self.retriever.register_db("intent", self.intent_db, ("id",))
        except Exception:
            pass

    # ------------------------------------------------------------------
    def index_modules(self, paths: Iterable[Path]) -> None:
        """Parse and embed Python modules listed in ``paths``.

        Each module is processed with :class:`IntentVectorizer` to extract
        high level intent text.  The text is then embedded and persisted using
        :meth:`SharedVectorService.vectorise_and_store` with ``kind="intent"``.
        A mapping of module path to the resulting vector identifier is stored
        in :attr:`module_ids` while the in-memory vector is cached for future
        clustering.
        """

        for path in paths:
            try:
                text = self.vectorizer.bundle(path)
            except Exception:  # pragma: no cover - parsing failure
                text = ""
            if not text:
                continue

            # Record in the database to obtain a stable identifier
            module_id = self.intent_db.add(str(path))

            vec = self.vector_service.vectorise_and_store(
                "intent",
                str(module_id),
                {"path": str(path), "text": text},
                origin_db="intent",
                metadata={"path": str(path)},
            )
            self._vectors[str(path)] = vec
            self.module_ids[str(path)] = module_id

    # ------------------------------------------------------------------
    def cluster_intents(self, n_clusters: int) -> Dict[str, int]:
        """Group indexed modules into ``n_clusters`` functional clusters."""

        if not self._vectors:
            return {}

        vectors = list(self._vectors.values())
        km = KMeans(n_clusters=n_clusters)
        km.fit(vectors)
        # ``predict`` is available on both scikit-learn and the simple fallback
        labels = km.predict(vectors)
        self.clusters = {
            path: int(label) for path, label in zip(self._vectors.keys(), labels)
        }
        return dict(self.clusters)

    # ------------------------------------------------------------------
    def find_modules_related_to(
        self, prompt: str, top_k: int = 5
    ) -> List[Dict[str, float]]:
        """Return modules most relevant to ``prompt``.

        The ``prompt`` is embedded using :class:`SharedVectorService` and the
        resulting vector is queried against :class:`IntentDB` via the injected
        :class:`UniversalRetriever`.  A list of dictionaries containing the
        module path and similarity score is returned, ordered from most to
        least similar.
        """

        vec = self.vector_service.vectorise("text", {"text": prompt})
        if not vec:
            return []

        try:
            res, *_ = self.retriever.retrieve(vec, top_k=top_k, db_names=["intent"])
            hits = list(res)
        except Exception:  # pragma: no cover - compatibility fallback
            try:
                search = getattr(self.retriever, "search")
                hits = list(search(vec, top_k=top_k))
            except Exception:
                return []

        results: List[Dict[str, float]] = []
        for hit in hits:
            meta = getattr(hit, "metadata", hit)
            if isinstance(meta, dict):
                path = meta.get("path") or meta.get("record_id") or meta.get("id")
                score = meta.get("score") or getattr(hit, "score", 0.0)
            else:
                path = getattr(meta, "path", None)
                score = getattr(meta, "score", 0.0)
            if path is not None:
                results.append({"path": str(path), "score": float(score)})
        return results[:top_k]


__all__ = ["IntentClusterer"]

