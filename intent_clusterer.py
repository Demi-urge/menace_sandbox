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
        high level intent text.  The text is then embedded and persisted in
        :class:`IntentDB` so that similarity searches can be performed later.
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

            self.intent_db.add_embedding(module_id, {"path": str(path)}, "module")
            vec = self.intent_db.get_vector(module_id) or []
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

        The ``prompt`` is embedded using :class:`IntentDB`'s encoding model and
        the resulting vector is queried directly against the stored embeddings.
        A list of dictionaries containing the module path (or synergy cluster
        identifier) and similarity score is returned, ordered from most to
        least similar.
        """

        vec = self.intent_db.encode_text(prompt)
        if not vec:
            return []
        hits = self.intent_db.search_by_vector(vec, top_k)
        results: List[Dict[str, float]] = []
        for rid, dist in hits:
            row = self.intent_db.conn.execute(
                "SELECT path FROM intent_modules WHERE id=?", (rid,)
            ).fetchone()
            if row:
                score = 1.0 / (1.0 + float(dist))
                results.append({"path": str(row["path"]), "score": score})
        return results[:top_k]


__all__ = ["IntentClusterer"]

