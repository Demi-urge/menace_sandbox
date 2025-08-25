"""Intent clustering utilities for Python modules.

This module defines :class:`IntentClusterer` which can index Python modules,
extract natural language signals and persist embedding vectors in a vector
database compatible with :class:`~universal_retriever.UniversalRetriever`.

Embeddings are produced via :func:`governed_embeddings.governed_embed` and are
stored using the lightweight :class:`EmbeddableDBMixin` infrastructure.  The
clusterer keeps an in‑memory mapping of module paths to cluster identifiers and
provides a simple semantic search helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Any, Sequence
import ast
import io
import tokenize

from governed_embeddings import governed_embed
from universal_retriever import UniversalRetriever
from embeddable_db_mixin import EmbeddableDBMixin
from math import sqrt

try:  # pragma: no cover - optional dependency
    from sklearn.cluster import KMeans  # type: ignore
except Exception:  # pragma: no cover - fallback used in tests
    from knowledge_graph import _SimpleKMeans as KMeans  # type: ignore


@dataclass
class IntentMatch:
    path: str | None
    similarity: float
    cluster_ids: List[int]


def extract_intent_text(path: Path) -> str:
    """Return text describing the intent of the module at ``path``.

    The helper collects module and function docstrings, function or class
    names and inline comments appearing directly above definitions.  The
    approach mirrors :class:`intent_vectorizer.IntentVectorizer` but is kept
    lightweight so it can be used independently of that component.
    """

    try:
        with tokenize.open(path) as fh:
            source = fh.read()
    except OSError:
        return ""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""

    docstrings: List[str] = []
    names: List[str] = []

    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        docstrings.append(mod_doc)

    comment_map: Dict[int, List[str]] = {}
    code_lines: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        toknum, tokval, start, _end, _line = tok
        lineno = start[0]
        if toknum == tokenize.COMMENT:
            if lineno not in code_lines:
                text = tokval.lstrip("#").strip()
                if "coding:" in text:
                    continue
                if text:
                    comment_map.setdefault(lineno, []).append(text)
        elif toknum not in {
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENDMARKER,
            tokenize.ENCODING,
        }:
            code_lines.add(lineno)

    comments: List[str] = []
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
            doc = ast.get_docstring(node)
            if doc:
                docstrings.append(doc)
            collected: List[str] = []
            line = node.lineno - 1
            while line > 0:
                if line in code_lines:
                    break
                if line in comment_map:
                    parts = [" ".join(c.split()) for c in comment_map[line]]
                    collected.insert(0, " ".join(parts))
                    line -= 1
                    continue
                if lines[line - 1].strip() == "":
                    line -= 1
                    continue
                break
            if collected:
                comments.append(" ".join(collected).strip())

    return "\n".join(docstrings + names + comments)


class ModuleVectorDB(EmbeddableDBMixin):
    """Minimal vector DB for module intent embeddings."""

    def __init__(self, *, index_path: str | Path = "intent_vectors.ann", metadata_path: str | Path = "intent_vectors.json") -> None:
        super().__init__(index_path=index_path, metadata_path=metadata_path)
        self._texts: Dict[int, str] = {}
        self._paths: Dict[int, str] = {}

    # ------------------------------------------------------------------
    def add_module(self, path: str, text: str) -> int:
        rid = len(self._texts) + 1
        self._texts[rid] = text
        self._paths[rid] = path
        self.add_embedding(rid, text, "module", source_id=path)
        return rid

    # ------------------------------------------------------------------
    def vector(self, record: Any) -> List[float]:
        text = record if isinstance(record, str) else str(record)
        vec = governed_embed(text)
        return [float(x) for x in vec] if vec else []

    # ------------------------------------------------------------------
    def iter_records(self):
        for rid, text in self._texts.items():
            yield rid, text, "module"

    # ------------------------------------------------------------------
    def license_text(self, record: Any) -> str | None:
        return record if isinstance(record, str) else None

    # ------------------------------------------------------------------
    def encode_text(self, text: str) -> List[float]:  # type: ignore[override]
        vec = governed_embed(text)
        return [float(x) for x in vec] if vec else []

    # ------------------------------------------------------------------
    def get_path(self, rid: int) -> str | None:
        return self._paths.get(rid)


@dataclass(init=False)
class IntentClusterer:
    """Index modules, cluster intents and perform semantic search."""

    retriever: Any
    db: ModuleVectorDB
    module_ids: Dict[str, int]
    vectors: Dict[str, List[float]]
    clusters: Dict[str, int]
    vector_service: Any | None

    def __init__(
        self,
        retriever: Any | None = None,
        db: ModuleVectorDB | None = None,
        *,
        intent_db: ModuleVectorDB | None = None,
        vector_service: Any | None = None,
    ) -> None:
        self.vector_service = vector_service
        self.retriever = (
            retriever
            or type(
                "DummyRetriever",
                (),
                {
                    "register_db": lambda *a, **k: None,
                    "search": lambda *a, **k: [],
                    "add_vector": lambda *a, **k: None,
                },
            )()
        )
        self.db = db or intent_db or ModuleVectorDB()
        self.module_ids = {}
        self.vectors = {}
        self.clusters = {}
        self.cluster_map = self.clusters  # backward compatibility alias
        self.__post_init__()

    def __post_init__(self) -> None:
        try:  # pragma: no cover - best effort registration
            self.retriever.register_db("intent", self.db, ("path",))
        except Exception:
            pass

    # ------------------------------------------------------------------
    def index_modules(self, paths: Iterable[Path]) -> None:
        """Extract and embed intent signals from ``paths``."""

        for path in paths:
            text = extract_intent_text(path)
            if not text:
                continue
            if hasattr(self.db, "add_module"):
                rid = self.db.add_module(str(path), text)
            else:  # fall back to IntentDB style API
                rid = self.db.add(str(path))
                try:
                    self.db.add_embedding(rid, {"path": str(path), "text": text}, "module")
                except Exception:
                    continue
            vec = self.db.get_vector(rid) or []
            if vec:
                self.module_ids[str(path)] = rid
                self.vectors[str(path)] = vec
                # Persist in retriever for cross-module search when available
                if hasattr(self.retriever, "add_vector"):
                    try:
                        self.retriever.add_vector(vec, {"path": str(path)})
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    def cluster_intents(self, n_clusters: int) -> Dict[str, int]:
        """Group indexed modules into ``n_clusters`` clusters."""

        if not self.vectors:
            return {}
        vectors = list(self.vectors.values())
        km = KMeans(n_clusters=n_clusters)
        km.fit(vectors)
        labels = km.predict(vectors) if hasattr(km, "predict") else km.labels_
        self.clusters = {p: int(l) for p, l in zip(self.vectors.keys(), labels)}
        # Update retriever metadata with assigned cluster identifiers
        if hasattr(self.retriever, "add_vector"):
            for path, cid in self.clusters.items():
                vec = self.vectors.get(path)
                if vec:
                    try:
                        self.retriever.add_vector(vec, {"path": path, "cluster_id": int(cid)})
                    except Exception:
                        continue
        return dict(self.clusters)

    # ------------------------------------------------------------------
    def get_cluster_intents(self, cluster_id: int) -> tuple[str, List[float]]:
        """Return placeholder text and vector for ``cluster_id``.

        This method is meant to be overridden or monkeypatched in tests.
        """

        return "", []

    # ------------------------------------------------------------------
    def query(
        self,
        text: str,
        top_k: int = 5,
        *,
        threshold: float = 0.0,
        include_clusters: bool = True,
    ) -> List[IntentMatch]:
        """Search retriever for modules relevant to ``text``."""

        vec = governed_embed(text)
        if not vec:
            return []
        norm = sqrt(sum(x * x for x in vec)) or 1.0
        vec = [x / norm for x in vec]
        hits = self.retriever.search(vec, top_k=top_k) if hasattr(self.retriever, "search") else []
        results: List[IntentMatch] = []
        for item in hits:
            meta = item.get("metadata", {})
            target_vec: Sequence[float] = item.get("vector", [])
            tnorm = sqrt(sum(x * x for x in target_vec)) or 1.0
            target_vec = [x / tnorm for x in target_vec]
            sim = sum(a * b for a, b in zip(vec, target_vec))
            path = meta.get("path")
            cid = meta.get("cluster_id")
            cluster_ids: List[int] = []
            if sim < threshold and cid is not None:
                _text, cvec = self.get_cluster_intents(int(cid))
                if cvec:
                    cnorm = sqrt(sum(x * x for x in cvec)) or 1.0
                    cvec = [x / cnorm for x in cvec]
                    sim = sum(a * b for a, b in zip(vec, cvec))
                    path = None
            if sim < threshold:
                continue
            if include_clusters and cid is not None:
                cluster_ids = [int(cid)]
            results.append(IntentMatch(path=path, similarity=sim, cluster_ids=cluster_ids))
        return results

    # ------------------------------------------------------------------
    def find_modules_related_to(self, prompt: str, top_k: int = 5) -> List[Dict[str, float]]:
        """Return modules most semantically similar to ``prompt``.

        The query text is embedded and searched using the configured
        :class:`~universal_retriever.UniversalRetriever` instance.  If the
        retriever is unavailable, the method falls back to the underlying
        vector database.
        """

        vec = self.db.encode_text(prompt)
        if not vec:
            return []

        # Prefer retriever-based search when available
        if hasattr(self.retriever, "search"):
            norm = sqrt(sum(x * x for x in vec)) or 1.0
            qvec = [x / norm for x in vec]
            try:
                hits = self.retriever.search(qvec, top_k=top_k) or []
            except Exception:
                hits = []
            results: List[Dict[str, float]] = []
            for item in hits:
                meta = item.get("metadata", {})
                path = meta.get("path")
                target_vec: Sequence[float] = item.get("vector", [])
                tnorm = sqrt(sum(x * x for x in target_vec)) or 1.0
                target_vec = [x / tnorm for x in target_vec]
                score = sum(a * b for a, b in zip(qvec, target_vec))
                if path:
                    results.append({"path": path, "score": score})
            if results:
                return results[:top_k]

        # Fallback: direct search on the vector database
        hits = []
        try:
            hits = self.db.search_by_vector(vec, top_k)
        except Exception:
            return []
        results: List[Dict[str, float]] = []
        for rid, dist in hits:
            path: str | None = None
            if hasattr(self.db, "get_path"):
                path = self.db.get_path(int(rid))
            elif hasattr(self.db, "conn"):
                try:
                    row = self.db.conn.execute(
                        "SELECT path FROM intent_modules WHERE id=?", (rid,)
                    ).fetchone()
                    if row:
                        path = str(row["path"])
                except Exception:
                    path = None
            if path:
                score = 1.0 / (1.0 + float(dist))
                results.append({"path": path, "score": score})
        return results[:top_k]


def find_modules_related_to(query: str, top_k: int = 5) -> List[Dict[str, float]]:
    """Convenience wrapper to query a fresh clusterer instance."""

    clusterer = IntentClusterer()
    return clusterer.find_modules_related_to(query, top_k=top_k)


__all__ = ["IntentClusterer", "extract_intent_text", "find_modules_related_to"]

