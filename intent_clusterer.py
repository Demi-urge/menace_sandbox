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

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Any, Sequence
import ast
import io
import tokenize
import sqlite3
import json
import pickle
from datetime import datetime

from governed_embeddings import governed_embed
from embeddable_db_mixin import EmbeddableDBMixin
from math import sqrt
from vector_utils import persist_embedding
from db_router import init_db_router, LOCAL_TABLES
from logging_utils import get_logger

try:  # pragma: no cover - optional dependency
    from sklearn.cluster import KMeans  # type: ignore
except Exception:  # pragma: no cover - fallback used in tests
    from knowledge_graph import _SimpleKMeans as KMeans  # type: ignore


logger = get_logger(__name__)


@dataclass
class IntentMatch:
    path: str | None
    similarity: float
    cluster_ids: List[int]
    label: str | None = None


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


def derive_cluster_label(texts: List[str], top_k: int = 3) -> str:
    """Return a concise label summarising ``texts``.

    The helper extracts the ``top_k`` keywords using a TF‑IDF weighting
    scheme.  When ``scikit-learn`` is unavailable a simple frequency based
    fallback is employed.  The resulting keywords are space separated to form
    a short label representing the cluster's intent.
    """

    if not texts:
        return ""
    try:  # pragma: no cover - optional dependency
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

        vec = TfidfVectorizer(stop_words="english")
        matrix = vec.fit_transform(texts)
        scores = matrix.sum(axis=0).A1
        terms = vec.get_feature_names_out()
        top = scores.argsort()[-top_k:][::-1]
        return " ".join(terms[i] for i in top if scores[i] > 0)
    except Exception:
        import re
        from collections import Counter

        words = re.findall(r"[A-Za-z][A-Za-z0-9_]+", " ".join(texts).lower())
        stop = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "from",
            "this",
            "are",
            "was",
            "but",
            "not",
            "use",
            "used",
            "using",
            "into",
            "over",
            "there",
            "their",
        }
        words = [w for w in words if w not in stop]
        if not words:
            return ""
        return " ".join(w for w, _ in Counter(words).most_common(top_k))


class ModuleVectorDB(EmbeddableDBMixin):
    """Minimal vector DB for module intent embeddings."""

    def __init__(
        self,
        *,
        index_path: str | Path = "intent_vectors.ann",
        metadata_path: str | Path = "intent_vectors.json",
    ) -> None:
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
    router: Any
    conn: sqlite3.Connection
    _cluster_cache: Dict[int, tuple[str, List[float]]]

    def __init__(
        self,
        retriever: Any | None = None,
        db: ModuleVectorDB | None = None,
        *,
        intent_db: ModuleVectorDB | None = None,
        vector_service: Any | None = None,
        menace_id: str | None = None,
        local_db_path: str | Path | None = None,
        shared_db_path: str | Path | None = None,
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
        self._cluster_cache = {}

        LOCAL_TABLES.add("intent_embeddings")
        if retriever is not None and hasattr(retriever, "router"):
            self.router = retriever.router
        else:
            try:  # pragma: no cover - best effort
                from universal_retriever import router as _ur_router

                self.router = _ur_router
            except Exception:  # pragma: no cover - fallback
                mid = menace_id or "intent"
                local = str(local_db_path or f"./{mid}.db")
                shared = str(shared_db_path or local)
                self.router = init_db_router(mid, local, shared)
        self.conn = self.router.get_connection("intent_embeddings")
        self._ensure_table()
        self.__post_init__()

    # ------------------------------------------------------------------
    def _load_synergy_groups(self, root: Path) -> Dict[str, List[str]]:
        """Return mapping of ``group_id`` to module paths under ``root``."""

        groups: Dict[str, List[str]] = {}
        map_file = root / "sandbox_data" / "module_map.json"
        if map_file.exists():
            try:
                mapping = json.loads(map_file.read_text())
                for mod, gid in mapping.items():
                    path = root / f"{mod}.py"
                    groups.setdefault(str(gid), []).append(str(path))
                return groups
            except Exception as exc:
                logger.warning("failed to read module map from %s: %s", map_file, exc)
        try:  # pragma: no cover - optional dependency
            from module_synergy_grapher import ModuleSynergyGrapher
            import networkx as nx

            grapher = ModuleSynergyGrapher(root=root)
            graph = grapher.load()
            for idx, comp in enumerate(nx.connected_components(graph.to_undirected())):
                groups[str(idx)] = [str(root / f"{m}.py") for m in comp]
        except Exception as exc:
            logger.exception("failed to build synergy graph under %s: %s", root, exc)
            return {}
        return groups

    # ------------------------------------------------------------------
    def _index_clusters(self, groups: Dict[str, List[str]]) -> None:
        """Aggregate embeddings for ``groups`` and persist as cluster entries."""

        desired = {f"cluster:{gid}" for gid in groups}
        try:
            existing = {
                rid
                for rid in getattr(self.db, "_metadata", {})  # type: ignore[attr-defined]
                if str(rid).startswith("cluster:")
            }
        except Exception:
            existing = set()

        stale = existing - desired
        for rid in list(stale):
            try:
                with self.conn:
                    self.conn.execute(
                        "DELETE FROM intent_embeddings WHERE module_path = ?",
                        (rid,),
                    )
                if rid in self.db._metadata:  # type: ignore[attr-defined]
                    del self.db._metadata[rid]  # type: ignore[attr-defined]
                if rid in getattr(self.db, "_id_map", []):  # type: ignore[attr-defined]
                    self.db._id_map.remove(rid)  # type: ignore[attr-defined]
            except Exception as exc:
                logger.warning("failed to remove stale cluster %s: %s", rid, exc)

        for gid, members in groups.items():
            try:
                self._cluster_cache.pop(int(gid), None)
            except Exception:
                pass
            vectors = [self.vectors.get(m) for m in members if self.vectors.get(m)]
            if not vectors:
                continue
            dim = len(vectors[0])
            agg = [0.0] * dim
            for vec in vectors:
                if len(vec) != dim:
                    continue
                for i, val in enumerate(vec):
                    agg[i] += float(val)
            mean = [v / len(vectors) for v in agg]
            entry = f"cluster:{gid}"
            texts: List[str] = []
            for m in members:
                try:
                    txt = extract_intent_text(Path(m))
                    if txt:
                        texts.append(txt)
                except Exception as exc:
                    logger.warning("failed to extract intent text from %s: %s", m, exc)
                    continue
            label = derive_cluster_label(texts)
            text = "\n".join(texts)

            meta = {
                "members": sorted(members),
                "kind": "cluster",
                "cluster_ids": [int(gid)],
                "path": entry,
                "label": label,
                "text": text,
                "intent_text": text,
            }
            try:
                blob = sqlite3.Binary(pickle.dumps(mean))
                with self.conn:
                    self.conn.execute(
                        "REPLACE INTO intent_embeddings (module_path, vector, metadata) "
                        "VALUES (?, ?, ?)",
                        (entry, blob, json.dumps(meta)),
                    )
            except Exception as exc:
                logger.exception("failed to persist cluster %s: %s", gid, exc)
            if hasattr(self.retriever, "add_vector"):
                try:
                    self.retriever.add_vector(mean, meta)
                except Exception as exc:
                    logger.warning("failed to add cluster %s to retriever: %s", gid, exc)
            try:
                rid = entry
                if rid not in self.db._metadata:  # type: ignore[attr-defined]
                    self.db._id_map.append(rid)  # type: ignore[attr-defined]
                self.db._metadata[rid] = {  # type: ignore[attr-defined]
                    "vector": list(mean),
                    "created_at": datetime.utcnow().isoformat(),
                    "embedding_version": getattr(self.db, "embedding_version", 1),
                    "kind": "cluster",
                    "source_id": rid,
                    "redacted": True,
                    "members": sorted(members),
                    "path": entry,
                    "cluster_ids": [int(gid)],
                    "label": label,
                    "text": text,
                    "intent_text": text,
                }
            except Exception as exc:
                logger.warning(
                    "failed to update vector DB for cluster %s: %s", gid, exc
                )

        try:
            self.db._rebuild_index()  # type: ignore[attr-defined]
            self.db.save_index()  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("failed to rebuild cluster index: %s", exc)

    def _ensure_table(self) -> None:
        """Create or migrate the ``intent_embeddings`` table."""
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS intent_embeddings (
                    module_path TEXT PRIMARY KEY,
                    vector BLOB,
                    metadata JSON
                )
                """
            )
            cols = {
                row[1]
                for row in self.conn.execute(
                    "PRAGMA table_info(intent_embeddings)"
                ).fetchall()
            }
            if "vector" not in cols:
                self.conn.execute(
                    "ALTER TABLE intent_embeddings ADD COLUMN vector BLOB"
                )
            if "metadata" not in cols:
                self.conn.execute(
                    "ALTER TABLE intent_embeddings ADD COLUMN metadata JSON"
                )

    def __post_init__(self) -> None:
        try:  # pragma: no cover - best effort registration
            self.retriever.register_db("intent", self.db, ("path",))
        except Exception as exc:
            logger.warning("failed to register intent DB with retriever: %s", exc)

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
                except Exception as exc:
                    logger.warning("failed to add embedding for %s: %s", path, exc)
                    continue
            vec = self.db.get_vector(rid) or []
            if vec:
                self.module_ids[str(path)] = rid
                self.vectors[str(path)] = vec
                try:
                    persist_embedding(
                        "intent",
                        str(path),
                        vec,
                        origin_db="intent",
                        metadata={"type": "intent", "module": str(path)},
                    )
                except TypeError:  # pragma: no cover - backwards compatibility
                    persist_embedding("intent", str(path), vec)
                # Persist in retriever for cross-module search when available
                if hasattr(self.retriever, "add_vector"):
                    try:
                        self.retriever.add_vector(vec, {"path": str(path)})
                    except Exception as exc:
                        logger.warning("failed to add %s to retriever: %s", path, exc)

    # ------------------------------------------------------------------
    def index_repository(self, repo_path: str | Path) -> None:
        """Embed all modules in ``repo_path`` and store vectors in the table.

        Only database entries for modules that are new or whose modification
        time changed are updated.
        """

        root = Path(repo_path)
        paths = list(root.rglob("*.py"))
        if not paths:
            return

        self.index_modules(paths)

        existing: Dict[str, Dict[str, Any]] = {}
        try:
            cur = self.conn.execute(
                "SELECT module_path, metadata FROM intent_embeddings"
            )
            for mpath, meta_json in cur.fetchall():
                try:
                    existing[str(mpath)] = json.loads(meta_json or "{}")
                except Exception:
                    existing[str(mpath)] = {}
        except Exception:
            existing = {}

        with self.conn:
            for path in paths:
                mpath = str(path)
                mtime = path.stat().st_mtime
                meta = existing.get(mpath, {})
                if meta.get("mtime") == mtime:
                    continue
                vec = self.vectors.get(mpath)
                if not vec:
                    continue
                blob = sqlite3.Binary(pickle.dumps(vec))
                new_meta = {"mtime": mtime}
                self.conn.execute(
                    "REPLACE INTO intent_embeddings (module_path, vector, metadata) "
                    "VALUES (?, ?, ?)",
                    (mpath, blob, json.dumps(new_meta)),
                )

        groups = self._load_synergy_groups(root)
        if groups:
            self._index_clusters(groups)

    # ------------------------------------------------------------------
    def cluster_intents(
        self, n_clusters: int, *, threshold: float = 0.8
    ) -> Dict[str, List[int]]:
        """Group indexed modules into ``n_clusters`` clusters.

        Unlike the previous hard assignment approach, modules may now belong to
        multiple clusters.  For each module we compute the Euclidean distance to
        every cluster centroid and convert it to a similarity score using the
        ``1 / (1 + distance)`` transform.  All clusters whose similarity exceeds
        ``threshold`` are associated with the module.  At least one cluster is
        always assigned (the closest centroid).
        """

        if not self.vectors:
            return {}
        vectors = list(self.vectors.values())
        km = KMeans(n_clusters=n_clusters)
        km.fit(vectors)
        centers = [list(c) for c in getattr(km, "cluster_centers_", [])]

        self.clusters = {}
        cluster_members: Dict[str, List[str]] = {
            str(i): [] for i in range(len(centers))
        }
        for path, vec in self.vectors.items():
            sims: List[int] = []
            scores: List[float] = []
            for idx, center in enumerate(centers):
                dist = sqrt(sum((a - b) * (a - b) for a, b in zip(vec, center)))
                score = 1.0 / (1.0 + dist)
                if score >= threshold:
                    sims.append(idx)
                scores.append(score)
            if not sims and scores:
                # Always associate with the closest cluster
                sims = [int(max(range(len(scores)), key=lambda i: scores[i]))]
            self.clusters[path] = [int(cid) for cid in sims]
            for cid in self.clusters[path]:
                cluster_members.setdefault(str(cid), []).append(path)
            # Update retriever metadata with the assigned cluster identifiers
            if hasattr(self.retriever, "add_vector"):
                try:
                    self.retriever.add_vector(
                        vec, {"path": path, "cluster_ids": self.clusters[path]}
                    )
                except Exception:
                    continue
        self._index_clusters({k: v for k, v in cluster_members.items() if v})
        return dict(self.clusters)

    # ------------------------------------------------------------------
    def get_cluster_intents(self, cluster_id: int) -> tuple[str, List[float]]:
        """Return concatenated intent text and vector for ``cluster_id``.

        All member module texts are gathered and concatenated into a single
        string which is then embedded via :func:`governed_embed` to produce the
        cluster vector.  Results are cached to avoid repeated embedding work.
        """

        cid = int(cluster_id)
        cached = self._cluster_cache.get(cid)
        if cached:
            return cached

        entry = f"cluster:{cid}"
        meta: Dict[str, Any] = {}

        # Try to load persisted cluster metadata to obtain members or text
        try:
            row = self.conn.execute(
                "SELECT metadata FROM intent_embeddings WHERE module_path = ?",
                (entry,),
            ).fetchone()
            if row and row[0]:
                meta = json.loads(row[0] or "{}")
        except Exception:
            meta = {}
        if not meta:
            meta = getattr(self.db, "_metadata", {}).get(entry, {})

        text = str(meta.get("intent_text") or meta.get("text") or "")
        if not text:
            members = meta.get("members") or [
                path
                for path, cids in self.clusters.items()
                if cid in (cids if isinstance(cids, list) else [cids])
            ]
            texts: List[str] = []
            for path in members:
                rid = self.module_ids.get(path)
                mod_meta = {}
                if rid is not None:
                    mod_meta = getattr(self.db, "_metadata", {}).get(str(rid), {})
                mod_text = mod_meta.get("text") if mod_meta else None
                if not mod_text:
                    try:
                        mod_text = extract_intent_text(Path(path))
                    except Exception:
                        mod_text = None
                if mod_text:
                    texts.append(mod_text)
            text = "\n".join(texts)

        vec = governed_embed(text) if text else []
        vector = [float(x) for x in vec] if vec else []
        result = (text, vector)
        self._cluster_cache[cid] = result
        return result

    # ------------------------------------------------------------------
    def _get_cluster_label(self, cluster_id: int) -> str | None:
        """Return persisted label for ``cluster_id`` if available."""

        entry = f"cluster:{int(cluster_id)}"
        meta = getattr(self.db, "_metadata", {}).get(entry)
        if meta and meta.get("label"):
            return str(meta.get("label"))
        try:
            row = self.conn.execute(
                "SELECT metadata FROM intent_embeddings WHERE module_path = ?",
                (entry,),
            ).fetchone()
        except Exception:
            row = None
        if row:
            try:
                data = json.loads(row[0] or "{}")
                lbl = data.get("label")
                if lbl:
                    return str(lbl)
            except Exception:
                return None
        return None

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
        hits = (
            self.retriever.search(vec, top_k=top_k)
            if hasattr(self.retriever, "search")
            else []
        )
        results: List[IntentMatch] = []
        for item in hits:
            meta = item.get("metadata", {})
            label = meta.get("label")
            target_vec: Sequence[float] = item.get("vector", [])
            tnorm = sqrt(sum(x * x for x in target_vec)) or 1.0
            target_vec = [x / tnorm for x in target_vec]
            sim = sum(a * b for a, b in zip(vec, target_vec))
            path = meta.get("path")
            kind = meta.get("kind")
            cids = meta.get("cluster_ids")
            if cids is None:
                cid = meta.get("cluster_id")
                if cid is not None:
                    cids = [cid]
            cluster_ids: List[int] = []
            if kind == "cluster":
                if not include_clusters or sim < threshold:
                    continue
                if cids:
                    cluster_ids = [int(c) for c in cids]
                if label is None and cluster_ids:
                    label = self._get_cluster_label(cluster_ids[0])
                results.append(
                    IntentMatch(
                        path=None, similarity=sim, cluster_ids=cluster_ids, label=label
                    )
                )
                continue
            if sim < threshold and cids:
                cluster_hits: List[tuple[int, float]] = []
                best_sim = sim
                for cid in cids:
                    _text, cvec = self.get_cluster_intents(int(cid))
                    if cvec:
                        cnorm = sqrt(sum(x * x for x in cvec)) or 1.0
                        cvec = [x / cnorm for x in cvec]
                        c_sim = sum(a * b for a, b in zip(vec, cvec))
                        if c_sim >= threshold:
                            cluster_hits.append((int(cid), c_sim))
                            if c_sim > best_sim:
                                best_sim = c_sim
                if cluster_hits:
                    sim = best_sim
                    path = None
                    if include_clusters:
                        cluster_ids = [cid for cid, _ in cluster_hits]
                        if cluster_ids:
                            label = self._get_cluster_label(cluster_ids[0])
            if sim < threshold:
                continue
            if include_clusters and not cluster_ids and cids:
                cluster_ids = [int(c) for c in cids]
                if cluster_ids and label is None:
                    label = self._get_cluster_label(cluster_ids[0])
            results.append(
                IntentMatch(path=path, similarity=sim, cluster_ids=cluster_ids, label=label)
            )
        return results

    # ------------------------------------------------------------------
    def _search_related(self, prompt: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return raw intent matches for ``prompt``.

        Each result contains a similarity ``score`` and an ``origin`` field
        describing whether the entry refers to a ``module`` or a synergy
        ``cluster``.  When a retriever is unavailable the method falls back to
        the local vector index.
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
            results: List[Dict[str, Any]] = []
            for item in hits:
                meta = item.get("metadata", {})
                path = meta.get("path")
                members = meta.get("members")
                origin = meta.get("kind") or meta.get("source_id") or path
                cluster_ids = meta.get("cluster_ids")
                if cluster_ids is None:
                    cid = meta.get("cluster_id")
                    if cid is not None:
                        cluster_ids = [cid]
                label = meta.get("label")
                target_vec: Sequence[float] = item.get("vector", [])
                tnorm = sqrt(sum(x * x for x in target_vec)) or 1.0
                target_vec = [x / tnorm for x in target_vec]
                score = sum(a * b for a, b in zip(qvec, target_vec))
                if path or members:
                    entry: Dict[str, Any] = {"score": score, "origin": origin}
                    if path:
                        entry["path"] = path
                    if members:
                        entry["members"] = list(members)
                    if cluster_ids:
                        entry["cluster_ids"] = [int(c) for c in cluster_ids]
                    if label:
                        entry["label"] = str(label)
                    results.append(entry)
            if results:
                return results[:top_k]

        # Fallback: direct search on the vector database
        hits = []
        try:
            hits = self.db.search_by_vector(vec, top_k)
        except Exception:
            return []
        results: List[Dict[str, Any]] = []
        for rid, dist in hits:
            path: str | None = None
            if hasattr(self.db, "get_path"):
                try:
                    path = self.db.get_path(int(rid))
                except Exception:
                    path = None
            elif hasattr(self.db, "conn"):
                try:
                    row = self.db.conn.execute(
                        "SELECT path FROM intent_modules WHERE id=?", (rid,)
                    ).fetchone()
                    if row:
                        path = str(row["path"])
                except Exception:
                    path = None
            meta = getattr(self.db, "_metadata", {}).get(str(rid), {})
            if not path:
                path = meta.get("path")
            members = meta.get("members")
            cluster_ids = meta.get("cluster_ids")
            if cluster_ids is None:
                cid = meta.get("cluster_id")
                if cid is not None:
                    cluster_ids = [cid]
            label = meta.get("label")
            origin = meta.get("kind") or meta.get("source_id") or path
            if path or members:
                score = 1.0 / (1.0 + float(dist))
                entry: Dict[str, Any] = {"score": score, "origin": origin}
                if path:
                    entry["path"] = path
                if members:
                    entry["members"] = list(members)
                if cluster_ids:
                    entry["cluster_ids"] = [int(c) for c in cluster_ids]
                if label:
                    entry["label"] = str(label)
                results.append(entry)
        return results[:top_k]

    # ------------------------------------------------------------------
    def find_modules_related_to(
        self, prompt: str, top_k: int = 5, *, include_clusters: bool = False
    ) -> List[Dict[str, Any]]:
        """Return modules related to ``prompt``.

        When ``include_clusters`` is ``True`` the result set may also contain
        synergy cluster entries with ``origin`` set to ``"cluster"``.
        """

        results = self._search_related(prompt, top_k * 2)
        if not include_clusters:
            results = [r for r in results if r.get("origin") != "cluster"]
        return results[:top_k]

    # ------------------------------------------------------------------
    def find_clusters_related_to(self, prompt: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return synergy clusters related to ``prompt``."""

        results = [
            r for r in self._search_related(prompt, top_k * 2) if r.get("origin") == "cluster"
        ]
        return results[:top_k]


def find_modules_related_to(
    query: str, top_k: int = 5, *, include_clusters: bool = False
) -> List[Dict[str, Any]]:
    """Convenience wrapper to query a fresh clusterer instance."""

    clusterer = IntentClusterer()
    return clusterer.find_modules_related_to(
        query, top_k=top_k, include_clusters=include_clusters
    )


def find_clusters_related_to(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Convenience wrapper returning synergy clusters for ``query``."""

    clusterer = IntentClusterer()
    return clusterer.find_clusters_related_to(query, top_k=top_k)


def _main(argv: Iterable[str] | None = None) -> int:
    """Minimal CLI for semantic module search."""

    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, dest="top_k")
    args = parser.parse_args(list(argv) if argv is not None else None)

    results = find_modules_related_to(args.query, top_k=args.top_k)
    print(json.dumps(results, indent=2))
    return 0


__all__ = [
    "IntentClusterer",
    "extract_intent_text",
    "find_modules_related_to",
    "find_clusters_related_to",
]
if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(_main())
