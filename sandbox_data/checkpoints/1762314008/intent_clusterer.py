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
import os
from datetime import datetime
import asyncio
import threading

from governed_embeddings import governed_embed
try:
    from menace_sandbox.embeddable_db_mixin import EmbeddableDBMixin
except ModuleNotFoundError:  # pragma: no cover - legacy flat import support
    from embeddable_db_mixin import EmbeddableDBMixin
from math import sqrt
from vector_utils import persist_embedding
from db_router import init_db_router, LOCAL_TABLES
from logging_utils import get_logger

try:  # pragma: no cover - optional dependency
    from sklearn.cluster import KMeans  # type: ignore
except Exception:  # pragma: no cover - fallback used in tests
    from knowledge_graph import _SimpleKMeans as KMeans  # type: ignore

try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path, resolve_module_path  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path, resolve_module_path  # type: ignore

try:  # pragma: no cover - optional dependency
    from sklearn.metrics import silhouette_score  # type: ignore
except Exception:  # pragma: no cover - fallback when sklearn is absent
    silhouette_score = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from watchfiles import watch
except Exception:  # pragma: no cover - graceful fallback
    watch = None


logger = get_logger(__name__)


def _load_canonical_categories() -> List[str]:
    """Return category names from environment or configuration."""

    default = [
        "scraping",
        "cleaning",
        "formatting",
        "automation",
        "decision-making",
        "authentication",
        "payment",
    ]

    file_path = os.getenv("INTENT_CATEGORIES_FILE")
    if file_path:
        try:
            data = json.loads(Path(file_path).read_text())
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
        except Exception:
            pass

    env = os.getenv("INTENT_CATEGORIES")
    if env:
        try:
            if env.strip().startswith("["):
                data = json.loads(env)
            else:
                data = [c.strip() for c in env.split(",") if c.strip()]
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
        except Exception:
            pass

    return default


CANONICAL_CATEGORIES: List[str] = _load_canonical_categories()

_CATEGORY_EMBED_PATH = Path(
    os.getenv("INTENT_CATEGORY_EMBEDDINGS", "category_embeddings.json")
)


def _load_category_vectors() -> Dict[str, List[float]]:
    """Load persisted category embeddings if available."""

    try:
        data = json.loads(_CATEGORY_EMBED_PATH.read_text())
        if isinstance(data, dict):
            return {
                str(k): [float(x) for x in v]
                for k, v in data.items()
                if isinstance(v, list)
            }
    except Exception:
        pass
    return {}


_CATEGORY_VECTORS: Dict[str, List[float]] = _load_category_vectors()


def _persist_category_vectors() -> None:
    """Persist category embeddings to disk."""

    try:
        _CATEGORY_EMBED_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CATEGORY_EMBED_PATH.write_text(json.dumps(_CATEGORY_VECTORS))
    except OSError:
        pass


def _categorise(label: str | None, summary: str | None) -> str | None:
    """Map ``label`` and ``summary`` to the closest canonical category."""

    text = " ".join(part for part in [label, summary] if part)
    if not text:
        return None
    vec = governed_embed(text)
    if not vec:
        return None
    norm = sqrt(sum(x * x for x in vec)) or 1.0
    vec = [x / norm for x in vec]
    best: tuple[str | None, float] = (None, -1.0)
    for cat in CANONICAL_CATEGORIES:
        cvec = _CATEGORY_VECTORS.get(cat)
        if not cvec:
            cvec = governed_embed(cat)
            _CATEGORY_VECTORS[cat] = list(cvec) if cvec else []
            _persist_category_vectors()
        if not cvec:
            continue
        cnorm = sqrt(sum(x * x for x in cvec)) or 1.0
        cvec_norm = [x / cnorm for x in cvec]
        sim = sum(a * b for a, b in zip(vec, cvec_norm))
        if sim > best[1]:
            best = (cat, sim)
    return best[0]


@dataclass
class IntentMatch:
    path: str | None
    similarity: float
    cluster_ids: List[int]
    label: str | None = None
    origin: str | None = None
    members: List[str] | None = None
    summary: str | None = None
    category: str | None = None
    intent_text: str | None = None


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


def summarise_texts(texts: List[str], method: str = "tfidf", top_k: int = 5) -> str:
    """Return a short summary for ``texts``.

    The helper relies solely on local processing and avoids heavyweight models
    or network calls.  Noun phrases (approximated via n‑grams) are scored using
    a tiny TF‑IDF scheme.  ``method`` can be set to ``"freq"`` to use raw
    frequency counts or ``"rake"`` to apply a lightweight RAKE keyword extractor.
    The ``top_k`` highest scoring phrases are returned, joined by spaces.
    """

    import math
    import re
    from collections import Counter

    if not texts:
        return ""

    scores: Counter[str]
    if method == "rake":
        try:
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
                "to",
                "of",
                "in",
                "on",
                "a",
                "an",
                "as",
                "is",
                "it",
                "be",
                "at",
                "by",
                "or",
                "if",
                "then",
                "else",
                "than",
                "so",
                "such",
                "via",
            }
            text = ". ".join(t for t in texts if t).lower()
            sentences = re.split(r"[.!?,;:\n]", text)
            phrases: List[List[str]] = []
            for sentence in sentences:
                words = re.findall(r"[a-z][a-z0-9_]*", sentence)
                phrase: List[str] = []
                for word in words:
                    if word in stop:
                        if phrase:
                            phrases.append(phrase)
                            phrase = []
                    else:
                        phrase.append(word)
                if phrase:
                    phrases.append(phrase)
            if not phrases:
                return ""
            freq = Counter()
            degree = Counter()
            for phrase in phrases:
                deg = len(phrase)
                for word in phrase:
                    freq[word] += 1
                    degree[word] += deg
            word_score = {w: degree[w] / freq[w] for w in freq}
            scores = Counter()
            for phrase in phrases:
                phrase_str = " ".join(phrase)
                scores[phrase_str] += sum(word_score[w] for w in phrase)
        except Exception:
            return ""
    else:
        def _phrases(text: str) -> List[str]:
            tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]*", text.lower())
            phrases = list(tokens)
            for n in (2, 3):
                phrases.extend(
                    " ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)
                )
            return phrases

        docs = [_phrases(t) for t in texts if t]
        if not docs:
            return ""

        if method == "tfidf":
            df = Counter()
            for doc in docs:
                df.update(set(doc))
            scores = Counter()
            N = len(docs)
            for doc in docs:
                tf = Counter(doc)
                for term, freq in tf.items():
                    idf = math.log((1 + N) / (1 + df[term])) + 1.0
                    scores[term] += freq * idf
        else:  # ``freq`` or any unknown method
            scores = Counter(p for doc in docs for p in doc)

    ranked = sorted(scores.items(), key=lambda x: (x[1], len(x[0].split())), reverse=True)
    terms = [term for term, _ in ranked[:top_k]]
    return " ".join(terms)


def derive_cluster_label(
    texts: List[str], top_k: int = 3, method: str = "rake"
) -> tuple[str, str]:
    """Return a concise ``(label, summary)`` pair for ``texts``.

    ``summarise_texts`` provides both the label and summary.  When no summary
    can be produced a final frequency based heuristic is used which guarantees
    deterministic output without external dependencies.
    """

    if not texts:
        return "", ""

    summary = ""
    try:
        summary = summarise_texts(texts, method=method, top_k=top_k)
    except Exception:
        summary = ""
    if not summary and method != "tfidf":
        try:
            summary = summarise_texts(texts, method="tfidf", top_k=top_k)
        except Exception:
            summary = ""
    if summary:
        return summary, summary

    # Final safety net: frequency based heuristic
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
        return "", ""
    label = " ".join(w for w, _ in Counter(words).most_common(top_k))
    return label, label


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
    clusters: Dict[str, List[int]]
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
        summary_method: str = "rake",
        summary_top_k: int = 3,
    ) -> None:
        self.vector_service = vector_service
        self.retriever = retriever
        self.summary_method = summary_method
        self.summary_top_k = summary_top_k
        if self.retriever is None and self.vector_service is None:
            self.retriever = type(
                "DummyRetriever",
                (),
                {
                    "register_db": lambda *a, **k: None,
                    "search": lambda *a, **k: [],
                    "add_vector": lambda *a, **k: None,
                },
            )()
        self.db = db or intent_db or ModuleVectorDB()
        if self.retriever is not None and hasattr(self.retriever, "register_db"):
            try:
                self.retriever.register_db(
                    "intent_cluster", self.db, ("path", "module_path")
                )
            except Exception:
                logger.warning(
                    "failed to register intent cluster DB with retriever", exc_info=True
                )
        self.module_ids = {}
        self.vectors = {}
        self.clusters: Dict[str, List[int]] = {}
        self.cluster_map: Dict[str, List[int]] = self.clusters  # backward compatibility alias
        self._cluster_cache = {}

        LOCAL_TABLES.add("intent_embeddings")
        if hasattr(self.db, "router"):
            self.router = self.db.router
        elif retriever is not None and hasattr(retriever, "router"):
            self.router = retriever.router
        else:
            try:  # pragma: no cover - best effort
                from universal_retriever import router as _ur_router

                self.router = _ur_router
            except Exception:  # pragma: no cover - fallback
                mid = menace_id or "intent"
                if local_db_path is None:
                    local = str(resolve_path(f"{mid}.db"))
                else:
                    try:
                        local = str(resolve_path(local_db_path))
                    except FileNotFoundError:
                        local = str(local_db_path)
                if shared_db_path is None:
                    shared = local
                else:
                    try:
                        shared = str(resolve_path(shared_db_path))
                    except FileNotFoundError:
                        shared = str(shared_db_path)
                self.router = init_db_router(mid, local, shared)
        self.conn = self.router.get_connection("intent_embeddings")
        self._ensure_table()
        self._load_existing_mappings()
        self.__post_init__()

    # ------------------------------------------------------------------
    def _load_synergy_groups(self, root: Path) -> Dict[str, List[str]]:
        """Return mapping of ``group_id`` to module paths under ``root``."""

        groups: Dict[str, List[str]] = {}
        map_file = Path(resolve_path("sandbox_data")) / "module_map.json"
        if map_file.exists():
            try:
                mapping = json.loads(map_file.read_text())
                for mod, gid in mapping.items():
                    path = resolve_module_path(mod.replace("/", "."))
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
                groups[str(idx)] = [
                    str(resolve_module_path(m.replace("/", "."))) for m in comp
                ]
            return groups
        except Exception as exc:
            logger.exception("failed to build synergy graph under %s: %s", root, exc)

        try:
            from module_graph_analyzer import build_import_graph
            import networkx as nx

            graph = build_import_graph(root)
            for idx, comp in enumerate(nx.connected_components(graph.to_undirected())):
                groups[str(idx)] = [
                    str(resolve_module_path(m.replace("/", "."))) for m in comp
                ]
            return groups
        except Exception as exc:
            logger.warning(
                "failed to build dependency graph under %s: %s", root, exc
            )
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
            label, summary = derive_cluster_label(
                texts, top_k=self.summary_top_k, method=self.summary_method
            )
            text = "\n".join(texts)
            category = _categorise(label, summary)

            meta = {
                "members": sorted(members),
                "kind": "cluster",
                "cluster_id": int(gid),
                "cluster_ids": [int(gid)],
                "path": entry,
                "label": label,
                "summary": summary,
                "category": category,
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
            if self.retriever is not None and hasattr(self.retriever, "add_vector"):
                try:
                    self.retriever.add_vector(mean, metadata=meta)
                except Exception as exc:
                    logger.warning("failed to add cluster %s to retriever: %s", gid, exc)
            if self.vector_service is not None and hasattr(
                self.vector_service, "add_vector"
            ):
                try:
                    self.vector_service.add_vector(mean, meta)
                except Exception as exc:
                    logger.warning(
                        "failed to add cluster %s to vector service: %s", gid, exc
                    )
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
                    "cluster_id": int(gid),
                    "cluster_ids": [int(gid)],
                    "label": label,
                    "summary": summary,
                    "category": category,
                    "text": text,
                    "intent_text": text,
                }
            except Exception as exc:
                logger.warning(
                    "failed to update vector DB for cluster %s: %s", gid, exc
                )
            try:
                persist_embedding(
                    "intent",
                    entry,
                    mean,
                    origin_db="intent",
                    metadata={
                        "type": "cluster",
                        "cluster_id": int(gid),
                        "members": sorted(members),
                        "intent_text": text,
                        "category": category,
                    },
                )
            except TypeError:  # pragma: no cover - backwards compatibility
                persist_embedding("intent", entry, mean)

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

    def _load_existing_mappings(self) -> None:
        """Load persisted vectors and cluster mappings from storage."""

        try:
            db_dir = Path(
                self.conn.execute("PRAGMA database_list").fetchone()[2]
            ).resolve().parent
        except Exception:
            db_dir = Path.cwd()

        try:
            cur = self.conn.execute(
                "SELECT module_path, vector, metadata FROM intent_embeddings"
            )
            for mpath, blob, meta_json in cur.fetchall():
                mpath = str(mpath)
                p = Path(mpath)
                if not (
                    mpath.startswith("cluster:")
                    or (p.exists() and p.resolve().is_relative_to(db_dir))
                ):
                    continue
                if blob and mpath not in self.vectors:
                    try:
                        vec = [float(x) for x in pickle.loads(blob)]
                    except Exception:
                        vec = []
                    if vec:
                        self.vectors[mpath] = vec
                try:
                    meta = json.loads(meta_json or "{}") if meta_json else {}
                except Exception:
                    meta = {}
                if not mpath.startswith("cluster:"):
                    cids = meta.get("cluster_ids")
                    if cids:
                        self.clusters[mpath] = [int(c) for c in cids]
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("failed to load stored intent embeddings: %s", exc)

        try:
            for rid, meta in getattr(self.db, "_metadata", {}).items():
                path = meta.get("source_id") or meta.get("path")
                if not path:
                    continue
                p = Path(path)
                if not (p.exists() and p.resolve().is_relative_to(db_dir)):
                    continue
                try:
                    self.module_ids[path] = int(rid)
                except Exception:
                    continue
                vec = meta.get("vector")
                if vec and path not in self.vectors:
                    self.vectors[path] = [float(x) for x in vec]
                cids = meta.get("cluster_ids")
                if cids and path not in self.clusters:
                    self.clusters[path] = [int(c) for c in cids]
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("failed to load module metadata: %s", exc)

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
                vec = self.db.get_vector(rid) or []
            else:  # fall back to IntentDB style API without persisting
                rid = self.db.add(str(path))
                vec = self.db.vector({"path": str(path), "text": text})
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
                if self.retriever is not None and hasattr(self.retriever, "add_vector"):
                    try:
                        self.retriever.add_vector(vec, {"path": str(path)})
                    except Exception as exc:
                        logger.warning("failed to add %s to retriever: %s", path, exc)
                if self.vector_service is not None and hasattr(
                    self.vector_service, "add_vector"
                ):
                    try:
                        self.vector_service.add_vector(vec, {"path": str(path)})
                    except Exception as exc:
                        logger.warning(
                            "failed to add %s to vector service: %s", path, exc
                        )

    # ------------------------------------------------------------------
    def update_modules(self, paths: Iterable[Path]) -> None:
        """Incrementally index ``paths`` updating only changed modules."""

        paths = [Path(p) for p in paths]
        if not paths:
            return

        wanted = {str(p) for p in paths}

        existing: Dict[str, Dict[str, Any]] = {}
        try:
            cur = self.conn.execute(
                "SELECT module_path, vector, metadata FROM intent_embeddings "
                "WHERE module_path NOT LIKE 'cluster:%'"
            )
            for mpath, blob, meta_json in cur.fetchall():
                mpath = str(mpath)
                meta = json.loads(meta_json or "{}") if meta_json else {}
                existing[mpath] = meta
                if blob and mpath not in self.vectors:
                    try:
                        vec = [float(x) for x in pickle.loads(blob)]
                    except Exception:
                        vec = []
                    if vec:
                        self.vectors[mpath] = vec
        except Exception:
            existing = {}

        stale = set(existing) - wanted
        if stale:
            with self.conn:
                for mpath in stale:
                    self.conn.execute(
                        "DELETE FROM intent_embeddings WHERE module_path = ?",
                        (mpath,),
                    )
                    self.module_ids.pop(mpath, None)
                    self.vectors.pop(mpath, None)
                    self.clusters.pop(mpath, None)

        changed: List[Path] = []
        for path in paths:
            mpath = str(path)
            mtime = path.stat().st_mtime
            meta = existing.get(mpath, {})
            if meta.get("mtime") != mtime or mpath not in self.vectors:
                changed.append(path)

        if changed:
            self.index_modules(changed)
            with self.conn:
                for path in changed:
                    mpath = str(path)
                    vec = self.vectors.get(mpath)
                    if not vec:
                        continue
                    blob = sqlite3.Binary(pickle.dumps(vec))
                    meta = {"mtime": path.stat().st_mtime}
                    self.conn.execute(
                        "REPLACE INTO intent_embeddings (module_path, vector, metadata) "
                        "VALUES (?, ?, ?)",
                        (mpath, blob, json.dumps(meta)),
                    )

        root = Path(os.path.commonpath([str(p.parent) for p in paths]))
        groups = self._load_synergy_groups(root)
        if groups:
            filtered: Dict[str, List[str]] = {}
            for gid, members in groups.items():
                valid = [m for m in members if m in wanted]
                if valid:
                    filtered[gid] = valid
                if filtered:
                    self._index_clusters(filtered)
                    try:
                        valid = {f"cluster:{gid}" for gid in filtered}
                        if hasattr(self.retriever, "items"):
                            self.retriever.items = [
                                it
                                for it in getattr(self.retriever, "items", [])
                                if not str(
                                    it.get("metadata", {}).get("path", "")
                                ).startswith("cluster:")
                                or it.get("metadata", {}).get("path") in valid
                            ]
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    def index_repository(self, repo_path: str | Path) -> None:
        """Embed modules under ``repo_path`` using incremental updates."""

        root = Path(resolve_path(repo_path))
        paths = list(root.rglob("*.py"))
        if not paths:
            return

        current = {str(p) for p in paths}
        previous: set[str] = set()
        try:
            cur = self.conn.execute(
                "SELECT module_path FROM intent_embeddings "
                "WHERE module_path NOT LIKE 'cluster:%'"
            )
            previous = {str(row[0]) for row in cur.fetchall()}
        except Exception:
            previous = set()

        missing = previous - current
        if missing:
            ids_to_remove: List[int] = []
            with self.conn:
                for mpath in missing:
                    self.conn.execute(
                        "DELETE FROM intent_embeddings WHERE module_path = ?",
                        (mpath,),
                    )
                    rid = self.module_ids.pop(mpath, None)
                    if rid is None:
                        for k, v in list(getattr(self.db, "_paths", {}).items()):
                            if v == mpath:
                                rid = k
                                break
                    if rid is not None:
                        ids_to_remove.append(int(rid))
                    self.vectors.pop(mpath, None)
                    self.clusters.pop(mpath, None)

            if ids_to_remove:
                for rid in ids_to_remove:
                    rid_str = str(rid)
                    getattr(self.db, "_texts", {}).pop(rid, None)
                    getattr(self.db, "_paths", {}).pop(rid, None)
                    getattr(self.db, "_metadata", {}).pop(rid_str, None)
                    try:
                        self.db._id_map.remove(rid_str)
                    except ValueError:
                        pass
                try:
                    self.db._rebuild_index()
                    self.db.save_index()
                except Exception:
                    logger.exception("failed to rebuild intent vector index")

        self.update_modules(paths)

    # ------------------------------------------------------------------
    def watch_repository(self, repo_path: str | Path):
        """Watch ``repo_path`` for Python file changes and keep the index fresh.

        This spawns a background thread that monitors ``.py`` files under
        ``repo_path`` using :mod:`watchfiles`.  Whenever a file is created,
        modified or removed the repository is re-indexed and synergy clusters
        are rebuilt via :meth:`_index_clusters`.

        The function returns a ``stop`` callable that terminates the watcher
        thread when invoked.  Example:

        .. code-block:: python

            clusterer = IntentClusterer()
            stop = clusterer.watch_repository(Path('.'))
            # ... perform work ...
            stop()

        Exceptions raised during indexing are caught and logged so the watcher
        can continue running without crashing the host process.
        """

        if watch is None:
            raise RuntimeError("watchfiles is required for watch_repository")

        repo_path = Path(resolve_path(repo_path))
        stop_event = threading.Event()

        def _run() -> None:
            for changes in watch(repo_path, stop_event=stop_event):
                relevant = [p for _evt, p in changes if p.endswith('.py')]
                if not relevant:
                    continue
                try:
                    self.index_repository(repo_path)
                    try:
                        groups = self._load_synergy_groups(repo_path)
                        if groups:
                            self._index_clusters(groups)
                    except Exception:
                        logger.exception("failed to rebuild clusters for %s", repo_path)
                except Exception:
                    logger.exception("error updating repository %s", repo_path)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        def stop() -> None:
            stop_event.set()
            thread.join()

        return stop

    # ------------------------------------------------------------------
    def _optimal_cluster_count(self, vectors: List[List[float]]) -> int:
        """Estimate an appropriate number of clusters for ``vectors``."""

        max_k = min(10, len(vectors))
        if max_k <= 1:
            return 1

        if silhouette_score:  # pragma: no cover - optional dependency
            best_k = 2
            best_score = -1.0
            for k in range(2, max_k + 1):
                km = KMeans(n_clusters=k)
                km.fit(vectors)
                labels = getattr(km, "labels_", km.predict(vectors))
                try:
                    score = float(silhouette_score(vectors, labels))
                except Exception:
                    score = -1.0
                if score > best_score:
                    best_score = score
                    best_k = k
            return best_k

        def _sse(k: int) -> float:
            km = KMeans(n_clusters=k)
            km.fit(vectors)
            centers = getattr(km, "cluster_centers_", getattr(km, "centers", []))
            labels = getattr(km, "labels_", km.predict(vectors))
            sse = 0.0
            for vec, idx in zip(vectors, labels):
                center = centers[idx]
                sse += sum((a - b) ** 2 for a, b in zip(vec, center))
            return sse

        prev = _sse(1)
        best_k = 1
        for k in range(2, max_k + 1):
            curr = _sse(k)
            improvement = (prev - curr) / prev if prev else 0.0
            if improvement < 0.1:
                break
            best_k = k
            prev = curr
        return max(best_k, 1)

    # ------------------------------------------------------------------
    def cluster_intents(
        self, n_clusters: int | None = None, *, threshold: float = 0.8
    ) -> Dict[str, List[int]]:
        """Group indexed modules into clusters.

        When ``n_clusters`` is ``None`` a best effort is made to determine an
        optimal cluster count using silhouette or elbow analysis.

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
        if n_clusters is None:
            n_clusters = self._optimal_cluster_count(vectors)
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
            # Persist cluster identifiers for each module
            try:
                row = self.conn.execute(
                    "SELECT vector, metadata FROM intent_embeddings WHERE module_path = ?",
                    (path,),
                ).fetchone()
                if row:
                    blob, meta_json = row
                    meta = json.loads(meta_json or "{}") if meta_json else {}
                else:
                    blob = sqlite3.Binary(pickle.dumps(vec))
                    meta = {}
                if not blob:
                    blob = sqlite3.Binary(pickle.dumps(vec))
                meta["cluster_ids"] = [int(cid) for cid in self.clusters[path]]
                with self.conn:
                    self.conn.execute(
                        "REPLACE INTO intent_embeddings (module_path, vector, metadata) "
                        "VALUES (?, ?, ?)",
                        (path, blob, json.dumps(meta)),
                    )
            except Exception as exc:
                logger.warning("failed to persist cluster ids for %s: %s", path, exc)
            rid = self.module_ids.get(path)
            if rid is not None:
                try:
                    meta = self.db._metadata.get(str(rid), {})
                    meta["cluster_ids"] = [int(cid) for cid in self.clusters[path]]
                    self.db._metadata[str(rid)] = meta
                except Exception:
                    pass
        self._index_clusters({k: v for k, v in cluster_members.items() if v})
        return dict(self.clusters)

    # ------------------------------------------------------------------
    def get_cluster_intents(self, cluster_id: int) -> tuple[str, List[float]]:
        """Return stored intent text and vector for ``cluster_id``.

        Persisted cluster metadata is loaded from the ``intent_embeddings``
        table or the vector DB.  When no stored vector is found the method falls
        back to aggregating member module texts and embedding them.  Results are
        cached to avoid repeated work.
        """

        cid = int(cluster_id)
        cached = self._cluster_cache.get(cid)
        if cached:
            return cached

        entry = f"cluster:{cid}"
        text = ""
        vector: List[float] = []
        meta: Dict[str, Any] = {}

        try:
            row = self.conn.execute(
                "SELECT vector, metadata FROM intent_embeddings WHERE module_path = ?",
                (entry,),
            ).fetchone()
        except Exception:
            row = None
        if row:
            vec_blob, meta_json = row
            if vec_blob:
                try:
                    vector = [float(x) for x in pickle.loads(vec_blob)]
                except Exception:
                    vector = []
            if meta_json:
                try:
                    meta = json.loads(meta_json or "{}")
                    text = str(meta.get("intent_text") or meta.get("text") or "")
                except Exception:
                    meta = {}

        if not meta:
            meta = getattr(self.db, "_metadata", {}).get(entry, {})
            if meta and not text:
                text = str(meta.get("intent_text") or meta.get("text") or "")
            if meta and not vector:
                vec = meta.get("vector")
                if vec:
                    vector = [float(x) for x in vec]

        if not vector:
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
    def _get_cluster_summary(self, cluster_id: int) -> str | None:
        """Return persisted summary for ``cluster_id`` if available."""

        entry = f"cluster:{int(cluster_id)}"
        meta = getattr(self.db, "_metadata", {}).get(entry)
        if meta and meta.get("summary"):
            return str(meta.get("summary"))
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
                summ = data.get("summary")
                if summ:
                    return str(summ)
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    def _get_cluster_category(self, cluster_id: int) -> str | None:
        """Return persisted category for ``cluster_id`` if available."""

        entry = f"cluster:{int(cluster_id)}"
        meta = getattr(self.db, "_metadata", {}).get(entry)
        if meta:
            cat = meta.get("category")
            if not cat:
                cat = _categorise(meta.get("label"), meta.get("summary"))
                if cat:
                    meta["category"] = cat
            if cat:
                return str(cat)
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
                cat = data.get("category")
                if not cat:
                    cat = _categorise(data.get("label"), data.get("summary"))
                if cat:
                    return str(cat)
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    def cluster_label(self, cluster_id: int) -> tuple[str, str]:
        """Return ``(label, summary)`` for ``cluster_id``.

        Labels and summaries are stored in the cluster metadata when clusters
        are indexed.  When either piece of information is missing, the method
        derives both on the fly using :func:`derive_cluster_label` and returns
        the result without persisting it.
        """

        lbl = self._get_cluster_label(cluster_id)
        summ = self._get_cluster_summary(cluster_id)
        if lbl is not None and summ is not None:
            return lbl, summ
        text, _vec = self.get_cluster_intents(cluster_id)
        if text:
            d_lbl, d_summ = derive_cluster_label(
                [text], top_k=self.summary_top_k, method=self.summary_method
            )
            changed = False
            if d_lbl is not None and d_lbl != lbl:
                lbl = d_lbl
                changed = True
            if d_summ is not None and d_summ != summ:
                summ = d_summ
                changed = True
            if changed:
                entry = f"cluster:{int(cluster_id)}"
                meta = {}
                try:
                    row = self.conn.execute(
                        "SELECT metadata FROM intent_embeddings WHERE module_path = ?",
                        (entry,),
                    ).fetchone()
                except Exception:
                    row = None
                if row:
                    try:
                        meta = json.loads(row[0] or "{}")
                    except Exception:
                        meta = {}
                if lbl is not None:
                    meta["label"] = lbl
                if summ is not None:
                    meta["summary"] = summ
                try:
                    self.conn.execute(
                        "UPDATE intent_embeddings SET metadata = ? WHERE module_path = ?",
                        (json.dumps(meta), entry),
                    )
                except Exception:
                    pass
                db_meta = getattr(self.db, "_metadata", None)
                if isinstance(db_meta, dict):
                    mem = db_meta.get(entry, {})
                    if lbl is not None:
                        mem["label"] = lbl
                    if summ is not None:
                        mem["summary"] = summ
                    db_meta[entry] = mem
                    try:
                        self.db._rebuild_index()  # type: ignore[attr-defined]
                        self.db.save_index()  # type: ignore[attr-defined]
                    except Exception as exc:  # pragma: no cover - best effort
                        logger.warning("failed to rebuild cluster index: %s", exc)
        return lbl or "", summ or ""

    # ------------------------------------------------------------------
    async def query_async(
        self,
        text: str,
        top_k: int = 5,
        *,
        threshold: float = 0.0,
        include_clusters: bool = True,
    ) -> List[IntentMatch]:
        """Asynchronously search retriever for modules relevant to ``text``."""

        vec = await asyncio.to_thread(self.db.encode_text, text)
        if not vec:
            return []
        norm = sqrt(sum(x * x for x in vec)) or 1.0
        vec = [x / norm for x in vec]
        hits = (
            await asyncio.to_thread(self.retriever.search, vec, top_k=top_k)
            if hasattr(self.retriever, "search")
            else []
        )
        results: List[IntentMatch] = []
        for item in hits:
            meta = item.get("metadata", {})
            label = meta.get("label")
            summary = meta.get("summary")
            category = meta.get("category")
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
                if cluster_ids:
                    if label is None:
                        label = await asyncio.to_thread(
                            self._get_cluster_label, cluster_ids[0]
                        )
                    if summary is None:
                        summary = await asyncio.to_thread(
                            self._get_cluster_summary, cluster_ids[0]
                        )
                    if category is None:
                        category = await asyncio.to_thread(
                            self._get_cluster_category, cluster_ids[0]
                        )
                results.append(
                    IntentMatch(
                        path=None,
                        similarity=sim,
                        cluster_ids=cluster_ids,
                        label=label,
                        category=category,
                        summary=summary,
                    )
                )
                continue
            if sim < threshold and cids:
                cluster_hits: List[tuple[int, float]] = []
                best_sim = sim
                for cid in cids:
                    _text, cvec = await asyncio.to_thread(
                        self.get_cluster_intents, int(cid)
                    )
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
                            if label is None:
                                label = await asyncio.to_thread(
                                    self._get_cluster_label, cluster_ids[0]
                                )
                            if summary is None:
                                summary = await asyncio.to_thread(
                                    self._get_cluster_summary, cluster_ids[0]
                                )
                            if category is None:
                                category = await asyncio.to_thread(
                                    self._get_cluster_category, cluster_ids[0]
                                )
            if sim < threshold:
                continue
            if include_clusters and not cluster_ids and cids:
                cluster_ids = [int(c) for c in cids]
                if cluster_ids:
                    if label is None:
                        label = await asyncio.to_thread(
                            self._get_cluster_label, cluster_ids[0]
                        )
                    if summary is None:
                        summary = await asyncio.to_thread(
                            self._get_cluster_summary, cluster_ids[0]
                        )
                    if category is None:
                        category = await asyncio.to_thread(
                            self._get_cluster_category, cluster_ids[0]
                        )
            elif not include_clusters and cids and len(cids) > 1:
                cluster_ids = [int(c) for c in cids]
            results.append(
                IntentMatch(
                    path=path,
                    similarity=sim,
                    cluster_ids=cluster_ids,
                    label=label,
                    category=category,
                    summary=summary,
                )
            )
        return results

    def query(
        self,
        text: str,
        top_k: int = 5,
        *,
        threshold: float = 0.0,
        include_clusters: bool = True,
    ) -> List[IntentMatch]:
        """Synchronous wrapper for :meth:`query_async`."""

        return asyncio.run(
            self.query_async(
                text,
                top_k=top_k,
                threshold=threshold,
                include_clusters=include_clusters,
            )
        )

    # ------------------------------------------------------------------
    def _search_related(self, prompt: str, top_k: int = 5) -> List[IntentMatch]:
        """Return intent matches for ``prompt``.

        Each result contains a similarity ``score`` and an ``origin`` field
        describing whether the entry refers to a ``module`` or a synergy
        ``cluster``.  When a retriever is unavailable the method falls back to
        the local vector index.
        """

        vec = self.db.encode_text(prompt)
        if not vec:
            return []

        # Prefer retriever-based search when available
        if self.retriever is not None and hasattr(self.retriever, "search"):
            norm = sqrt(sum(x * x for x in vec)) or 1.0
            qvec = [x / norm for x in vec]
            try:
                hits = self.retriever.search(qvec, top_k=top_k) or []
            except Exception:
                hits = []
            results: List[IntentMatch] = []
            for item in hits:
                meta = item.get("metadata", {})
                path = meta.get("path")
                members = meta.get("members")
                origin = meta.get("kind") or meta.get("source_id") or path
                cluster_ids = meta.get("cluster_ids")
                cluster_id = meta.get("cluster_id")
                if cluster_ids is None and cluster_id is not None:
                    cluster_ids = [cluster_id]
                label = meta.get("label")
                summary = meta.get("summary")
                intent_text = meta.get("intent_text")
                category = meta.get("category")
                if category is None and cluster_ids:
                    category = self._get_cluster_category(cluster_ids[0])
                if category is None:
                    category = _categorise(label, summary)
                target_vec: Sequence[float] = item.get("vector", [])
                tnorm = sqrt(sum(x * x for x in target_vec)) or 1.0
                target_vec = [x / tnorm for x in target_vec]
                score = sum(a * b for a, b in zip(qvec, target_vec))
                if path or members:
                    results.append(
                        IntentMatch(
                            path=path,
                            similarity=score,
                            cluster_ids=[int(c) for c in cluster_ids] if cluster_ids else [],
                            label=str(label) if label else None,
                            origin=str(origin) if origin else None,
                            members=list(members) if members else None,
                            summary=str(summary) if summary is not None else None,
                            category=str(category) if category else None,
                            intent_text=str(intent_text) if intent_text else None,
                        )
                    )
            if results:
                return results[:top_k]
        elif self.vector_service is not None and hasattr(self.vector_service, "search"):
            norm = sqrt(sum(x * x for x in vec)) or 1.0
            qvec = [x / norm for x in vec]
            try:
                hits = self.vector_service.search(qvec, top_k=top_k) or []
            except Exception:
                hits = []
            results: List[IntentMatch] = []
            for item in hits:
                meta = item.get("metadata", {})
                path = meta.get("path")
                members = meta.get("members")
                origin = meta.get("kind") or meta.get("source_id") or path
                cluster_ids = meta.get("cluster_ids")
                cluster_id = meta.get("cluster_id")
                if cluster_ids is None and cluster_id is not None:
                    cluster_ids = [cluster_id]
                label = meta.get("label")
                summary = meta.get("summary")
                intent_text = meta.get("intent_text")
                category = meta.get("category")
                if category is None and cluster_ids:
                    category = self._get_cluster_category(cluster_ids[0])
                if category is None:
                    category = _categorise(label, summary)
                target_vec: Sequence[float] = item.get("vector", [])
                tnorm = sqrt(sum(x * x for x in target_vec)) or 1.0
                target_vec = [x / tnorm for x in target_vec]
                score = sum(a * b for a, b in zip(qvec, target_vec))
                if path or members:
                    results.append(
                        IntentMatch(
                            path=path,
                            similarity=score,
                            cluster_ids=[int(c) for c in cluster_ids] if cluster_ids else [],
                            label=str(label) if label else None,
                            origin=str(origin) if origin else None,
                            members=list(members) if members else None,
                            summary=str(summary) if summary is not None else None,
                            category=str(category) if category else None,
                            intent_text=str(intent_text) if intent_text else None,
                        )
                    )
            if results:
                return results[:top_k]

        # Fallback: direct search on the vector database
        hits = []
        try:
            hits = self.db.search_by_vector(vec, top_k)
        except Exception:
            return []
        results: List[IntentMatch] = []
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
            cluster_id = meta.get("cluster_id")
            if cluster_ids is None and cluster_id is not None:
                cluster_ids = [cluster_id]
            label = meta.get("label")
            summary = meta.get("summary")
            intent_text = meta.get("intent_text")
            category = meta.get("category")
            if category is None and cluster_ids:
                category = self._get_cluster_category(cluster_ids[0])
            if category is None:
                category = _categorise(label, summary)
            origin = meta.get("kind") or meta.get("source_id") or path
            if path or members:
                score = 1.0 / (1.0 + float(dist))
                results.append(
                    IntentMatch(
                        path=path,
                        similarity=score,
                        cluster_ids=[int(c) for c in cluster_ids] if cluster_ids else [],
                        label=str(label) if label else None,
                        origin=str(origin) if origin else None,
                        members=list(members) if members else None,
                        summary=str(summary) if summary is not None else None,
                        category=str(category) if category else None,
                        intent_text=str(intent_text) if intent_text else None,
                    )
                )
        return results[:top_k]

    # ------------------------------------------------------------------
    def find_modules_related_to(
        self, prompt: str, top_k: int = 5, *, include_clusters: bool = False
    ) -> List[IntentMatch]:
        """Return modules related to ``prompt``."""

        return asyncio.run(
            self.find_modules_related_to_async(
                prompt, top_k=top_k, include_clusters=include_clusters
            )
        )

    async def find_modules_related_to_async(
        self, prompt: str, top_k: int = 5, *, include_clusters: bool = False
    ) -> List[IntentMatch]:
        """Asynchronously return modules related to ``prompt``."""

        results = await asyncio.to_thread(self._search_related, prompt, top_k * 2)
        if not include_clusters:
            results = [r for r in results if r.origin != "cluster"]
        return results[:top_k]

    # ------------------------------------------------------------------
    def find_clusters_related_to(self, prompt: str, top_k: int = 5) -> List[IntentMatch]:
        """Return synergy clusters related to ``prompt``."""

        return asyncio.run(self.find_clusters_related_to_async(prompt, top_k=top_k))

    async def find_clusters_related_to_async(
        self, prompt: str, top_k: int = 5
    ) -> List[IntentMatch]:
        """Asynchronously return synergy clusters related to ``prompt``."""

        results = [
            r
            for r in await asyncio.to_thread(self._search_related, prompt, top_k * 2)
            if r.origin == "cluster"
        ]
        return results[:top_k]


def find_modules_related_to(
    query: str, top_k: int = 5, *, include_clusters: bool = False
) -> List[IntentMatch]:
    """Convenience wrapper to query a fresh clusterer instance."""
    clusterer = IntentClusterer()
    if hasattr(clusterer, "find_modules_related_to_async"):
        return asyncio.run(
            clusterer.find_modules_related_to_async(
                query, top_k=top_k, include_clusters=include_clusters
            )
        )
    return clusterer.find_modules_related_to(
        query, top_k=top_k, include_clusters=include_clusters
    )


async def find_modules_related_to_async(
    query: str, top_k: int = 5, *, include_clusters: bool = False
) -> List[IntentMatch]:
    """Asynchronously query a fresh clusterer instance."""
    clusterer = IntentClusterer()
    if hasattr(clusterer, "find_modules_related_to_async"):
        return await clusterer.find_modules_related_to_async(
            query, top_k=top_k, include_clusters=include_clusters
        )
    return clusterer.find_modules_related_to(
        query, top_k=top_k, include_clusters=include_clusters
    )


def find_clusters_related_to(query: str, top_k: int = 5) -> List[IntentMatch]:
    """Convenience wrapper returning synergy clusters for ``query``."""
    clusterer = IntentClusterer()
    if hasattr(clusterer, "find_clusters_related_to_async"):
        return asyncio.run(
            clusterer.find_clusters_related_to_async(query, top_k=top_k)
        )
    return clusterer.find_clusters_related_to(query, top_k=top_k)


async def find_clusters_related_to_async(query: str, top_k: int = 5) -> List[IntentMatch]:
    """Asynchronously return synergy clusters for ``query``."""
    clusterer = IntentClusterer()
    if hasattr(clusterer, "find_clusters_related_to_async"):
        return await clusterer.find_clusters_related_to_async(query, top_k=top_k)
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
    "find_modules_related_to_async",
    "find_clusters_related_to",
    "find_clusters_related_to_async",
]
if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(_main())
