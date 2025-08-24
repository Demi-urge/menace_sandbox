"""Intent clustering and retrieval utilities."""

from __future__ import annotations

import ast
import io
import json
import logging
import math
import sqlite3
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from governed_embeddings import governed_embed
from universal_retriever import UniversalRetriever
from vector_utils import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class IntentMatch:
    """Structured search result."""

    path: str | None
    similarity: float
    cluster_ids: List[int]


def _normalize(vec: Iterable[float]) -> List[float]:
    """Return unit normalized copy of ``vec``."""

    lst = list(vec)
    norm = math.sqrt(sum(x * x for x in lst))
    return [x / norm for x in lst] if norm else lst


SCHEMA_VERSION = 2


class IntentClusterer:
    """Index repository modules and search by high level intent.

    The class builds a lightweight vector index of Python modules by
    extracting natural language signals such as docstrings, comments and
    structure.  The resulting vectors are persisted via an injected
    :class:`UniversalRetriever` instance which abstracts the underlying
    storage backend.
    """

    def __init__(self, retriever: UniversalRetriever) -> None:
        self.retriever = retriever
        self.root: Path | None = None
        self.cluster_map: Dict[str, int] = {}
        self._index: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    def _collect_intent(self, module_path: Path) -> tuple[str, Dict[str, Any]]:
        """Return intent text and metadata for ``module_path``."""

        try:
            with tokenize.open(module_path) as fh:
                source = fh.read()
        except OSError:
            return "", {"path": str(module_path), "names": []}

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return "", {"path": str(module_path), "names": []}

        docstrings: List[str] = []
        names: List[str] = []

        mod_doc = ast.get_docstring(tree)
        if mod_doc:
            docstrings.append(mod_doc)

        # collect comment-only lines and track code lines
        comment_map: dict[int, list[str]] = {}
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
                # gather preceding comment-only lines
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

        intent_text = "\n".join(docstrings + names + comments)
        metadata = {
            "path": str(module_path),
            "names": names,
        }
        if docstrings:
            metadata["docstrings"] = docstrings
        if comments:
            metadata["comments"] = comments
        return intent_text, metadata

    # ------------------------------------------------------------------
    def index_repository(self, root_path: str | Path) -> None:
        """Embed and store intent vectors for modules under ``root_path``."""
        root = Path(root_path)
        self.root = root
        # determine cluster membership using module_graph_analyzer or synergy grapher
        try:
            from module_graph_analyzer import build_import_graph, cluster_modules
            graph = build_import_graph(root)
            self.cluster_map = cluster_modules(graph)
        except Exception:
            self.cluster_map = {}
            try:  # pragma: no cover - optional dependency
                from module_synergy_grapher import ModuleSynergyGrapher
                from module_graph_analyzer import cluster_modules as _cluster
                grapher = ModuleSynergyGrapher(root=root)
                if grapher.graph is None:
                    graph = build_import_graph(root)
                else:
                    graph = grapher.graph
                self.cluster_map = _cluster(graph)
            except Exception:
                pass

        current_paths: set[str] = set()
        for file in root.rglob("*.py"):
            if any(part in {"tests", "test", "config", "configs"} for part in file.parts):
                continue
            rel = file.relative_to(root) if file.is_relative_to(root) else file
            rel_path = rel.as_posix()
            current_paths.add(rel_path)
            mtime = file.stat().st_mtime
            existing = self._index.get(rel_path)
            if existing and existing.get("mtime", 0) >= mtime:
                continue
            try:
                intent_text, metadata = self._collect_intent(file)
            except Exception:  # pragma: no cover - parse errors shouldn't abort indexing
                logger.exception("failed to parse %s", file)
                continue
            vector = governed_embed(intent_text)
            if vector is None:
                continue
            metadata["path"] = rel_path
            # map file to module name for cluster lookup
            try:
                rel_mod = rel.with_suffix("") if rel.name != "__init__.py" else rel.parent
                mod_key = rel_mod.as_posix()
            except Exception:
                mod_key = rel_path
            cid = self.cluster_map.get(mod_key)
            if cid is not None:
                metadata["cluster_id"] = cid
            try:
                self.retriever.add_vector(vector, metadata)
            except Exception:  # pragma: no cover - best effort persistence
                logger.exception("failed to store vector for %s", file)
                continue
            self._index[rel_path] = {
                "vector": vector,
                "metadata": metadata,
                "mtime": mtime,
            }

        # drop entries for files that no longer exist
        for stale in set(self._index) - current_paths:
            self._index.pop(stale, None)

    # ------------------------------------------------------------------
    def save_index(self, path: str | Path) -> None:
        """Persist the current intent index to ``path``."""

        if not self._index:
            return
        p = Path(path)
        conn = sqlite3.connect(p)
        with conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    path TEXT PRIMARY KEY,
                    vector TEXT,
                    metadata TEXT,
                    mtime REAL
                )
                """
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS clusters (module TEXT PRIMARY KEY, cluster_id INTEGER)"
            )
            conn.execute("DELETE FROM vectors")
            for mod_path, rec in self._index.items():
                conn.execute(
                    "INSERT OR REPLACE INTO vectors(path, vector, metadata, mtime) VALUES (?,?,?,?)",
                    (
                        mod_path,
                        json.dumps(rec.get("vector")),
                        json.dumps(rec.get("metadata")),
                        float(rec.get("mtime", 0.0)),
                    ),
                )
            conn.execute("DELETE FROM clusters")
            conn.executemany(
                "INSERT OR REPLACE INTO clusters(module, cluster_id) VALUES (?,?)",
                list(self.cluster_map.items()),
            )
            conn.execute("DELETE FROM meta")
            conn.execute(
                "INSERT INTO meta(key, value) VALUES ('version', ?)",
                (SCHEMA_VERSION,),
            )
            if self.root is not None:
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('root', ?)",
                    (self.root.as_posix(),),
                )
        conn.close()

    # ------------------------------------------------------------------
    def load_index(self, path: str | Path) -> None:
        """Load an intent index previously stored via :meth:`save_index`."""

        p = Path(path)
        if not p.exists():
            return
        conn = sqlite3.connect(p)
        with conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS vectors (path TEXT PRIMARY KEY, vector TEXT, metadata TEXT)"
            )
            cols = [r[1] for r in conn.execute("PRAGMA table_info(vectors)")]
            if "mtime" not in cols:
                conn.execute("ALTER TABLE vectors ADD COLUMN mtime REAL DEFAULT 0")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS clusters (module TEXT PRIMARY KEY, cluster_id INTEGER)"
            )
            row = conn.execute("SELECT value FROM meta WHERE key='version'").fetchone()
            version = int(row[0]) if row else 1
            if version < SCHEMA_VERSION:
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES ('version', ?)",
                    (SCHEMA_VERSION,),
                )
            root_row = conn.execute("SELECT value FROM meta WHERE key='root'").fetchone()
            self.root = Path(root_row[0]) if root_row and root_row[0] else None
            self.cluster_map = {
                m: int(cid)
                for m, cid in conn.execute("SELECT module, cluster_id FROM clusters")
            }
            self._index = {}
            for rel_path, vec_json, meta_json, mtime in conn.execute(
                "SELECT path, vector, metadata, mtime FROM vectors"
            ):
                try:
                    vector = json.loads(vec_json) if vec_json else []
                    metadata = json.loads(meta_json) if meta_json else {}
                    self._index[rel_path] = {
                        "vector": vector,
                        "metadata": metadata,
                        "mtime": float(mtime or 0.0),
                    }
                    self.retriever.add_vector(vector, metadata)
                except Exception:  # pragma: no cover - best effort load
                    logger.exception("failed to load vector for %s", rel_path)
        conn.close()

    # ------------------------------------------------------------------
    def get_cluster_intents(self, cluster_id: int) -> tuple[str, List[float] | None]:
        """Return combined intent text and embedding for ``cluster_id``."""

        if not self.root or not self.cluster_map:
            return "", None
        modules = [m for m, cid in self.cluster_map.items() if cid == cluster_id]
        texts: List[str] = []
        for mod in modules:
            path = self.root / f"{mod}.py"
            if not path.exists():
                path = self.root / mod / "__init__.py"
            if path.exists():
                try:
                    text, _meta = self._collect_intent(path)
                    texts.append(text)
                except Exception:  # pragma: no cover - best effort
                    continue
        combined = "\n".join(texts)
        vector = governed_embed(combined) if combined else None
        return combined, vector

    # ------------------------------------------------------------------
    def find_modules_related_to(self, query: str, top_k: int = 10) -> List[Any]:
        """Return modules or clusters with intent similar to ``query``."""

        vector = governed_embed(query)
        if vector is None:
            return []

        q = query.lower()
        if any(term in q for term in {"group", "cluster"}) and self.cluster_map:
            results: List[Dict[str, Any]] = []
            for cid in set(self.cluster_map.values()):
                _text, cvec = self.get_cluster_intents(cid)
                if not cvec:
                    continue
                score = cosine_similarity(vector, cvec)
                results.append({"cluster_id": cid, "score": score})
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        try:
            search = getattr(self.retriever, "search")
            return list(search(vector, top_k=top_k))
        except AttributeError:
            hits, *_rest = self.retriever.retrieve(vector, top_k=top_k)
            return list(hits)

    # ------------------------------------------------------------------
    def query(
        self,
        query_text: str,
        *,
        include_clusters: bool = True,
        threshold: float = 0.5,
    ) -> List[IntentMatch]:
        """Return ranked intent matches for ``query_text``.

        The method embeds and normalises ``query_text`` before computing
        cosine similarity with stored module vectors.  If no module level
        matches meet ``threshold`` and ``include_clusters`` is ``True`` a
        secondary pass over cluster summaries is performed.
        """

        vector = governed_embed(query_text)
        if vector is None:
            return []
        qvec = _normalize(vector)

        try:
            search = getattr(self.retriever, "search")
            candidates = list(search(qvec, top_k=20))
        except AttributeError:
            hits, *_rest = self.retriever.retrieve(qvec, top_k=20)
            candidates = list(hits)

        results: List[IntentMatch] = []
        for cand in candidates:
            meta = cand.get("metadata", cand)
            cvec = cand.get("vector") or meta.get("vector")
            if cvec is not None:
                score = sum(x * y for x, y in zip(qvec, _normalize(cvec)))
            else:
                score = cand.get("score")
            if score is None or score < threshold:
                continue
            path = meta.get("path")
            cid = meta.get("cluster_id")
            clusters = [cid] if include_clusters and cid is not None else []
            results.append(IntentMatch(path=path, similarity=score, cluster_ids=clusters))

        if results:
            results.sort(key=lambda r: r.similarity, reverse=True)
            return results

        if include_clusters and self.cluster_map:
            cluster_results: List[IntentMatch] = []
            for cid in set(self.cluster_map.values()):
                _text, cvec = self.get_cluster_intents(cid)
                if not cvec:
                    continue
                score = sum(x * y for x, y in zip(qvec, _normalize(cvec)))
                if score >= threshold:
                    cluster_results.append(
                        IntentMatch(path=None, similarity=score, cluster_ids=[cid])
                    )
            cluster_results.sort(key=lambda r: r.similarity, reverse=True)
            return cluster_results

        return []


__all__ = ["IntentClusterer", "IntentMatch"]
