"""Intent clustering and retrieval utilities."""

from __future__ import annotations

import ast
import io
import logging
import math
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

    # ------------------------------------------------------------------
    def _collect_intent(self, module_path: Path) -> tuple[str, Dict[str, Any]]:
        """Return intent text and metadata for ``module_path``."""

        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        docstrings: List[str] = []
        names: List[str] = []

        mod_doc = ast.get_docstring(tree)
        if mod_doc:
            docstrings.append(mod_doc)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name)
                doc = ast.get_docstring(node)
                if doc:
                    docstrings.append(doc)

        comments: List[str] = []
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                comment = tok.string.lstrip("#").strip()
                if comment:
                    comments.append(comment)

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

        for file in root.rglob("*.py"):
            if any(part in {"tests", "test", "config", "configs"} for part in file.parts):
                continue
            try:
                intent_text, metadata = self._collect_intent(file)
            except Exception:  # pragma: no cover - parse errors shouldn't abort indexing
                logger.exception("failed to parse %s", file)
                continue
            vector = governed_embed(intent_text)
            if vector is None:
                continue
            rel = file.relative_to(root) if file.is_relative_to(root) else file
            metadata["path"] = rel.as_posix()
            # map file to module name for cluster lookup
            try:
                rel_mod = rel.with_suffix("") if rel.name != "__init__.py" else rel.parent
                mod_key = rel_mod.as_posix()
            except Exception:
                mod_key = rel.as_posix()
            cid = self.cluster_map.get(mod_key)
            if cid is not None:
                metadata["cluster_id"] = cid
            try:
                self.retriever.add_vector(vector, metadata)
            except Exception:  # pragma: no cover - best effort persistence
                logger.exception("failed to store vector for %s", file)

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
