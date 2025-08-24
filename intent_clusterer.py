"""Intent clustering and retrieval utilities."""

from __future__ import annotations

import ast
import io
import logging
import tokenize
from pathlib import Path
from typing import Any, Dict, Iterable, List

from governed_embeddings import governed_embed
from universal_retriever import UniversalRetriever

logger = logging.getLogger(__name__)


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
            metadata["path"] = str(file.relative_to(root)) if file.is_relative_to(root) else str(file)
            try:
                self.retriever.add_vector(vector, metadata)
            except Exception:  # pragma: no cover - best effort persistence
                logger.exception("failed to store vector for %s", file)

    # ------------------------------------------------------------------
    def find_modules_related_to(self, query: str, top_k: int = 10) -> List[Any]:
        """Return modules with intent similar to ``query``."""

        vector = governed_embed(query)
        if vector is None:
            return []
        try:
            search = getattr(self.retriever, "search")
            return list(search(vector, top_k=top_k))
        except AttributeError:
            hits, *_rest = self.retriever.retrieve(vector, top_k=top_k)
            return list(hits)


__all__ = ["IntentClusterer"]
