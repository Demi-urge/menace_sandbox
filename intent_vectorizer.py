from __future__ import annotations

"""Utilities for extracting high level intent from Python modules.

The :class:`IntentVectorizer` reads a module file and collects natural language
signals such as docstrings, function or class names and leading comments.  The
resulting text bundle can be embedded via ``SentenceTransformer`` or the
lightweight local model bundled with :mod:`vector_service`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
import ast
import io
import tokenize

from governed_embeddings import governed_embed
try:  # pragma: no cover - optional heavy dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - allow running without dependency
    SentenceTransformer = None  # type: ignore

try:  # pragma: no cover - reuse tiny local model when transformers absent
    from vector_service.vectorizer import _local_embed  # type: ignore
except Exception:  # pragma: no cover - minimal fallback returning zeros
    def _local_embed(_text: str) -> List[float]:
        return []


@dataclass
class IntentVectorizer:
    """Extract docstrings, names and comments from Python modules."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        self._model: SentenceTransformer | None = None
        if SentenceTransformer is not None:
            try:  # pragma: no cover - model load heavy
                from huggingface_hub import login
                import os

                login(token=os.getenv("HUGGINGFACE_API_TOKEN"))
                self._model = SentenceTransformer(self.model_name)
            except Exception:  # pragma: no cover - best effort
                self._model = None

    # ------------------------------------------------------------------
    def _encode(self, text: str) -> List[float]:
        if self._model is not None:
            vec = governed_embed(text, self._model)
            if vec is not None:
                return [float(x) for x in vec]
        return _local_embed(text)

    # ------------------------------------------------------------------
    def bundle(self, path: str | Path) -> str:
        """Return intent text extracted from ``path``."""

        module_path = Path(path)
        try:
            with tokenize.open(module_path) as fh:
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

    # ------------------------------------------------------------------
    def transform(self, rec: Dict[str, Any]) -> List[float]:
        """Return an embedding vector for ``rec``.

        ``rec`` must provide a ``path`` pointing to a Python module.
        """

        path = rec.get("path") or rec.get("module_path") or rec.get("file")
        if not path:
            return []
        text = self.bundle(path)
        if not text:
            return []
        return self._encode(text)


DB_MODULE = "intent_db"
DB_CLASS = "IntentDB"

__all__ = ["IntentVectorizer"]
