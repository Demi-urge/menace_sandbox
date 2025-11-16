from __future__ import annotations

"""Shared helpers for text embeddings.

This module centralises the logic for obtaining embeddings for a list of
texts.  It performs a best effort to use a locally available model, fall back to
an optional ``SharedVectorService`` and finally to a remote HTTP endpoint.

The function :func:`get_text_embeddings` accepts optional ``model`` and
``service`` instances which allows tests to inject lightweight stubs.
"""

from typing import List
import json
import os
import urllib.request

import governed_embeddings

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy missing
    np = None  # type: ignore

try:  # pragma: no cover - heavy dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - package may be missing
    SentenceTransformer = None  # type: ignore

try:  # pragma: no cover - optional service
    from vector_service.vectorizer import SharedVectorService  # type: ignore
except Exception:  # pragma: no cover - dependency may be missing
    SharedVectorService = None  # type: ignore

_MODEL: "SentenceTransformer | None" = None
EMBED_DIM = 384

_SERVICE: "SharedVectorService | None" = None
_REMOTE_URL = os.environ.get("VECTOR_SERVICE_URL")


def _remote_embed(url: str, text: str) -> List[float]:
    data = json.dumps({"kind": "text", "record": {"text": text}}).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/vectorise",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:  # pragma: no cover - network
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("vector", [])


def _ensure_model() -> "SentenceTransformer | None":
    """Return a cached ``SentenceTransformer`` instance."""

    global _MODEL, EMBED_DIM
    if _MODEL is None:
        mdl = governed_embeddings.get_embedder()
        if mdl is not None:
            _MODEL = mdl
            try:
                EMBED_DIM = int(mdl.get_sentence_embedding_dimension())
            except Exception:
                pass
        else:
            _MODEL = None
    return _MODEL


def get_text_embeddings(
    texts: List[str],
    *,
    model: "SentenceTransformer | None" = None,
    service: "SharedVectorService | None" = None,
) -> List[List[float]]:
    """Return embeddings for ``texts``.

    The function tries a local ``SentenceTransformer`` model first.  If that is
    not available it falls back to a ``SharedVectorService`` instance.  As a last
    resort it attempts to call a remote HTTP service specified via the
    ``VECTOR_SERVICE_URL`` environment variable.
    """

    if not texts:
        return []

    mdl = model or _ensure_model()
    if mdl is not None:
        vecs = mdl.encode(texts)  # type: ignore[arg-type]
        if np is not None:
            arr = np.atleast_2d(vecs)
            return [list(map(float, v)) for v in arr]
        if hasattr(vecs, "tolist"):
            vecs = vecs.tolist()
        if isinstance(vecs, list) and vecs and isinstance(vecs[0], (list, tuple)):
            return [list(map(float, v)) for v in vecs]
        return [list(map(float, vecs))]

    svc = service
    global _SERVICE
    if svc is None and SharedVectorService is not None:
        if _SERVICE is None:
            try:
                embedder = model or _ensure_model()
                _SERVICE = SharedVectorService(embedder)  # type: ignore[misc]
            except Exception:
                _SERVICE = None
        svc = _SERVICE

    if svc is not None:
        try:
            return [svc.vectorise("text", {"text": t}) for t in texts]
        except Exception:
            pass

    url = _REMOTE_URL
    if url:
        try:
            return [_remote_embed(url, t) for t in texts]
        except Exception:
            pass

    raise RuntimeError("No embedding backend available")


__all__ = ["get_text_embeddings", "EMBED_DIM"]
