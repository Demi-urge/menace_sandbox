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
import numpy as np

try:  # pragma: no cover - heavy dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - package may be missing
    SentenceTransformer = None  # type: ignore

try:  # pragma: no cover - optional service
    from vector_service.vectorizer import SharedVectorService  # type: ignore
except Exception:  # pragma: no cover - dependency may be missing
    SharedVectorService = None  # type: ignore

_MODEL = None
EMBED_DIM = 384
if SentenceTransformer is not None:  # pragma: no cover - model download may be slow
    try:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        EMBED_DIM = int(_MODEL.get_sentence_embedding_dimension())
    except Exception:
        _MODEL = None

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

    mdl = model or _MODEL
    if mdl is not None:
        vecs = mdl.encode(texts)  # type: ignore[arg-type]
        return [list(map(float, v)) for v in np.atleast_2d(vecs)]

    svc = service
    global _SERVICE
    if svc is None and SharedVectorService is not None:
        if _SERVICE is None:
            try:
                embedder = None
                if SentenceTransformer is not None:
                    try:
                        embedder = SentenceTransformer("all-MiniLM-L6-v2")
                    except Exception:
                        embedder = None
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
