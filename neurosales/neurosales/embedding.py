from __future__ import annotations


try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional heavy deps
    SentenceTransformer = None  # type: ignore

import logging
from governed_embeddings import governed_embed
from analysis.semantic_diff_filter import find_semantic_risks
from security.secret_redactor import redact

logger = logging.getLogger(__name__)

_MODEL: SentenceTransformer | None = None
_DEFAULT_DIM = 384


def get_model() -> SentenceTransformer | None:
    global _MODEL
    if _MODEL is None and SentenceTransformer is not None:
        from huggingface_hub import login
        import os

        login(token=os.getenv("HUGGINGFACE_API_TOKEN"))
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL


def embed_text(text: str) -> list[float]:
    """Return the embedding vector for ``text``.

    ``sentence-transformers`` must be installed.  The previous behaviour
    returned a deterministic hash when the library was missing, but this
    hid errors in production.  Now a ``RuntimeError`` is raised so that the
    caller can ensure the dependency is available.  Before embedding, the
    text is scanned with :func:`find_semantic_risks` and a ``ValueError`` is
    raised when any unsafe patterns are detected so that callers can take
    appropriate action.
    """
    model = get_model()
    if model is None:
        raise RuntimeError("sentence-transformers is required for embed_text")

    # Guard against embedding text that matches known unsafe patterns.
    alerts = find_semantic_risks(text.splitlines())
    if alerts:
        logger.warning("semantic risks detected: %s", [a[1] for a in alerts])
        raise ValueError("Semantic risks detected")

    cleaned = redact(text)
    if cleaned != text:
        logger.warning("redacted secrets prior to embedding")
    vec = governed_embed(cleaned, model)
    if vec is None:
        raise RuntimeError("Embedding failed or skipped")
    return vec


def embedding_dimension() -> int:
    model = get_model()
    if model is not None:
        return model.get_sentence_embedding_dimension()
    return _DEFAULT_DIM

