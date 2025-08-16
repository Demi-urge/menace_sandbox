from __future__ import annotations


try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:  # pragma: no cover - optional heavy deps
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore

_MODEL: SentenceTransformer | None = None
_DEFAULT_DIM = 384


def get_model() -> SentenceTransformer | None:
    global _MODEL
    if _MODEL is None and SentenceTransformer is not None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def embed_text(text: str) -> list[float]:
    """Return the embedding vector for ``text``.

    ``sentence-transformers`` must be installed.  The previous behaviour
    returned a deterministic hash when the library was missing, but this
    hid errors in production.  Now a ``RuntimeError`` is raised so that the
    caller can ensure the dependency is available.
    """
    model = get_model()
    if model is None or np is None:
        raise RuntimeError("sentence-transformers is required for embed_text")
    arr = model.encode(text, convert_to_numpy=True).astype("float32")
    return arr.tolist()


def embedding_dimension() -> int:
    model = get_model()
    if model is not None:
        return model.get_sentence_embedding_dimension()
    return _DEFAULT_DIM

