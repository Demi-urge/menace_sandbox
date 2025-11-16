"""Semantic domain parser using NLP to detect disguised high-risk domains.

This module augments the static :mod:`risk_domain_classifier` by using
semantic similarity to detect when log entries reference risky domains in
indirect language. It prefers a Sentence-BERT model but falls back to a
simple TF–IDF approach if embeddings are unavailable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import logging
import os

import numpy as np

from governed_embeddings import governed_embed, get_embedder
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore

logger = logging.getLogger(__name__)


def _parse_timeout(env_var: str, default: float) -> float:
    """Return a positive float from *env_var* or ``default`` when invalid."""

    raw = os.getenv(env_var, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except Exception:
        logger.warning("invalid %s value %r; using %.1fs", env_var, raw, default)
        return default
    if value < 0:
        logger.warning("%s must be positive; using %.1fs", env_var, default)
        return default
    return value

# Reference domain anchors used for semantic matching.
TARGET_ANCHORS: List[str] = [
    "military",
    "pharma",
    "lobbying",
    "surveillance",
    "crypto_schemes",
    "political influence",
]

_MODEL: Any | None = None
_VECTORIZER: TfidfVectorizer | None = None
_ANCHOR_VECS: Any = None
_METHOD: str | None = None
_EMBEDDER_TIMEOUT = _parse_timeout("NLP_DOMAIN_EMBEDDER_TIMEOUT", 5.0)


def load_model() -> str:
    """Initialise the NLP model and return the chosen method.

    The function first attempts to load the SBERT model
    ``"all-MiniLM-L6-v2"``.  If sentence-transformers is unavailable or the
    model cannot be loaded, a fallback TF–IDF vectoriser is created instead.
    """

    global _MODEL, _VECTORIZER, _ANCHOR_VECS, _METHOD

    if _METHOD:
        return _METHOD

    embedder = get_embedder(timeout=_EMBEDDER_TIMEOUT)
    if embedder is not None:
        try:
            _MODEL = embedder
            vecs = [governed_embed(kw, _MODEL) for kw in TARGET_ANCHORS]
            if any(v is None for v in vecs):
                raise RuntimeError("failed to embed anchors")
            _ANCHOR_VECS = [np.array(v) for v in vecs]  # type: ignore[list-item]
            _METHOD = "sbert"
            logger.debug("Loaded governed embedder for domain parsing")
            return _METHOD
        except Exception as exc:  # pragma: no cover - runtime issues
            logger.warning("Failed to load governed embedder: %s", exc)
            _MODEL = None

    if TfidfVectorizer is None:
        raise RuntimeError("No NLP backend available for domain parsing")

    _VECTORIZER = TfidfVectorizer().fit(TARGET_ANCHORS)
    _ANCHOR_VECS = _VECTORIZER.transform(TARGET_ANCHORS)
    _METHOD = "tfidf"
    logger.debug("Using TF-IDF vectoriser as fallback")
    return _METHOD


def _ensure_model() -> None:
    if _METHOD is None:
        load_model()


def _prepare_text(text_or_entry: str | Dict[str, Any]) -> str:
    if isinstance(text_or_entry, dict):
        domain = str(text_or_entry.get("target_domain", ""))
        desc = str(text_or_entry.get("action_description", ""))
        return f"{domain} {desc}".strip()
    return str(text_or_entry).strip()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def classify_text(text: str | Dict[str, Any]) -> List[Tuple[str, float]]:
    """Return the top three matching domain anchors for *text*.

    Parameters
    ----------
    text : str or dict
        Either a raw string to analyse or a log entry containing
        ``"target_domain"`` and optionally ``"action_description"``.
    """

    _ensure_model()
    query = _prepare_text(text)
    if not query:
        return []

    if _METHOD == "sbert" and _MODEL is not None:
        vec = governed_embed(query, _MODEL)
        if vec is None:
            return []
        arr = np.array(vec)
        scores = [
            _cosine_sim(arr, anchor_vec) for anchor_vec in _ANCHOR_VECS  # type: ignore[arg-type]
        ]
    else:
        assert _VECTORIZER is not None and cosine_similarity is not None
        vec = _VECTORIZER.transform([query])
        sims = cosine_similarity(vec, _ANCHOR_VECS)[0]
        scores = [float(s) for s in sims]

    pairs = sorted(zip(TARGET_ANCHORS, scores), key=lambda p: p[1], reverse=True)
    return pairs[:3]


def flag_if_similar(text: str | Dict[str, Any], threshold: float = 0.7) -> bool:
    """Return ``True`` if *text* is semantically close to any risky anchor."""

    results = classify_text(text)
    return any(score >= threshold for _, score in results)


__all__ = [
    "TARGET_ANCHORS",
    "load_model",
    "classify_text",
    "flag_if_similar",
]
