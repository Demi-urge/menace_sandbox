from __future__ import annotations

from typing import List, Optional

import logging
import os

try:  # pragma: no cover - optional heavy dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - simplify in environments without the package
    SentenceTransformer = None  # type: ignore

from security.secret_redactor import redact
from compliance.license_fingerprint import (
    check as license_check,
    fingerprint as license_fingerprint,
)
from analysis.semantic_diff_filter import find_semantic_risks

logger = logging.getLogger(__name__)

_EMBEDDER: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer | None:
    """Return a lazily-instantiated shared :class:`SentenceTransformer`.

    The model is created on first use and reused for subsequent calls.  Errors
    during initialisation result in ``None`` being cached to avoid repeated
    expensive failures.
    """
    global _EMBEDDER
    if _EMBEDDER is None and SentenceTransformer is not None:
        token = os.getenv("HUGGINGFACE_API_TOKEN")
        if token:
            try:  # pragma: no cover - optional dependency may be missing
                from huggingface_hub import login
            except Exception as exc:  # pragma: no cover - import failure
                logger.warning("failed to import huggingface_hub: %s", exc)
            else:
                try:  # pragma: no cover - network interaction
                    login(token=token)
                except Exception as exc:  # pragma: no cover - hub issues
                    logger.warning("huggingface login failed: %s", exc)
        try:  # pragma: no cover - heavy model download
            _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:
            logger.warning("failed to initialise sentence transformer: %s", exc)
            _EMBEDDER = None
    return _EMBEDDER


def governed_embed(text: str, embedder: SentenceTransformer | None = None) -> Optional[List[float]]:
    """Return an embedding vector for ``text`` with safety checks.

    The input text is first scanned for disallowed licences.  If any are
    detected the function returns ``None``.  Secrets are redacted before
    computing the embedding to avoid storing sensitive data in the vector
    space.  Any runtime failures during embedding are swallowed and ``None``
    is returned.
    """

    if not text:
        return None
    lic = license_check(text)
    if lic:
        try:  # pragma: no cover - best effort logging
            logger.warning(
                "skipping embedding due to license %s", lic,
                extra={"fingerprint": license_fingerprint(text)},
            )
        except Exception:
            logger.warning("skipping embedding due to license %s", lic)
        return None

    risks = find_semantic_risks(text.splitlines())
    if risks:
        logger.warning("skipping embedding due to semantic risks: %s", [r[1] for r in risks])
        return None
    cleaned = redact(text)
    if cleaned != text:
        logger.warning("redacted secrets prior to embedding")
    model = embedder or get_embedder()
    if model is None:
        return None
    try:  # pragma: no cover - external model may fail at runtime
        return model.encode([cleaned])[0].tolist()
    except Exception:
        return None


__all__ = ["governed_embed", "get_embedder"]
