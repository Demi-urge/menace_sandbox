from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

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
_MODEL_NAME = "all-MiniLM-L6-v2"


def _cache_base() -> Optional[Path]:
    """Return the configured Hugging Face cache directory when available."""

    for env in ("TRANSFORMERS_CACHE", "HF_HOME"):
        loc = os.getenv(env)
        if loc:
            return Path(loc).expanduser()
    default = Path.home() / ".cache" / "huggingface"
    return default if default.exists() else None


def _cached_model_path(cache_dir: Path, model_name: str) -> Path:
    """Return the expected cache path for ``model_name`` within ``cache_dir``."""

    safe_name = model_name.replace("/", "--")
    return cache_dir / "hub" / f"models--sentence-transformers--{safe_name}"


def _load_embedder() -> SentenceTransformer | None:
    """Load the shared ``SentenceTransformer`` instance with offline fallbacks."""

    if SentenceTransformer is None:  # pragma: no cover - optional dependency missing
        return None

    cache_dir = _cache_base()
    local_kwargs: dict[str, object] = {}
    if cache_dir is not None:
        local_kwargs["cache_folder"] = str(cache_dir)
        model_cache = _cached_model_path(cache_dir, _MODEL_NAME)
        if os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"):
            local_kwargs["local_files_only"] = True
        elif model_cache.exists():
            # We already have the files locally; avoid slow network calls.
            local_kwargs["local_files_only"] = True

    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if token:
        # The transformers stack honours these environment variables directly and
        # avoids the interactive ``huggingface_hub.login`` flow that can hang in
        # restricted environments.
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", token)
        os.environ.setdefault("HF_HUB_TOKEN", token)

    try:
        return SentenceTransformer(_MODEL_NAME, **local_kwargs)
    except Exception as exc:
        if local_kwargs.pop("local_files_only", None):
            try:
                return SentenceTransformer(_MODEL_NAME, **local_kwargs)
            except Exception:
                logger.warning("failed to initialise sentence transformer: %s", exc)
                return None
        logger.warning("failed to initialise sentence transformer: %s", exc)
        return None


def get_embedder() -> SentenceTransformer | None:
    """Return a lazily-instantiated shared :class:`SentenceTransformer`.

    The model is created on first use and reused for subsequent calls.  Errors
    during initialisation result in ``None`` being cached to avoid repeated
    expensive failures.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = _load_embedder()
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
