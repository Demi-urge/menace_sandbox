from __future__ import annotations

"""Lazy bootstrap helpers for vector service assets.

This module centralises deferred initialisation for heavyweight resources such
as the bundled embedding model and the embedding scheduler.  Callers can either
rely on the on-demand helpers (which cache results) or invoke the warmup
routine to pre-populate caches before the first real request.
"""

import importlib.util
import logging
import threading
from pathlib import Path
from typing import Any, Callable

try:  # pragma: no cover - lightweight import wrapper
    from dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback when run standalone
    resolve_path = Path  # type: ignore[assignment]


_MODEL_LOCK = threading.Lock()
_MODEL_READY = False
_SCHEDULER_LOCK = threading.Lock()
_SCHEDULER: Any | None | bool = None  # False means attempted and unavailable


def _model_bundle_path() -> Path:
    return resolve_path("vector_service/minilm/tiny-distilroberta-base.tar.xz")


def ensure_embedding_model(*, logger: logging.Logger | None = None, warmup: bool = False) -> Path | None:
    """Ensure the bundled embedding model archive exists.

    The download is performed at most once per process and only when the model
    is missing.  When ``warmup`` is False the function favours fast failure so
    first-use callers can fall back gracefully; during warmup we log and swallow
    errors to avoid breaking bootstrap flows.
    """

    global _MODEL_READY
    log = logger or logging.getLogger(__name__)
    if _MODEL_READY:
        return _model_bundle_path()

    with _MODEL_LOCK:
        if _MODEL_READY:
            return _model_bundle_path()
        dest = _model_bundle_path()
        if dest.exists():
            _MODEL_READY = True
            return dest

        if importlib.util.find_spec("huggingface_hub") is None:
            log.info(
                "embedding model download skipped (huggingface-hub unavailable); will retry on demand"
            )
            return None

        try:
            from . import download_model as _dm

            _dm.bundle(dest)
            _MODEL_READY = True
            return dest
        except Exception as exc:  # pragma: no cover - best effort during warmup
            log.warning("embedding model bootstrap failed: %s", exc)
            if warmup:
                return None
            raise


def ensure_scheduler_started(*, logger: logging.Logger | None = None) -> Any | None:
    """Start the embedding scheduler once and cache the result."""

    global _SCHEDULER
    log = logger or logging.getLogger(__name__)
    with _SCHEDULER_LOCK:
        if _SCHEDULER is not None:
            return None if _SCHEDULER is False else _SCHEDULER
        try:
            from .embedding_scheduler import start_scheduler_from_env

            _SCHEDULER = start_scheduler_from_env()
            return _SCHEDULER
        except Exception as exc:  # pragma: no cover - defensive logging
            log.warning("embedding scheduler warmup failed: %s", exc)
            _SCHEDULER = False
            return None


def warmup_vector_service(
    *,
    download_model: bool = True,
    probe_model: bool = False,
    hydrate_handlers: bool = False,
    start_scheduler: bool = False,
    run_vectorise: bool | None = None,
    check_budget: Callable[[], None] | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Eagerly initialise vector assets and caches.

    The default behaviour favours a "light" warmup that validates scheduler
    configuration and optional model presence without instantiating
    ``SharedVectorService``.  Callers may opt-in to handler hydration and
    vectorisation by setting ``hydrate_handlers=True`` (and optionally
    ``run_vectorise=True``) when a heavier warmup is desired.
    """

    log = logger or logging.getLogger(__name__)

    def _guard(stage: str) -> None:
        if check_budget is None:
            return
        check_budget()
        log.debug("vector warmup budget check after %s", stage)

    summary: dict[str, str] = {}

    _guard("init")
    if download_model:
        path = ensure_embedding_model(logger=log, warmup=True)
        if path:
            summary["model"] = f"ready:{path}" if path.exists() else "ready"
        else:
            summary["model"] = "missing"
    elif probe_model:
        dest = _model_bundle_path()
        if dest.exists():
            log.info("embedding model already present at %s (probe only)", dest)
            summary["model"] = "present"
        else:
            log.info("embedding model probe: archive missing; will fetch on demand")
            summary["model"] = "absent-probe"
    else:
        log.info("Skipping embedding model download (disabled)")
        summary["model"] = "skipped"

    _guard("model")

    svc = None
    if hydrate_handlers:
        try:
            from .vectorizer import SharedVectorService

            svc = SharedVectorService()
            summary["handlers"] = "hydrated"
        except Exception as exc:  # pragma: no cover - best effort logging
            log.warning("SharedVectorService warmup failed: %s", exc)
            summary["handlers"] = "failed"
    else:
        log.info("Vector handler hydration skipped")
        summary["handlers"] = "skipped"

    _guard("handlers")

    if start_scheduler:
        ensure_scheduler_started(logger=log)
        summary["scheduler"] = "started"
    else:
        log.info("Scheduler warmup skipped")
        summary["scheduler"] = "skipped"

    _guard("scheduler")

    should_vectorise = run_vectorise if run_vectorise is not None else hydrate_handlers
    if should_vectorise and svc is not None:
        try:
            svc.vectorise("text", {"text": "warmup"})
            summary["vectorise"] = "ok"
        except Exception:  # pragma: no cover - allow partial warmup
            log.debug("vector warmup transform failed; continuing", exc_info=True)
            summary["vectorise"] = "failed"
    else:
        if should_vectorise:
            summary["vectorise"] = "skipped-no-service"
            log.info("Vectorise warmup skipped: service unavailable")
        else:
            summary["vectorise"] = "skipped"
            log.info("Vectorise warmup skipped")

    log.debug("vector warmup summary: %s", summary)


__all__ = ["ensure_embedding_model", "ensure_scheduler_started", "warmup_vector_service"]
