from __future__ import annotations

"""Lazy bootstrap helpers for vector service assets.

This module centralises deferred initialisation for heavyweight resources such
as the bundled embedding model and the embedding scheduler.  Callers can either
rely on the on-demand helpers (which cache results) or invoke the warmup
routine to pre-populate caches before the first real request.
"""

import importlib.util
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable

import metrics_exporter as _metrics

try:  # pragma: no cover - lightweight import wrapper
    from dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback when run standalone
    resolve_path = Path  # type: ignore[assignment]


_MODEL_LOCK = threading.Lock()
_MODEL_READY = False
_SCHEDULER_LOCK = threading.Lock()
_SCHEDULER: Any | None | bool = None  # False means attempted and unavailable

VECTOR_WARMUP_STAGE_TOTAL = getattr(
    _metrics,
    "vector_warmup_stage_total",
    _metrics.Gauge(
        "vector_warmup_stage_total",
        "Vector warmup stage results by status",
        ["stage", "status"],
    ),
)


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
    download_model: bool = False,
    probe_model: bool = False,
    hydrate_handlers: bool = False,
    start_scheduler: bool = False,
    run_vectorise: bool | None = None,
    check_budget: Callable[[], None] | None = None,
    logger: logging.Logger | None = None,
    force_heavy: bool = False,
    bootstrap_fast: bool | None = None,
    warmup_lite: bool = False,
    warmup_model: bool | None = None,
    warmup_handlers: bool | None = None,
    warmup_probe: bool | None = None,
) -> None:
    """Eagerly initialise vector assets and caches.

    The default behaviour favours a "light" warmup that validates scheduler
    configuration and optional model presence without instantiating
    ``SharedVectorService``.  Callers may opt-in to handler hydration and
    vectorisation by setting ``hydrate_handlers=True`` (and optionally
    ``run_vectorise=True``) when a heavier warmup is desired.

    When ``bootstrap_fast`` is True the vector service keeps the patch handler
    stubbed during warmup and avoids loading heavy indexes.  Setting
    ``warmup_lite`` further short-circuits handler hydration and vector store
    acquisition so readiness checks do not incur heavy startup costs.
    """

    log = logger or logging.getLogger(__name__)
    bootstrap_context = any(
        os.getenv(flag, "").strip().lower() in {"1", "true", "yes", "on"}
        for flag in ("MENACE_BOOTSTRAP", "MENACE_BOOTSTRAP_FAST", "MENACE_BOOTSTRAP_MODE")
    )

    fast_vector_env = os.getenv("MENACE_VECTOR_WARMUP_FAST", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if warmup_model is not None:
        download_model = warmup_model
    if warmup_handlers is not None:
        hydrate_handlers = warmup_handlers
    if warmup_probe is not None:
        probe_model = warmup_probe

    if fast_vector_env and not force_heavy:
        if download_model:
            log.info("Fast vector warmup requested; skipping embedding model download")
        if probe_model:
            log.info("Fast vector warmup requested; skipping embedding model probe")
        if hydrate_handlers:
            log.info("Fast vector warmup requested; skipping handler hydration")
        download_model = False
        probe_model = False
        hydrate_handlers = False
        run_vectorise = False

    if bootstrap_context and not force_heavy:
        if download_model:
            log.info("Bootstrap context detected; skipping embedding model download")
            download_model = False
        if hydrate_handlers:
            log.info("Bootstrap context detected; deferring handler hydration")
            hydrate_handlers = False
        if start_scheduler:
            log.info("Bootstrap context detected; scheduler start deferred")
            start_scheduler = False

    summary: dict[str, str] = {}

    def _record(stage: str, status: str) -> None:
        summary[stage] = status
        try:
            VECTOR_WARMUP_STAGE_TOTAL.labels(stage, status).inc()
        except Exception:  # pragma: no cover - metrics best effort
            log.debug("failed emitting vector warmup metric", exc_info=True)

    budget_exhausted = False

    def _guard(stage: str) -> bool:
        nonlocal budget_exhausted
        if budget_exhausted:
            _record(stage, "skipped-budget")
            log.info("Vector warmup budget already exhausted; skipping %s", stage)
            return False
        if check_budget is None:
            return True
        try:
            check_budget()
            log.debug("vector warmup budget check after %s", stage)
            return True
        except TimeoutError as exc:
            budget_exhausted = True
            _record(stage, "skipped-budget")
            log.warning("Vector warmup deadline reached before %s: %s", stage, exc)
            return False

    _guard("init")
    if _guard("model"):
        if download_model:
            path = ensure_embedding_model(logger=log, warmup=True)
            if path:
                _record("model", f"ready:{path}" if path.exists() else "ready")
            else:
                _record("model", "missing")
        elif probe_model:
            dest = _model_bundle_path()
            if dest.exists():
                log.info("embedding model already present at %s (probe only)", dest)
                _record("model", "present")
            else:
                log.info("embedding model probe: archive missing; will fetch on demand")
                _record("model", "absent-probe")
        else:
            log.info("Skipping embedding model download (disabled)")
            _record("model", "skipped")

    svc = None
    if _guard("handlers"):
        if hydrate_handlers:
            try:
                from .vectorizer import SharedVectorService

                svc = SharedVectorService(
                    bootstrap_fast=bootstrap_fast,
                    warmup_lite=warmup_lite,
                )
                _record("handlers", "hydrated")
            except Exception as exc:  # pragma: no cover - best effort logging
                log.warning("SharedVectorService warmup failed: %s", exc)
                _record("handlers", "failed")
        else:
            log.info("Vector handler hydration skipped")
            _record("handlers", "skipped")

    if _guard("scheduler"):
        if start_scheduler:
            ensure_scheduler_started(logger=log)
            _record("scheduler", "started")
        else:
            log.info("Scheduler warmup skipped")
            _record("scheduler", "skipped")

    should_vectorise = run_vectorise if run_vectorise is not None else hydrate_handlers
    if _guard("vectorise"):
        if should_vectorise and svc is not None:
            try:
                svc.vectorise("text", {"text": "warmup"})
                _record("vectorise", "ok")
            except Exception:  # pragma: no cover - allow partial warmup
                log.debug("vector warmup transform failed; continuing", exc_info=True)
                _record("vectorise", "failed")
        else:
            if should_vectorise:
                _record("vectorise", "skipped-no-service")
                log.info("Vectorise warmup skipped: service unavailable")
            else:
                _record("vectorise", "skipped")
                log.info("Vectorise warmup skipped")

    log.info(
        "vector warmup stages recorded", extra={"event": "vector-warmup", "warmup": summary}
    )
    log.debug("vector warmup summary: %s", summary)


__all__ = ["ensure_embedding_model", "ensure_scheduler_started", "warmup_vector_service"]
