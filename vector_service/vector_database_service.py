"""Async wrapper around :class:`SharedVectorService`.

The module exposes a small HTTP API implemented with :mod:`FastAPI` that
provides two primary operations:

``/vectorise-and-store``
    Vectorise an arbitrary record and persist the resulting vector in the
    configured :class:`~vector_service.vector_store.VectorStore`.

``/search``
    Vectorise a query record and return the closest matches from the vector
    store.

The service is intended to run as a lightweight daemon.  The CLI entry
``python -m vector_service.vector_database_service`` starts a Uvicorn server
listening on ``VECTOR_SERVICE_HOST``/``VECTOR_SERVICE_PORT`` or on a Unix
domain socket specified by ``VECTOR_SERVICE_SOCKET``.  A bootstrap placeholder
is advertised during import so any late-loading helpers reuse the shared
promise instead of spawning redundant ``prepare_pipeline_for_bootstrap``
attempts when vector dependencies touch the automation pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable
import asyncio
import os
import threading
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

try:  # pragma: no cover - prefer package import when available
    from menace_sandbox.coding_bot_interface import (
        _bootstrap_dependency_broker,
        advertise_bootstrap_placeholder,
        get_active_bootstrap_pipeline,
    )
except Exception:  # pragma: no cover - flat execution fallback
    from coding_bot_interface import (  # type: ignore
        _bootstrap_dependency_broker,
        advertise_bootstrap_placeholder,
        get_active_bootstrap_pipeline,
    )

from .vectorizer import SharedVectorService
from .embedding_backfill import (
    watch_databases,
    check_staleness,
    ensure_embeddings_fresh,
    schedule_backfill,
    StaleEmbeddingsError,
)
from .embedding_scheduler import start_scheduler_from_env

logger = logging.getLogger(__name__)


def _embedding_all_zero(vec: Iterable[Any]) -> bool:
    seen_value = False
    for value in vec:
        seen_value = True
        try:
            if float(value) != 0.0:
                return False
        except (TypeError, ValueError):
            return False
    return seen_value


def _record_truncated(record: Dict[str, Any]) -> bool:
    if record.get("truncated") is True:
        return True
    meta = record.get("meta")
    if isinstance(meta, dict) and meta.get("truncated") is True:
        return True
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and metadata.get("truncated") is True:
        return True
    return False


# ---------------------------------------------------------------------------
# Service setup
# ---------------------------------------------------------------------------

app = FastAPI()
_svc = SharedVectorService()
_dependency_broker = _bootstrap_dependency_broker()
_active_pipeline, _active_manager = get_active_bootstrap_pipeline()
(
    _BOOTSTRAP_PLACEHOLDER_PIPELINE,
    _BOOTSTRAP_PLACEHOLDER_MANAGER,
) = advertise_bootstrap_placeholder(
    dependency_broker=_dependency_broker,
    pipeline=_active_pipeline,
    manager=_active_manager,
)
try:
    _svc.bootstrap_manager = _BOOTSTRAP_PLACEHOLDER_MANAGER  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - advisory only
    logger.debug("unable to attach bootstrap placeholder manager to vector service", exc_info=True)


def _watcher() -> None:
    """Background thread embedding newly added database records.

    ``watch_databases`` runs in a dedicated thread and blocks until it is
    stopped.  We wrap it in a loop so any unexpected errors are logged and the
    watcher is restarted after a short delay.
    """

    while True:
        try:
            check_staleness([])
            with watch_databases():
                # ``watch_databases`` yields control once the watcher thread is
                # running; block here until it terminates or raises.
                while True:
                    time.sleep(60)
        except Exception:  # pragma: no cover - best effort logging
            logger.exception("watch_databases terminated unexpectedly")
            time.sleep(5)


def _spawn_watcher() -> None:
    thread = threading.Thread(target=_watcher, daemon=True)
    thread.start()
    app.state.watch_thread = thread


async def _monitor_watcher() -> None:
    while True:  # pragma: no cover - best effort monitoring
        await asyncio.sleep(5)
        thread = getattr(app.state, "watch_thread", None)
        if thread is None or not thread.is_alive():
            logger.warning("watch_databases thread stopped; restarting")
            _spawn_watcher()


@app.on_event("startup")
async def _start_watcher() -> None:
    dbs = ["code", "bot", "error", "workflow"]
    try:
        await asyncio.to_thread(ensure_embeddings_fresh, dbs)
    except StaleEmbeddingsError as exc:
        await schedule_backfill(dbs=list(exc.stale_dbs))
        await asyncio.to_thread(ensure_embeddings_fresh, dbs)
    _spawn_watcher()
    app.state.monitor_task = asyncio.create_task(_monitor_watcher())
    app.state.embedding_scheduler = start_scheduler_from_env()


@app.on_event("shutdown")
async def _stop_scheduler() -> None:
    scheduler = getattr(app.state, "embedding_scheduler", None)
    if scheduler is not None:
        scheduler.stop()
    monitor = getattr(app.state, "monitor_task", None)
    if monitor is not None:
        monitor.cancel()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class VectoriseRequest(BaseModel):
    kind: str
    record: Dict[str, Any]


class AddRequest(VectoriseRequest):
    record_id: str
    origin_db: str | None = None
    metadata: Dict[str, Any] | None = None


class SearchRequest(VectoriseRequest):
    top_k: int = 5


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/vectorise-and-store")
async def vectorise_and_store(req: AddRequest) -> Dict[str, Any]:
    """Vectorise ``req.record`` and persist it in the vector store."""

    vec = await asyncio.to_thread(
        _svc.vectorise_and_store,
        req.kind,
        req.record_id,
        req.record,
        origin_db=req.origin_db,
        metadata=req.metadata,
    )
    return {"status": "ok", "vector": vec}


@app.post("/vectorise")
async def vectorise(req: VectoriseRequest) -> Dict[str, Any]:
    """Return an embedding for ``req.record`` without storing it."""

    vec = await asyncio.to_thread(_svc.vectorise, req.kind, req.record)
    return {"status": "ok", "vector": vec}


@app.post("/search")
async def search(req: SearchRequest) -> Dict[str, Any]:
    """Vectorise ``req.record`` and query the vector store."""

    vec = await asyncio.to_thread(_svc.vectorise, req.kind, req.record)
    if (
        not isinstance(vec, list)
        or not vec
        or _embedding_all_zero(vec)
        or _record_truncated(req.record)
    ):
        return {
            "status": "ok",
            "data": [],
            "warning": "Input truncated; results may be partial",
        }
    if _svc.vector_store is None:
        raise HTTPException(status_code=500, detail="Vector store not configured")
    results = await asyncio.to_thread(_svc.vector_store.query, vec, top_k=req.top_k)
    return {"status": "ok", "data": results}


@app.get("/health/live")
async def live() -> Dict[str, str]:  # pragma: no cover - trivial
    return {"status": "ok"}


@app.get("/health/ready")
async def ready() -> Dict[str, Any]:
    thread = getattr(app.state, "watch_thread", None)
    watcher_alive = bool(thread and thread.is_alive())
    scheduler = getattr(app.state, "embedding_scheduler", None)
    scheduler_running = bool(scheduler and getattr(scheduler, "running", False))
    ready = watcher_alive and scheduler_running
    return {
        "status": "ok" if ready else "error",
        "watcher_alive": watcher_alive,
        "scheduler_running": scheduler_running,
    }


@app.get("/status")
async def status() -> Dict[str, Any]:
    """Return basic service status information.

    Includes explicit fields for the background watcher thread and embedding
    scheduler state.
    """

    thread = getattr(app.state, "watch_thread", None)
    watcher_alive = bool(thread and thread.is_alive())
    scheduler = getattr(app.state, "embedding_scheduler", None)
    scheduler_running = bool(scheduler and getattr(scheduler, "running", False))
    return {
        "status": "ok",
        "watcher_alive": watcher_alive,
        "scheduler_running": scheduler_running,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover - simple server runner
    target = os.environ.get("VECTOR_SERVICE_SOCKET")
    if target:
        config = uvicorn.Config(
            "vector_service.vector_database_service:app", uds=target, log_level="info"
        )
    else:
        host = os.environ.get("VECTOR_SERVICE_HOST", "127.0.0.1")
        port = int(os.environ.get("VECTOR_SERVICE_PORT", "8000"))
        config = uvicorn.Config(
            "vector_service.vector_database_service:app",
            host=host,
            port=port,
            log_level="info",
        )
    uvicorn.Server(config).run()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
__all__ = ["app", "main"]
