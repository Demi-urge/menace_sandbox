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
domain socket specified by ``VECTOR_SERVICE_SOCKET``.
"""

from __future__ import annotations

from typing import Any, Dict
import asyncio
import os
import threading
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from .vectorizer import SharedVectorService
from .embedding_backfill import watch_databases, check_staleness
from .embedding_scheduler import start_scheduler_from_env

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Service setup
# ---------------------------------------------------------------------------

app = FastAPI()
_svc = SharedVectorService()


def _watcher() -> None:
    """Background thread embedding newly added database records.

    ``watch_databases`` is blocking and may raise exceptions.  We call it in a
    daemon thread so the API remains responsive while new records are embedded
    in the background.
    """

    try:
        check_staleness([])
        watch_databases()
    except Exception:  # pragma: no cover - best effort logging
        logger.exception("watch_databases terminated unexpectedly")


@app.on_event("startup")
async def _start_watcher() -> None:
    thread = threading.Thread(target=_watcher, daemon=True)
    thread.start()
    app.state.watch_thread = thread
    app.state.embedding_scheduler = start_scheduler_from_env()


@app.on_event("shutdown")
async def _stop_scheduler() -> None:
    scheduler = getattr(app.state, "embedding_scheduler", None)
    if scheduler is not None:
        scheduler.stop()


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
    if _svc.vector_store is None:
        raise HTTPException(status_code=500, detail="Vector store not configured")
    results = await asyncio.to_thread(_svc.vector_store.query, vec, top_k=req.top_k)
    return {"status": "ok", "data": results}


@app.get("/status")
async def status() -> Dict[str, str]:  # pragma: no cover - trivial
    return {"status": "ok"}


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
