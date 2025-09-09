from __future__ import annotations

"""Minimal service exposing vector database operations.

Provides simple HTTP endpoints for adding records, querying existing
embeddings and checking service health.  A background thread calls
:func:`watch_databases` so newly added database records are continuously
embedded.

The service is intentionally lightweight; it is expected to run as a
separate daemon which other bots can interact with over HTTP or a UNIX
domain socket.
"""

from typing import Any, Dict
import threading
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .vectorizer import SharedVectorService
from .embedding_backfill import watch_databases

logger = logging.getLogger(__name__)

app = FastAPI()
svc = SharedVectorService()


def _watcher() -> None:
    try:
        watch_databases()
    except Exception:  # pragma: no cover - best effort
        logger.exception("watch_databases terminated unexpectedly")


@app.on_event("startup")
def _start_watcher() -> None:
    thread = threading.Thread(target=_watcher, daemon=True)
    thread.start()
    app.state.watch_thread = thread


class AddRequest(BaseModel):
    kind: str
    record_id: str
    record: Dict[str, Any]
    origin_db: str | None = None
    metadata: Dict[str, Any] | None = None


@app.post("/add")
def add(req: AddRequest) -> Dict[str, Any]:
    vec = svc.vectorise_and_store(
        req.kind,
        req.record_id,
        req.record,
        origin_db=req.origin_db,
        metadata=req.metadata,
    )
    return {"status": "ok", "vector": vec}


class QueryRequest(BaseModel):
    kind: str
    record: Dict[str, Any]
    top_k: int = 5


@app.post("/query")
def query(req: QueryRequest) -> Dict[str, Any]:
    vec = svc.vectorise(req.kind, req.record)
    if svc.vector_store is None:
        raise HTTPException(status_code=500, detail="Vector store not configured")
    results = svc.vector_store.query(vec, top_k=req.top_k)
    return {"status": "ok", "data": results}


@app.get("/status")
def status() -> Dict[str, str]:
    return {"status": "ok"}


__all__ = ["app"]
