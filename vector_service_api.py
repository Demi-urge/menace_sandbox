from __future__ import annotations

"""Lightweight FastAPI app exposing vector service helpers.

This module wires the small utility classes from :mod:`semantic_service` into a
minimal HTTP API.  Each endpoint delegates to the corresponding service method
and returns a JSON response containing a status indicator and basic metrics such
as execution duration.  The module is intentionally tiny so it can be embedded
in tests or examples without pulling in the rest of the sandbox infrastructure.
"""

from typing import Any, List, Sequence
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from semantic_service import (
    ContextBuilder,
    EmbeddingBackfill,
    PatchLogger,
    Retriever,
    VectorServiceError,
    FallbackResult,
)

try:  # pragma: no cover - optional dependency
    from semantic_service import ErrorResult  # type: ignore
except Exception:  # pragma: no cover - compatibility fallback
    class ErrorResult(Exception):
        """Fallback ErrorResult when semantic_service lacks explicit class."""

        pass

app = FastAPI()

# Service instances are kept globally for simplicity.  They are lightweight and
# expose stateless interfaces which makes them safe to reuse across requests.
_retriever = Retriever()
_context_builder = ContextBuilder()
_patch_logger = PatchLogger()
_backfill = EmbeddingBackfill()


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None
    min_score: float | None = None
    include_confidence: bool = False


@app.post("/search")
def search(req: SearchRequest) -> Any:
    """Run semantic search via :class:`semantic_service.Retriever`."""
    start = time.time()
    try:
        result = _retriever.search(
            req.query,
            top_k=req.top_k,
            min_score=req.min_score,
            include_confidence=req.include_confidence,
        )
    except VectorServiceError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc))
    duration = time.time() - start
    if isinstance(result, ErrorResult):
        raise HTTPException(status_code=500, detail=str(result))

    if isinstance(result, FallbackResult):
        payload = list(result)
        size = len(payload)
        return {
            "status": "fallback",
            "data": payload,
            "reason": getattr(result, "reason", ""),
            "metrics": {"duration": duration, "result_size": size},
        }

    if req.include_confidence and isinstance(result, tuple):
        payload, confidence = result
        size = len(payload)
        return {
            "status": "ok",
            "data": payload,
            "confidence": confidence,
            "metrics": {"duration": duration, "result_size": size},
        }

    size = len(result) if isinstance(result, Sequence) else 0
    return {
        "status": "ok",
        "data": result,
        "metrics": {"duration": duration, "result_size": size},
    }


class ContextRequest(BaseModel):
    task_description: str
    extras: dict[str, Any] | None = None


@app.post("/build-context")
def build_context(req: ContextRequest) -> Any:
    """Construct a context string using :class:`semantic_service.ContextBuilder`."""
    start = time.time()
    try:
        result = _context_builder.build(
            req.task_description, **(req.extras or {})
        )
    except VectorServiceError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc))
    duration = time.time() - start
    if isinstance(result, ErrorResult):
        raise HTTPException(status_code=500, detail=str(result))
    if isinstance(result, FallbackResult):
        result = ""
    length = len(result) if isinstance(result, str) else 0
    return {
        "status": "ok",
        "data": result,
        "metrics": {"duration": duration, "result_size": length},
    }


class TrackRequest(BaseModel):
    vector_ids: List[str]
    result: bool
    patch_id: str | None = ""


@app.post("/track-contributors")
def track_contributors(req: TrackRequest) -> Any:
    """Record contributor outcomes via :class:`semantic_service.PatchLogger`."""
    start = time.time()
    try:
        _patch_logger.track_contributors(
            req.vector_ids, req.result, patch_id=req.patch_id or ""
        )
    except VectorServiceError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc))
    duration = time.time() - start
    return {"status": "ok", "metrics": {"duration": duration}}


class BackfillRequest(BaseModel):
    batch_size: int | None = None
    backend: str | None = None


@app.post("/backfill-embeddings")
def backfill_embeddings(req: BackfillRequest) -> Any:
    """Kick off embedding backfill using :class:`semantic_service.EmbeddingBackfill`."""
    start = time.time()
    try:
        _backfill.run(batch_size=req.batch_size, backend=req.backend)
    except VectorServiceError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc))
    duration = time.time() - start
    return {"status": "ok", "metrics": {"duration": duration}}


__all__ = ["app"]
