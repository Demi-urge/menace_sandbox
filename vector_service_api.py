from __future__ import annotations

"""Lightweight FastAPI app exposing vector service helpers.

This module wires the small utility classes from :mod:`vector_service` into a
minimal HTTP API.  Each endpoint delegates to the corresponding service method
and returns a JSON response containing a status indicator and basic metrics such
as execution duration.  The module is intentionally tiny so it can be embedded
in tests or examples without pulling in the rest of the sandbox infrastructure.
"""

from typing import Any, List, Sequence
import os
import time
import logging
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dynamic_path_router import resolve_path

from vector_service import (
    CognitionLayer,
    EmbeddingBackfill,
    PatchLogger,
    Retriever,
    VectorServiceError,
    FallbackResult,
)
from vector_service.context_builder import ContextBuilder
from vector_service.patch_logger import RoiTag
try:  # pragma: no cover - optional dependency
    from roi_tracker import ROITracker
except Exception:  # pragma: no cover - simple fallback for tests
    class ROITracker:  # type: ignore
        def update_db_metrics(self, metrics: dict | None = None) -> None:
            pass

        def origin_db_deltas(self) -> dict:
            return {}
try:  # pragma: no cover - optional scheduler
    from analytics.ranker_scheduler import start_scheduler_from_env
except Exception:  # pragma: no cover - fallback for tests
    def start_scheduler_from_env(*args, **kwargs):  # type: ignore
        return None

try:  # pragma: no cover - optional dependency
    from vector_service import ErrorResult  # type: ignore
except Exception:  # pragma: no cover - compatibility fallback
    class ErrorResult(Exception):
        """Fallback ErrorResult when vector_service lacks explicit class."""

        pass

app = FastAPI()

# Service instances are kept globally for simplicity.  They are lightweight and
# expose stateless interfaces which makes them safe to reuse across requests.
_retriever = Retriever()
_roi_tracker = ROITracker()

# The context builder and dependent services are initialised lazily so tests and
# embedding environments can inject custom builders.
_builder: ContextBuilder | None = None
_cognition_layer: CognitionLayer | None = None
_patch_logger: PatchLogger | None = None
_backfill: EmbeddingBackfill | None = None
_ranker_scheduler = None


def create_app(builder: ContextBuilder | None = None) -> FastAPI:
    """Initialise service dependencies and return the FastAPI app.

    A :class:`ContextBuilder` instance can be supplied to explicitly control
    how context is assembled.  When omitted a default ``ContextBuilder`` is
    constructed.  The returned ``FastAPI`` app is ready for use by tests or
    service start-up scripts.
    """

    global _builder, _cognition_layer, _patch_logger, _backfill, _ranker_scheduler

    _builder = builder or ContextBuilder()
    try:
        _builder.refresh_db_weights()
    except Exception:  # pragma: no cover - best effort
        pass

    _cognition_layer = CognitionLayer(
        roi_tracker=_roi_tracker,
        context_builder=_builder,
    )
    _patch_logger = PatchLogger(roi_tracker=_roi_tracker)
    _backfill = EmbeddingBackfill()
    _ranker_scheduler = start_scheduler_from_env([_cognition_layer])
    return app


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Authentication and rate limiting
# ---------------------------------------------------------------------------
API_TOKEN = os.environ.get("VECTOR_SERVICE_API_TOKEN", "")
RATE_LIMIT = int(os.environ.get("VECTOR_SERVICE_RATE_LIMIT", "60"))
RATE_WINDOW = int(os.environ.get("VECTOR_SERVICE_RATE_WINDOW", "60"))

_request_log: dict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    """Validate API token and enforce a simple rate limit."""
    token = request.headers.get("X-API-Token") or request.query_params.get("token")
    if API_TOKEN and token != API_TOKEN:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    identifier = token or (request.client.host if request.client else "")
    now = time.time()
    window_start = now - RATE_WINDOW
    times = _request_log[identifier]
    while times and times[0] < window_start:
        times.popleft()
    if len(times) >= RATE_LIMIT:
        return JSONResponse(status_code=429, content={"detail": "Too Many Requests"})
    times.append(now)
    return await call_next(request)


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None
    min_score: float | None = None
    include_confidence: bool = False
    session_id: str = ""


@app.post("/search")
def search(req: SearchRequest) -> Any:
    """Run semantic search via :class:`vector_service.Retriever`."""
    start = time.time()
    try:
        result = _retriever.search(
            req.query,
            top_k=req.top_k,
            min_score=req.min_score,
            include_confidence=req.include_confidence,
            session_id=req.session_id,
        )
    except VectorServiceError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc))
    duration = time.time() - start
    if isinstance(result, ErrorResult):
        raise HTTPException(status_code=500, detail=str(result))

    if isinstance(result, FallbackResult):
        logger.warning("retriever fallback: %s", getattr(result, "reason", ""))
        payload = list(result)
        size = len(payload)
        response = {
            "status": "fallback",
            "data": payload,
            "reason": getattr(result, "reason", ""),
            "metrics": {"duration": duration, "result_size": size},
        }
        if req.include_confidence:
            response["confidence"] = getattr(result, "confidence", 0.0)
        return response

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


@app.post("/query")
def query(req: ContextRequest) -> Any:
    """Retrieve context via :class:`vector_service.CognitionLayer`."""
    start = time.time()
    if _cognition_layer is None:
        raise HTTPException(status_code=500, detail="Service not initialised")
    try:
        ctx, sid = _cognition_layer.query(
            req.task_description, **(req.extras or {})
        )
    except VectorServiceError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc))
    duration = time.time() - start
    length = len(ctx)
    return {
        "status": "ok",
        "data": ctx,
        "session_id": sid,
        "metrics": {"duration": duration, "result_size": length},
    }


class OutcomeRequest(BaseModel):
    session_id: str
    success: bool
    patch_id: str | None = None
    contribution: float | None = None


@app.post("/record-outcome")
def record_outcome(req: OutcomeRequest) -> Any:
    """Forward patch outcome to :class:`CognitionLayer`."""
    if _cognition_layer is None:
        raise HTTPException(status_code=500, detail="Service not initialised")
    try:
        _cognition_layer.record_patch_outcome(
            req.session_id,
            req.success,
            patch_id=req.patch_id or "",
            contribution=req.contribution,
        )
    except VectorServiceError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "ok"}


class TrackRequest(BaseModel):
    vector_ids: List[str]
    result: bool
    patch_id: str | None = ""
    session_id: str = ""
    diff: str | None = None
    summary: str | None = None
    roi_tag: RoiTag | None = None


@app.post("/track-contributors")
def track_contributors(req: TrackRequest) -> Any:
    """Record contributor outcomes via :class:`vector_service.PatchLogger`."""
    start = time.time()
    if _patch_logger is None:
        raise HTTPException(status_code=500, detail="Service not initialised")
    try:
        scores = _patch_logger.track_contributors(
            req.vector_ids,
            req.result,
            patch_id=req.patch_id or "",
            session_id=req.session_id,
            diff=req.diff,
            summary=req.summary,
            roi_tag=req.roi_tag,
        )
    except VectorServiceError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc))
    duration = time.time() - start
    payload = {"status": "ok", "metrics": {"duration": duration}}
    try:
        payload["risk_scores"] = dict(scores or {})
    except Exception:
        payload["risk_scores"] = {}
    return payload


class BackfillRequest(BaseModel):
    batch_size: int | None = None
    backend: str | None = None
    session_id: str = ""


@app.post("/backfill-embeddings")
def backfill_embeddings(req: BackfillRequest) -> Any:
    """Kick off embedding backfill using :class:`vector_service.EmbeddingBackfill`."""
    start = time.time()
    if _backfill is None:
        raise HTTPException(status_code=500, detail="Service not initialised")
    try:
        _backfill.run(
            session_id=req.session_id,
            batch_size=req.batch_size,
            backend=req.backend,
        )
    except VectorServiceError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc))
    duration = time.time() - start
    return {"status": "ok", "metrics": {"duration": duration}}


__all__ = ["app", "create_app"]


# ---------------------------------------------------------------------------
def evaluate_ranker(
    vector_db: str | os.PathLike[str] = resolve_path("vector_metrics.db"),
    patch_db: str | os.PathLike[str] = resolve_path("metrics.db"),
    strategy: str = "roi_weighted_cosine",
) -> dict[str, float]:
    """Evaluate ranking effectiveness on historical data.

    The helper loads training data from the vector and patch metrics databases
    and computes simple accuracy/AUC metrics using either the model-based
    ranker or a registered strategy from :mod:`retrieval_ranker`.
    """

    import numpy as np
    import retrieval_ranker as rr

    vdb = resolve_path(str(vector_db))
    pdb = resolve_path(str(patch_db))
    df = rr.load_training_data(vector_db=vdb, patch_db=pdb)
    if strategy == "model":
        _tm, metrics = rr.train(df)
        return metrics
    scores = rr.rank_candidates(df, strategy)
    y = df.get("label", 0).astype(int)
    acc = float(np.mean((scores >= 0.5) == y))
    auc = 0.0
    if rr.roc_auc_score is not None and y.nunique() > 1:
        try:
            auc = float(rr.roc_auc_score(y, scores))
        except Exception:
            auc = 0.0
    return {"accuracy": acc, "auc": auc}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for administrative utilities."""

    import argparse
    import json

    parser = argparse.ArgumentParser(description="Vector service utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate", help="Evaluate ranking effectiveness")
    ev.add_argument("--vector-db", default=resolve_path("vector_metrics.db"))
    ev.add_argument("--patch-db", default=resolve_path("metrics.db"))
    ev.add_argument(
        "--strategy",
        default="roi_weighted_cosine",
        help="Ranking strategy name or 'model' to train logistic model",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "evaluate":
        metrics = evaluate_ranker(
            vector_db=resolve_path(args.vector_db),
            patch_db=resolve_path(args.patch_db),
            strategy=args.strategy,
        )
        print(json.dumps(metrics))
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
