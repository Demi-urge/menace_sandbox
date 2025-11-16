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

try:  # pragma: no cover - compatibility when packaged
    from .vector_metrics_db import default_vector_metrics_path
except Exception:  # pragma: no cover - direct execution fallback
    from vector_metrics_db import default_vector_metrics_path  # type: ignore

from vector_service import (
    CognitionLayer,
    EmbeddingBackfill,
    PatchLogger,
    Retriever,
    VectorServiceError,
    FallbackResult,
)
from vector_service.context_builder import ContextBuilder
from context_builder_util import ensure_fresh_weights
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


@app.get("/health")
def health() -> dict[str, str]:
    """Simple liveness probe."""
    return {"status": "ok"}

# Service instances are stored on ``app.state`` instead of module level globals.
# This makes dependency injection explicit and avoids side effects when the
# module is imported without calling :func:`create_app`.


def create_app(builder: ContextBuilder) -> FastAPI:
    """Initialise service dependencies and return the FastAPI app.

    A :class:`ContextBuilder` instance must be supplied to explicitly control
    how context is assembled.  The returned ``FastAPI`` app is ready for use by
    tests or service start-up scripts once initialisation completes.
    """

    try:
        ensure_fresh_weights(builder)
    except Exception as exc:  # pragma: no cover - validation
        logging.getLogger(__name__).error("context builder refresh failed: %s", exc)
        raise RuntimeError("context builder refresh failed") from exc

    # Store the builder and dependent services on ``app.state`` so endpoint
    # handlers can access them via ``request.app.state`` without relying on
    # module level globals.
    app.state.builder = builder
    roi_tracker = ROITracker()
    app.state.roi_tracker = roi_tracker
    app.state.retriever = Retriever(context_builder=builder)
    app.state.cognition_layer = CognitionLayer(
        roi_tracker=roi_tracker,
        context_builder=builder,
    )
    app.state.patch_logger = PatchLogger(roi_tracker=roi_tracker)
    app.state.backfill = EmbeddingBackfill()
    app.state.ranker_scheduler = start_scheduler_from_env([app.state.cognition_layer])
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
def search(req: SearchRequest, request: Request) -> Any:
    """Run semantic search via :class:`vector_service.Retriever`."""
    retriever: Retriever | None = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(status_code=500, detail="Service not initialised")

    start = time.time()
    try:
        result = retriever.search(
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
def query(req: ContextRequest, request: Request) -> Any:
    """Retrieve context via :class:`vector_service.CognitionLayer`."""
    start = time.time()
    layer: CognitionLayer | None = getattr(request.app.state, "cognition_layer", None)
    if layer is None:
        raise HTTPException(status_code=500, detail="Service not initialised")
    try:
        ctx, sid = layer.query(
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
def record_outcome(req: OutcomeRequest, request: Request) -> Any:
    """Forward patch outcome to :class:`CognitionLayer`."""
    layer: CognitionLayer | None = getattr(request.app.state, "cognition_layer", None)
    if layer is None:
        raise HTTPException(status_code=500, detail="Service not initialised")
    try:
        layer.record_patch_outcome(
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
def track_contributors(req: TrackRequest, request: Request) -> Any:
    """Record contributor outcomes via :class:`vector_service.PatchLogger`."""
    start = time.time()
    logger_obj: PatchLogger | None = getattr(request.app.state, "patch_logger", None)
    if logger_obj is None:
        raise HTTPException(status_code=500, detail="Service not initialised")
    try:
        scores = logger_obj.track_contributors(
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
def backfill_embeddings(req: BackfillRequest, request: Request) -> Any:
    """Kick off embedding backfill using :class:`vector_service.EmbeddingBackfill`."""
    start = time.time()
    backfill_obj: EmbeddingBackfill | None = getattr(request.app.state, "backfill", None)
    if backfill_obj is None:
        raise HTTPException(status_code=500, detail="Service not initialised")
    try:
        backfill_obj.run(
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
    vector_db: str | os.PathLike[str] = default_vector_metrics_path(),
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
    ev.add_argument("--vector-db", default=str(default_vector_metrics_path()))
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
