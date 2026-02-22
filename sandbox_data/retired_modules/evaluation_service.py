from __future__ import annotations

"""FastAPI service exposing evaluation history and weights."""

from fastapi import FastAPI
from pydantic import BaseModel

from .evaluation_history_db import EvaluationHistoryDB, EvaluationRecord

app = FastAPI(title="Evaluation Service")
_db = EvaluationHistoryDB()


class WeightUpdate(BaseModel):
    engine: str
    weight: float


@app.get("/scores/{engine}")
def get_scores(engine: str, limit: int = 50):
    rows = _db.history(engine, limit)
    result = []
    for score, ts, passed, error in rows:
        entry = {"cv_score": score, "ts": ts, "passed": bool(passed)}
        if error:
            entry["error"] = error
        result.append(entry)
    return result


@app.get("/weights")
def get_weights():
    return _db.deployment_weights()


@app.post("/weights")
def set_weight(info: WeightUpdate):
    _db.set_weight(info.engine, info.weight)
    return {"ok": True}


@app.get("/compare")
def compare(limit: int = 50):
    data: dict[str, list] = {}
    for eng in _db.engines():
        data[eng] = [
            {"cv_score": s, "ts": t} for s, t in _db.history(eng, limit)
        ]
    return data


__all__ = ["app"]
