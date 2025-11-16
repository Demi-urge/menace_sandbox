import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

import menace.evaluation_service as svc
from menace.evaluation_history_db import EvaluationHistoryDB, EvaluationRecord
import db_router


def test_service_endpoints(tmp_path):
    router = db_router.DBRouter(
        "svc", str(tmp_path / "hist.db"), str(tmp_path / "hist.db")
    )
    svc._db = EvaluationHistoryDB(router=router)
    svc._db.add(EvaluationRecord(engine="a", cv_score=0.1))
    svc._db.add(EvaluationRecord(engine="b", cv_score=0.5))

    client = TestClient(svc.app)
    resp = client.get("/scores/a")
    assert resp.status_code == 200
    assert resp.json()[0]["cv_score"] == 0.1

    resp = client.get("/weights")
    assert resp.status_code == 200
    weights = resp.json()
    assert weights["b"] == 1.0

    resp = client.post("/weights", json={"engine": "a", "weight": 0.2})
    assert resp.status_code == 200
    resp = client.get("/weights")
    assert resp.json()["a"] == 0.2
