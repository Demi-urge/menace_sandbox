import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
import menace.allocator_service as svc
import menace.session_vault as sv


def test_get_and_report(tmp_path):
    svc._vault = sv.SessionVault(path=tmp_path / "s.db")
    sid = svc._vault.add(
        "example.com",
        sv.SessionData(cookies={"a": "1"}, user_agent="ua", fingerprint="fp", last_seen=0),
    )

    client = TestClient(svc.app)
    resp = client.post("/session/get", json={"domain": "example.com"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == sid

    resp = client.post("/session/report", json={"session_id": sid, "status": "success"})
    assert resp.status_code == 200
    row = svc._vault.get_least_recent("example.com")
    assert row and row.success_count == 1
