import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
import pytest
import logging
import json

try:  # pragma: no cover - skip if vector_service missing
    from neurosales.api_gateway import create_app
    from neurosales.orchestrator import SandboxOrchestrator
    from neurosales.security import RateLimiter
except Exception:  # pragma: no cover - dependency missing
    pytest.skip("vector_service not installed", allow_module_level=True)


def test_create_app_requires_env(monkeypatch):
    monkeypatch.delenv("NEURO_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        create_app()


def test_chat_and_memory(monkeypatch):
    monkeypatch.setenv("NEURO_API_KEY", "secret")
    app = create_app()
    client = TestClient(app)
    headers = {"X-API-Key": "secret"}
    monkeypatch.setattr("neurosales.embedding.embed_text", lambda x: [0.0])

    r = client.post("/chat", json={"user_id": "u1", "line": "hello"}, headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert "reply" in data and "confidence" in data

    u = client.get("/user/u1", headers=headers)
    assert u.status_code == 404  # user not created yet via PUT

    client.put("/user/u1", json={"elo": 1200}, headers=headers)
    u = client.get("/user/u1", headers=headers)
    assert u.status_code == 200
    assert u.json()["elo"] == 1200

    s = client.post("/memory/search", json={"user_id": "u1", "query": "hello"}, headers=headers)
    assert s.status_code == 200
    assert "matches" in s.json()


def test_harvest_and_users(monkeypatch):
    monkeypatch.setenv("NEURO_API_KEY", "secret")
    app = create_app()
    client = TestClient(app)
    headers = {"X-API-Key": "secret"}
    monkeypatch.setattr("neurosales.embedding.embed_text", lambda x: [0.0])

    client.post("/chat", json={"user_id": "u2", "line": "hey"}, headers=headers)
    r = client.get("/orchestrator/users", headers=headers)
    assert r.status_code == 200
    assert "u2" in r.json().get("users", [])

    def fake_harvest(self, url, username=None, password=None, selector="article"):
        return ["a", "b"]

    monkeypatch.setattr(SandboxOrchestrator, "harvest_content", fake_harvest)
    h = client.post("/harvest", json={"url": "http://x"}, headers=headers)
    assert h.status_code == 200
    assert h.json()["content"] == ["a", "b"]


def test_requires_api_key(monkeypatch):
    monkeypatch.setenv("NEURO_API_KEY", "secret")
    from neurosales.metrics import Metrics
    import importlib
    metrics_module = importlib.import_module("neurosales.metrics")
    import neurosales.api_gateway as ag
    import neurosales.security as sec

    m = Metrics()
    monkeypatch.setattr(metrics_module, "metrics", m)
    monkeypatch.setattr(ag, "metrics", m)
    monkeypatch.setattr(sec, "metrics", m)

    app = ag.create_app()
    client = TestClient(app)
    r = client.post("/chat", json={"user_id": "x", "line": "hi"})
    assert r.status_code == 401
    assert m.auth_failures._value.get() == 1.0


def test_rate_limit(monkeypatch):
    monkeypatch.setenv("NEURO_API_KEY", "secret")
    monkeypatch.setenv("NEURO_RATE_LIMIT", "1")
    monkeypatch.setenv("NEURO_RATE_PERIOD", "60")

    from neurosales.metrics import Metrics
    import importlib
    metrics_module = importlib.import_module("neurosales.metrics")
    import neurosales.api_gateway as ag
    import neurosales.security as sec

    m = Metrics()
    monkeypatch.setattr(metrics_module, "metrics", m)
    monkeypatch.setattr(ag, "metrics", m)
    monkeypatch.setattr(sec, "metrics", m)

    app = ag.create_app()
    client = TestClient(app)
    headers = {"X-API-Key": "secret"}
    monkeypatch.setattr("neurosales.embedding.embed_text", lambda x: [0.0])

    client.post("/chat", json={"user_id": "u", "line": "hi"}, headers=headers)
    r = client.post("/chat", json={"user_id": "u", "line": "hi"}, headers=headers)
    assert r.status_code == 429
    assert m.rate_limited._value.get() == 1.0


def test_rate_limiter_from_env(monkeypatch):
    monkeypatch.setenv("NEURO_RATE_LIMIT", "7")
    monkeypatch.setenv("NEURO_RATE_PERIOD", "30")

    rl = RateLimiter()

    assert rl.rate == 7
    assert rl.per == 30.0


def test_redis_logging(monkeypatch, caplog):
    monkeypatch.setenv("NEURO_API_KEY", "secret")

    class BadRedis:
        class Redis:
            def __init__(self, *a, **kw):
                raise RuntimeError("boom")

    import neurosales.api_gateway as ag

    monkeypatch.setattr(ag, "redis", BadRedis)
    headers = {"X-API-Key": "secret"}
    monkeypatch.setattr("neurosales.embedding.embed_text", lambda x: [0.0])
    with caplog.at_level(logging.ERROR):
        app = ag.create_app()
        client = TestClient(app)
        client.post("/chat", json={"user_id": "u", "line": "hi"}, headers=headers)
        assert any("Redis" in rec.getMessage() for rec in caplog.records)


def test_request_metrics(monkeypatch):
    monkeypatch.setenv("NEURO_API_KEY", "secret")
    from neurosales.metrics import Metrics
    import importlib
    metrics_module = importlib.import_module("neurosales.metrics")
    import neurosales.api_gateway as ag

    m = Metrics()
    monkeypatch.setattr(metrics_module, "metrics", m)
    monkeypatch.setattr(ag, "metrics", m)

    app = ag.create_app()
    client = TestClient(app)
    headers = {"X-API-Key": "secret"}
    monkeypatch.setattr("neurosales.embedding.embed_text", lambda x: [0.0])

    client.post("/chat", json={"user_id": "m", "line": "hi"}, headers=headers)
    client.get("/bad", headers=headers)

    assert m.request_count.labels(path="/chat")._value.get() == 1.0
    assert m.failure_count.labels(path="/bad")._value.get() == 1.0


def test_json_logging(monkeypatch, caplog):
    monkeypatch.setenv("NEURO_API_KEY", "secret")
    import neurosales.api_gateway as ag

    monkeypatch.setattr("neurosales.embedding.embed_text", lambda x: [0.0])
    headers = {"X-API-Key": "secret"}
    with caplog.at_level(logging.INFO):
        app = ag.create_app()
        client = TestClient(app)
        client.post("/chat", json={"user_id": "j", "line": "hi"}, headers=headers)

    record = None
    for rec in caplog.records:
        try:
            data = json.loads(rec.getMessage())
        except Exception:
            continue
        if data.get("path") == "/chat":
            record = data
            break

    assert record is not None
    assert record["status"] == 200
    assert isinstance(record["duration"], float)
