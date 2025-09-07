import types

import pytest
import types

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

import os
import tempfile
import db_router
import types
import sys

# Provide a minimal stub for the heavy ``config`` module used by ContextBuilder.
config_stub = types.ModuleType("config")


class ContextBuilderConfig:
    def __getattr__(self, name):
        if name == "roi_tag_penalties":
            return {}
        if name == "license_denylist":
            return set()
        if name == "precise_token_count":
            return False
        if name == "similarity_metric":
            return "cosine"
        return 0


config_stub.ContextBuilderConfig = ContextBuilderConfig
sys.modules.setdefault("config", config_stub)

tmp_dir = tempfile.gettempdir()
local_db = os.path.join(tmp_dir, "test_local.db")
shared_db = os.path.join(tmp_dir, "test_shared.db")
db_router.init_db_router("test", local_db_path=local_db, shared_db_path=shared_db)

import vector_service_api as api
from vector_service import VectorServiceError

client = TestClient(api.app)


def test_search_success(monkeypatch):
    called = {}

    def fake_search(query, top_k=None, min_score=None, include_confidence=False, session_id=""):
        called['args'] = (query, top_k, min_score, include_confidence, session_id)
        return ["hit"]

    monkeypatch.setattr(api, "_retriever", types.SimpleNamespace(search=fake_search))
    resp = client.post("/search", json={"query": "hello", "session_id": "s"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["data"] == ["hit"]
    assert called['args'] == ("hello", None, None, False, "s")


def test_search_error(monkeypatch):
    def boom(*args, **kwargs):
        raise VectorServiceError("bad")

    monkeypatch.setattr(api, "_retriever", types.SimpleNamespace(search=boom))
    resp = client.post("/search", json={"query": "x", "session_id": "s"})
    assert resp.status_code == 500
    assert resp.json()["detail"] == "bad"


def test_query_success(monkeypatch):
    def fake_query(task, **extras):
        return f"ctx:{task}", "sid"

    monkeypatch.setattr(api, "_cognition_layer", types.SimpleNamespace(query=fake_query))
    resp = client.post("/query", json={"task_description": "t"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["data"] == "ctx:t"
    assert data["session_id"] == "sid"


def test_query_error(monkeypatch):
    def boom(*args, **kwargs):
        raise VectorServiceError("fail")

    monkeypatch.setattr(api, "_cognition_layer", types.SimpleNamespace(query=boom))
    resp = client.post("/query", json={"task_description": "t"})
    assert resp.status_code == 500
    assert resp.json()["detail"] == "fail"


def test_track_contributors_success(monkeypatch):
    called = {}

    def fake_track(ids, result, patch_id="", session_id="", **kwargs):
        called['args'] = (ids, result, patch_id, session_id, kwargs.get("roi_tag"))

    monkeypatch.setattr(api, "_patch_logger", types.SimpleNamespace(track_contributors=fake_track))
    resp = client.post(
        "/track-contributors",
        json={
            "vector_ids": ["a"],
            "result": True,
            "patch_id": "p",
            "session_id": "s",
            "roi_tag": "success",
        },
    )
    assert resp.status_code == 200
    assert called['args'] == (["a"], True, "p", "s", "success")


def test_track_contributors_error(monkeypatch):
    def boom(*args, **kwargs):
        raise VectorServiceError("nope")

    monkeypatch.setattr(api, "_patch_logger", types.SimpleNamespace(track_contributors=boom))
    resp = client.post(
        "/track-contributors",
        json={"vector_ids": ["a"], "result": False, "session_id": "s"},
    )
    assert resp.status_code == 500
    assert resp.json()["detail"] == "nope"


def test_backfill_embeddings_success(monkeypatch):
    called = {}

    def fake_run(session_id="", batch_size=None, backend=None):
        called['args'] = (session_id, batch_size, backend)

    monkeypatch.setattr(api, "_backfill", types.SimpleNamespace(run=fake_run))
    resp = client.post(
        "/backfill-embeddings",
        json={"batch_size": 1, "backend": "annoy", "session_id": "s"},
    )
    assert resp.status_code == 200
    assert called['args'] == ("s", 1, "annoy")


def test_backfill_embeddings_error(monkeypatch):
    def boom(*args, **kwargs):
        raise VectorServiceError("oops")

    monkeypatch.setattr(api, "_backfill", types.SimpleNamespace(run=boom))
    resp = client.post("/backfill-embeddings", json={"session_id": "s"})
    assert resp.status_code == 500
    assert resp.json()["detail"] == "oops"
