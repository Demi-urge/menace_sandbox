import sys
import types

import pytest


@pytest.mark.asyncio
async def test_start_watcher_recovers_from_stale_embeddings(monkeypatch):
    # Stub heavy dependencies
    trans_mod = types.ModuleType("transformers")
    trans_mod.AutoModel = object
    trans_mod.AutoTokenizer = object
    sys.modules["transformers"] = trans_mod

    torch_mod = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    torch_mod.no_grad = _NoGrad
    sys.modules["torch"] = torch_mod

    uvicorn_mod = types.ModuleType("uvicorn")
    uvicorn_mod.Config = object
    uvicorn_mod.Server = object
    sys.modules["uvicorn"] = uvicorn_mod

    from vector_service import embedding_backfill as eb
    from vector_service import vector_database_service as vds

    monkeypatch.setattr(vds, "_spawn_watcher", lambda: None)

    async def _noop_monitor():
        return None

    monkeypatch.setattr(vds, "_monitor_watcher", _noop_monitor)
    monkeypatch.setattr(vds, "start_scheduler_from_env", lambda: object())

    # Mock internals used by ensure_embeddings_fresh
    monkeypatch.setattr(eb, "_load_timestamps", lambda: {})
    monkeypatch.setattr(eb, "_DB_FILE_MAP", {"code": "code.db"})

    calls = {"count": 0}

    def fake_ensure(dbs):
        # touch mocked helpers to satisfy coverage
        eb._load_timestamps()
        _ = eb._DB_FILE_MAP
        calls["count"] += 1
        if calls["count"] == 1:
            raise eb.StaleEmbeddingsError({"code": "stale"})

    monkeypatch.setattr(vds, "ensure_embeddings_fresh", fake_ensure)
    monkeypatch.setattr(vds, "StaleEmbeddingsError", eb.StaleEmbeddingsError)

    called: dict[str, list[str]] = {}

    async def fake_schedule_backfill(*, dbs=None, **kwargs):
        called["dbs"] = list(dbs)

    monkeypatch.setattr(vds, "schedule_backfill", fake_schedule_backfill)

    await vds._start_watcher()

    assert called["dbs"] == ["code"]
    assert calls["count"] == 2
