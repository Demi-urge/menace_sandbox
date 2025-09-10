import json
import importlib
import logging
import time
from pathlib import Path
import sys
import types

import pytest


async def _noop_schedule_backfill(*, dbs=None):
    return None


def test_ensure_embeddings_fresh_logs_and_raises(monkeypatch, tmp_path, caplog):
    # Stub heavy dependencies before import
    trans_mod = types.ModuleType("transformers")
    trans_mod.AutoModel = object
    trans_mod.AutoTokenizer = object
    sys.modules["transformers"] = trans_mod
    monkeypatch.setattr("dynamic_path_router.resolve_path", lambda p: Path(tmp_path / p))

    import vector_service.embedding_backfill as eb
    eb = importlib.reload(eb)

    monkeypatch.setattr(eb, "_TIMESTAMP_FILE", Path(tmp_path / "ts.json"))
    monkeypatch.setattr(eb, "_load_registry", lambda path=None: {})
    monkeypatch.setattr(eb, "schedule_backfill", _noop_schedule_backfill)

    db_path = Path(tmp_path / "dummy.db")
    db_path.write_text("x")

    ts = time.time() + 100
    eb._TIMESTAMP_FILE.write_text(json.dumps({"dummy": ts}))

    with caplog.at_level(logging.ERROR):
        with pytest.raises(eb.StaleEmbeddingsError) as exc:
            eb.ensure_embeddings_fresh(["dummy"], retries=1, delay=0)

    assert exc.value.stale_dbs == {"dummy": "embedding metadata missing"}
    assert "dummy (embedding metadata missing)" in caplog.text


def test_ensure_embeddings_fresh_returns_diagnostics(monkeypatch, tmp_path, caplog):
    trans_mod = types.ModuleType("transformers")
    trans_mod.AutoModel = object
    trans_mod.AutoTokenizer = object
    sys.modules["transformers"] = trans_mod
    monkeypatch.setattr("dynamic_path_router.resolve_path", lambda p: Path(tmp_path / p))

    import vector_service.embedding_backfill as eb
    eb = importlib.reload(eb)

    monkeypatch.setattr(eb, "_TIMESTAMP_FILE", Path(tmp_path / "ts.json"))

    class DummyDB:
        DB_FILE = "dummy.db"
        embedding_version = 1

        def __init__(self, *args, **kwargs):
            self._metadata = getattr(DummyDB, "_meta", {"1": {"embedding_version": 1}})

        def iter_records(self):
            return iter([(1, "a", None), (2, "b", None)])

    dummy_mod = types.ModuleType("dummy_mod")
    dummy_mod.DummyDB = DummyDB
    sys.modules["dummy_mod"] = dummy_mod

    monkeypatch.setattr(eb, "_load_registry", lambda path=None: {"dummy": ("dummy_mod", "DummyDB")})

    db_path = Path(tmp_path / "dummy.db")
    db_path.write_text("x")
    meta_path = Path(tmp_path / "dummy_embeddings.json")
    meta_path.write_text("{}")

    ts = time.time() + 100
    eb._TIMESTAMP_FILE.write_text(json.dumps({"dummy": ts}))

    captured: list = []

    async def fake_schedule_backfill(*, dbs=None):
        DummyDB._meta = {"1": {"embedding_version": 1}, "2": {"embedding_version": 1}}

    monkeypatch.setattr(eb, "schedule_backfill", fake_schedule_backfill)

    with caplog.at_level(logging.INFO):
        result = eb.ensure_embeddings_fresh(
            ["dummy"], retries=1, delay=0, return_details=True, log_hook=captured.append
        )

    assert result["dummy"]["record_count"] == 2
    assert result["dummy"]["vector_count"] == 1
    assert captured and "dummy" in captured[0]
    assert "record/vector count mismatch 2/1" in caplog.text
    assert "db_mtime" in caplog.text and "meta_mtime" in caplog.text

