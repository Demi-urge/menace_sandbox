import importlib
import json
import os
import sys
import time
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest


def test_embeddings_backfill_triggers(monkeypatch, tmp_path):
    # Stub heavy dependencies before importing embedding_backfill
    trans_mod = types.ModuleType("transformers")
    trans_mod.AutoModel = object
    trans_mod.AutoTokenizer = object
    sys.modules["transformers"] = trans_mod

    # Ensure resolve_path looks in our temporary directory
    monkeypatch.setattr("dynamic_path_router.resolve_path", lambda p: Path(tmp_path / p))

    import vector_service.embedding_backfill as eb
    eb = importlib.reload(eb)

    # Use temporary timestamp file
    ts_file = tmp_path / "ts.json"
    monkeypatch.setattr(eb, "_TIMESTAMP_FILE", ts_file)

    # Define a dummy CodeDB with mismatched embeddings
    class DummyCodeDB:
        DB_FILE = "code.db"
        embedding_version = 1

        @classmethod
        def default_embedding_paths(cls):
            return tmp_path / "code_embeddings.index", tmp_path / "code_embeddings.json"

        def __init__(self, *args, **kwargs):
            self._metadata = getattr(DummyCodeDB, "_meta", {"1": {"embedding_version": 1}})

        def iter_records(self):
            return iter([(1, "a", None), (2, "b", None)])

        def needs_refresh(self, record_id, record):
            return False

    dummy_mod = types.ModuleType("dummy_mod")
    dummy_mod.DummyCodeDB = DummyCodeDB
    sys.modules["dummy_mod"] = dummy_mod
    monkeypatch.setattr(eb, "_load_registry", lambda path=None: {"code": ("dummy_mod", "DummyCodeDB")})

    # Create temporary DB and metadata files
    db_path = tmp_path / "code.db"
    db_path.write_text("x")
    meta_path = tmp_path / "code.json"
    meta_path.write_text("{}")

    # Seed timestamp so db appears up to date except for vector mismatch
    ts = time.time() + 100
    ts_file.write_text(json.dumps({"code": ts}))

    call_count = 0

    async def fake_schedule_backfill(*, dbs=None):
        nonlocal call_count
        call_count += 1
        DummyCodeDB._meta = {
            "1": {"embedding_version": 1},
            "2": {"embedding_version": 1},
        }

    monkeypatch.setattr(eb, "schedule_backfill", fake_schedule_backfill)

    # Should trigger a backfill once and then succeed without raising
    eb.ensure_embeddings_fresh(["code"], retries=1, delay=0)
    assert call_count == 1


def test_backfill_updates_last_vectorization_for_mtime_changes(monkeypatch, tmp_path):
    trans_mod = types.ModuleType("transformers")
    trans_mod.AutoModel = object
    trans_mod.AutoTokenizer = object
    sys.modules["transformers"] = trans_mod
    monkeypatch.setattr("dynamic_path_router.resolve_path", lambda p: Path(tmp_path / p))

    import vector_service.embedding_backfill as eb
    eb = importlib.reload(eb)

    ts_file = tmp_path / "ts.json"
    monkeypatch.setattr(eb, "_TIMESTAMP_FILE", ts_file)

    class DummyDB:
        DB_FILE = "dummy.db"
        embedding_version = 1

        def __init__(self, *args, **kwargs):
            raise RuntimeError("metadata-only refresh")

        def iter_records(self):
            return iter([])

    dummy_mod = types.ModuleType("dummy_mod_mtime")
    dummy_mod.DummyDB = DummyDB
    sys.modules["dummy_mod_mtime"] = dummy_mod
    monkeypatch.setattr(
        eb, "_load_registry", lambda path=None: {"dummy": ("dummy_mod_mtime", "DummyDB")}
    )

    db_path = tmp_path / "dummy.db"
    db_path.write_text("x")
    meta_path = tmp_path / "dummy_embeddings.json"
    meta_path.write_text("{}")

    now = time.time()
    meta_time = now - 300
    db_time = now - 100
    os.utime(meta_path, (meta_time, meta_time))
    os.utime(db_path, (db_time, db_time))
    ts_file.write_text(json.dumps({"dummy": 0.0}))

    async def fake_schedule_backfill(*, dbs=None):
        payload = {
            "id_map": [],
            "metadata": {},
            "vector_dim": 0,
            "last_vectorization": datetime.utcnow().isoformat(),
        }
        meta_path.write_text(json.dumps(payload))
        stale_time = (datetime.utcnow() - timedelta(hours=1)).timestamp()
        os.utime(meta_path, (stale_time, stale_time))

    monkeypatch.setattr(eb, "schedule_backfill", fake_schedule_backfill)

    eb.ensure_embeddings_fresh(["dummy"], retries=1, delay=0)
    updated = json.loads(ts_file.read_text())["dummy"]
    assert updated >= db_time
