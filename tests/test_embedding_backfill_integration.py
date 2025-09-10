import importlib
import json
import sys
import time
import types
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

        def __init__(self, *args, **kwargs):
            self._metadata = getattr(DummyCodeDB, "_meta", {"1": {"embedding_version": 1}})

        def iter_records(self):
            return iter([(1, "a", None), (2, "b", None)])

    dummy_mod = types.ModuleType("dummy_mod")
    dummy_mod.DummyCodeDB = DummyCodeDB
    sys.modules["dummy_mod"] = dummy_mod
    monkeypatch.setattr(eb, "_load_registry", lambda path=None: {"code": ("dummy_mod", "DummyCodeDB")})

    # Create temporary DB and metadata files
    db_path = tmp_path / "code.db"
    db_path.write_text("x")
    meta_path = tmp_path / "code_embeddings.json"
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
