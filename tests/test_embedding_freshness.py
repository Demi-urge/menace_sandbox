import json
import importlib
import logging
import os
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


def test_needs_backfill_ignores_meta_mtime_when_vectorization_newer(
    monkeypatch, tmp_path
):
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
            self._metadata = {"1": {"embedding_version": 1}, "2": {"embedding_version": 1}}

        def iter_records(self):
            return iter([(1, "a", None), (2, "b", None)])

    dummy_mod = types.ModuleType("dummy_mod_meta")
    dummy_mod.DummyDB = DummyDB
    sys.modules["dummy_mod_meta"] = dummy_mod

    monkeypatch.setattr(
        eb, "_load_registry", lambda path=None: {"dummy": ("dummy_mod_meta", "DummyDB")}
    )

    db_path = Path(tmp_path / "dummy.db")
    db_path.write_text("x")
    meta_path = Path(tmp_path / "dummy_embeddings.json")
    meta_path.write_text("{}")

    now = time.time()
    meta_time = now - 200
    db_time = now - 100
    last_vec = now + 100
    eb._TIMESTAMP_FILE.write_text(json.dumps({"dummy": last_vec}))

    meta_path.touch()
    db_path.touch()
    os.utime(meta_path, (meta_time, meta_time))
    os.utime(db_path, (db_time, db_time))

    async def fail_schedule_backfill(*, dbs=None):
        raise AssertionError("schedule_backfill should not be called")

    monkeypatch.setattr(eb, "schedule_backfill", fail_schedule_backfill)

    result = eb.ensure_embeddings_fresh(["dummy"], retries=1, delay=0, return_details=True)

    assert result == {}


def test_ensure_embeddings_fresh_updates_timestamp_when_db_mtime_changes(
    monkeypatch, tmp_path
):
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
            self._metadata = {"1": {"embedding_version": 1}, "2": {"embedding_version": 1}}

        def iter_records(self):
            return iter([(1, "a", None), (2, "b", None)])

        def needs_refresh(self, record_id, record):
            return False

    dummy_mod = types.ModuleType("dummy_mod_mtime")
    dummy_mod.DummyDB = DummyDB
    sys.modules["dummy_mod_mtime"] = dummy_mod

    monkeypatch.setattr(
        eb, "_load_registry", lambda path=None: {"dummy": ("dummy_mod_mtime", "DummyDB")}
    )

    db_path = Path(tmp_path / "dummy.db")
    db_path.write_text("x")
    meta_path = Path(tmp_path / "dummy_embeddings.json")
    meta_path.write_text("{}")

    now = time.time()
    meta_time = now - 200
    db_time = now - 50
    last_vec = now - 150
    eb._TIMESTAMP_FILE.write_text(json.dumps({"dummy": last_vec}))

    os.utime(meta_path, (meta_time, meta_time))
    os.utime(db_path, (db_time, db_time))

    async def fail_schedule_backfill(*, dbs=None):
        raise AssertionError("schedule_backfill should not be called")

    monkeypatch.setattr(eb, "schedule_backfill", fail_schedule_backfill)

    result = eb.ensure_embeddings_fresh(["dummy"], retries=1, delay=0, return_details=True)

    assert result == {}
    updated = json.loads(eb._TIMESTAMP_FILE.read_text())["dummy"]
    assert updated >= db_time


def test_ensure_embeddings_fresh_uses_last_vectorization_over_stale_metadata(
    monkeypatch, tmp_path
):
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
            self._metadata = {"1": {"embedding_version": 1}, "2": {"embedding_version": 1}}

        def iter_records(self):
            return iter([(1, "a", None), (2, "b", None)])

    dummy_mod = types.ModuleType("dummy_mod_vec")
    dummy_mod.DummyDB = DummyDB
    sys.modules["dummy_mod_vec"] = dummy_mod

    monkeypatch.setattr(
        eb, "_load_registry", lambda path=None: {"dummy": ("dummy_mod_vec", "DummyDB")}
    )

    db_path = Path(tmp_path / "dummy.db")
    db_path.write_text("x")
    meta_path = Path(tmp_path / "dummy_embeddings.json")
    meta_path.write_text("{}")

    now = time.time()
    meta_time = now - 200
    db_time = now - 100
    last_vec = now + 100
    eb._TIMESTAMP_FILE.write_text(json.dumps({"dummy": last_vec}))

    os.utime(meta_path, (meta_time, meta_time))
    os.utime(db_path, (db_time, db_time))

    async def fail_schedule_backfill(*, dbs=None):
        raise AssertionError("schedule_backfill should not be called")

    monkeypatch.setattr(eb, "schedule_backfill", fail_schedule_backfill)

    result = eb.ensure_embeddings_fresh(["dummy"], retries=1, delay=0, return_details=True)

    assert result == {}


def test_metadata_path_fallback_to_db_suffix(monkeypatch, tmp_path, caplog):
    trans_mod = types.ModuleType("transformers")
    trans_mod.AutoModel = object
    trans_mod.AutoTokenizer = object
    sys.modules["transformers"] = trans_mod

    def resolve_path_stub(p):
        if p == "dummy_embeddings.json":
            raise FileNotFoundError(p)
        return Path(tmp_path / p)

    monkeypatch.setattr("dynamic_path_router.resolve_path", resolve_path_stub)

    import vector_service.embedding_backfill as eb
    eb = importlib.reload(eb)

    monkeypatch.setattr(eb, "_TIMESTAMP_FILE", Path(tmp_path / "ts.json"))

    class DummyDB:
        DB_FILE = "dummy.db"

        def __init__(self, required):
            self._metadata = {"1": {"embedding_version": 1}}

        def iter_records(self):
            return iter([(1, "a", None)])

    dummy_mod = types.ModuleType("dummy_mod_suffix")
    dummy_mod.DummyDB = DummyDB
    sys.modules["dummy_mod_suffix"] = dummy_mod

    monkeypatch.setattr(
        eb, "_load_registry", lambda path=None: {"dummy": ("dummy_mod_suffix", "DummyDB")}
    )

    db_path = Path(tmp_path / "dummy.db")
    db_path.write_text("x")
    meta_path = db_path.with_suffix(".json")

    future = time.time() + 100
    eb._TIMESTAMP_FILE.write_text(json.dumps({"dummy": future}))

    monkeypatch.setattr(eb, "schedule_backfill", _noop_schedule_backfill)

    with caplog.at_level(logging.INFO):
        with pytest.raises(eb.StaleEmbeddingsError) as exc:
            eb.ensure_embeddings_fresh(["dummy"], retries=1, delay=0, return_details=True)

    assert exc.value.stale_dbs == {"dummy": "embedding metadata missing"}
    assert str(meta_path) in caplog.text

    meta_path.write_text("{}")
    result = eb.ensure_embeddings_fresh(["dummy"], retries=1, delay=0, return_details=True)
    assert result == {}


def test_backfill_accepts_non_default_metadata_location(monkeypatch, tmp_path):
    trans_mod = types.ModuleType("transformers")
    trans_mod.AutoModel = object
    trans_mod.AutoTokenizer = object
    sys.modules["transformers"] = trans_mod
    monkeypatch.setattr("dynamic_path_router.resolve_path", lambda p: Path(tmp_path / p))

    import vector_service.embedding_backfill as eb
    eb = importlib.reload(eb)

    monkeypatch.setattr(eb, "_TIMESTAMP_FILE", Path(tmp_path / "ts.json"))
    monkeypatch.setattr(eb, "_load_registry", lambda path=None: {})

    db_path = Path(tmp_path / "dummy.db")
    db_path.write_text("x")
    alt_meta_path = db_path.with_suffix(".json")

    eb._TIMESTAMP_FILE.write_text(json.dumps({"dummy": 0}))

    async def fake_schedule_backfill(*, dbs=None):
        alt_meta_path.write_text("{}")

    monkeypatch.setattr(eb, "schedule_backfill", fake_schedule_backfill)

    eb.ensure_embeddings_fresh(["dummy"], retries=1, delay=0)
    assert alt_meta_path.exists()


def test_ensure_embeddings_fresh_does_not_update_timestamp_when_pending(
    monkeypatch, tmp_path, caplog
):
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
    initial_ts = 123.0
    eb._TIMESTAMP_FILE.write_text(json.dumps({"dummy": initial_ts}))

    with caplog.at_level(logging.WARNING):
        with pytest.raises(eb.StaleEmbeddingsError):
            eb.ensure_embeddings_fresh(["dummy"], retries=1, delay=0)

    updated = json.loads(eb._TIMESTAMP_FILE.read_text())
    assert updated["dummy"] == initial_ts
    assert "embedding still pending for dummy" in caplog.text
    assert "embedding metadata missing" in caplog.text
    assert str(db_path.with_suffix(".json")) in caplog.text


def test_ensure_embeddings_fresh_uses_index_metadata_path(monkeypatch, tmp_path):
    trans_mod = types.ModuleType("transformers")
    trans_mod.AutoModel = object
    trans_mod.AutoTokenizer = object
    sys.modules["transformers"] = trans_mod
    monkeypatch.setattr("dynamic_path_router.resolve_path", lambda p: Path(tmp_path / p))

    import vector_service.embedding_backfill as eb
    eb = importlib.reload(eb)

    monkeypatch.setattr(eb, "_TIMESTAMP_FILE", Path(tmp_path / "ts.json"))

    class CodeDB:
        DB_FILE = "code.db"
        embedding_version = 1

        def __init__(self, *args, **kwargs):
            self._metadata = {"1": {"embedding_version": 1}}

        def iter_records(self):
            return iter([(1, "a", None)])

    dummy_mod = types.ModuleType("dummy_code_mod")
    dummy_mod.CodeDB = CodeDB
    sys.modules["dummy_code_mod"] = dummy_mod

    monkeypatch.setattr(eb, "_load_registry", lambda path=None: {"code": ("dummy_code_mod", "CodeDB")})

    db_path = Path(tmp_path / "code.db")
    db_path.write_text("x")
    meta_path = Path(tmp_path / "code.json")
    meta_path.write_text("{}")

    now = time.time()
    meta_time = now - 100
    db_time = now
    os.utime(meta_path, (meta_time, meta_time))
    os.utime(db_path, (db_time, db_time))

    eb._TIMESTAMP_FILE.write_text(json.dumps({"code": 0.0}))

    async def fake_schedule_backfill(*, dbs=None):
        refreshed = time.time() + 200
        meta_path.write_text("{}")
        os.utime(meta_path, (refreshed, refreshed))

    monkeypatch.setattr(eb, "schedule_backfill", fake_schedule_backfill)

    captured: list = []
    result = eb.ensure_embeddings_fresh(
        ["code"],
        retries=1,
        delay=0,
        return_details=True,
        log_hook=captured.append,
    )

    assert result["code"]["meta_path"] == str(meta_path)
    assert captured[0]["code"]["meta_path"] == str(meta_path)
