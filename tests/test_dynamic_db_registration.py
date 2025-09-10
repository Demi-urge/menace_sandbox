import sys
import types

import pytest


def test_new_registry_entry_resolves_db_file(monkeypatch, tmp_path):
    dummy_mod = types.ModuleType("dummy_mod")

    class DummyDB:
        DB_FILE = "custom.sqlite"

        def __init__(self, *args, **kwargs):
            self._metadata = {}
            self.embedding_version = 1

        def iter_records(self):
            return iter([])

    dummy_mod.DummyDB = DummyDB
    sys.modules["dummy_mod"] = dummy_mod

    from vector_service import registry as reg

    monkeypatch.setattr(reg, "_VECTOR_REGISTRY", reg._VECTOR_REGISTRY.copy())
    reg.register_vectorizer(
        "dummy", "dummy_vec", "DummyVectorizer", db_module="dummy_mod", db_class="DummyDB"
    )

    import vector_service.embedding_backfill as eb

    db_path = tmp_path / "custom.sqlite"
    db_path.write_text("x")
    monkeypatch.setattr(eb, "_TIMESTAMP_FILE", tmp_path / "ts.json")
    eb._TIMESTAMP_FILE.write_text("{}")

    resolved = []

    def fake_resolve(path):
        resolved.append(path)
        return db_path if path == "custom.sqlite" else tmp_path / path

    monkeypatch.setattr(eb, "resolve_path", fake_resolve)

    called = {}

    async def fake_schedule_backfill(*, dbs=None, **_):
        called["dbs"] = dbs

    monkeypatch.setattr(eb, "schedule_backfill", fake_schedule_backfill)

    with pytest.raises(eb.StaleEmbeddingsError):
        eb.ensure_embeddings_fresh(["dummy"], retries=1, delay=0)

    assert "custom.sqlite" in resolved
    assert called["dbs"] == ["dummy"]

