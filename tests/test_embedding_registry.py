import json
import types
import sys

from vector_service.embedding_backfill import EmbeddingBackfill, EmbeddableDBMixin
import vector_service.embedding_backfill as eb


def test_backfill_loads_from_registry(tmp_path, monkeypatch):
    module = types.ModuleType("tempdb")

    class TempDB(EmbeddableDBMixin):
        def backfill_embeddings(self, batch_size=0):
            self.batch = batch_size

    module.TempDB = TempDB
    sys.modules["tempdb"] = module

    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"temp": {"module": "tempdb", "class": "TempDB"}}))
    monkeypatch.setattr(eb, "_REGISTRY_FILE", reg)

    processed = []

    def fake_process(self, db, *, batch_size, session_id=""):
        processed.append(type(db).__name__)
        return []

    monkeypatch.setattr(EmbeddingBackfill, "_process_db", fake_process)

    EmbeddingBackfill().run(dbs=["temp"])
    assert processed == ["TempDB"]


def test_verify_registry_warns_without_backfill(monkeypatch, tmp_path, caplog):
    module = types.ModuleType("nbdb")
    monkeypatch.setattr(eb, "EmbeddableDBMixin", type("Base", (), {}))

    class NBDB(eb.EmbeddableDBMixin):
        def iter_records(self):
            return []

        def vector(self, *a, **k):
            return None

    module.NBDB = NBDB
    sys.modules["nbdb"] = module

    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"nb": {"module": "nbdb", "class": "NBDB"}}))
    monkeypatch.setattr(eb, "_REGISTRY_FILE", reg)

    caplog.set_level("WARNING", logger="vector_service.embedding_backfill")
    EmbeddingBackfill()._verify_registry(["nb"])
    assert any(
        "missing backfill_embeddings" in r.getMessage() for r in caplog.records
    )
