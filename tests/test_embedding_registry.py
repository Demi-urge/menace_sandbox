import json
import types
import sys

from vector_service.embedding_backfill import EmbeddingBackfill
import vector_service.embedding_backfill as eb
from vector_service import EmbeddableDBMixin


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
    monkeypatch.setattr(eb.pkgutil, "walk_packages", lambda *a, **k: [])

    monkeypatch.setattr(
        EmbeddableDBMixin,
        "__subclasses__",
        lambda: [TempDB],
    )

    processed = []

    def fake_process(self, db, *, batch_size, session_id=""):
        processed.append(type(db).__name__)
        return []

    monkeypatch.setattr(EmbeddingBackfill, "_process_db", fake_process)

    EmbeddingBackfill().run(dbs=["temp"])
    assert processed == ["TempDB"]
