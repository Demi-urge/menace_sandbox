import types
import sys
from pathlib import Path

from vector_service.registry import register_vectorizer
from vector_service.vectorizer import SharedVectorService
from vector_service.embedding_backfill import (
    EmbeddingBackfill,
    EmbeddableDBMixin,
    _resolve_embedding_paths,
)
from code_database import CodeDB
from error_bot import ErrorDB
from task_handoff_bot import WorkflowDB


def test_registry_enables_vectorization_and_backfill(monkeypatch):
    module = types.ModuleType("toy_plugin")

    class ToyVectorizer:
        def transform(self, record):
            return [1.0, 2.0, float(record.get("x", 0))]

    class ToyDB(EmbeddableDBMixin):
        def __init__(self, vector_backend="annoy"):
            self.backend = vector_backend

        @staticmethod
        def iter_records():
            return []

        @staticmethod
        def vector(record):
            return [0.0]

        def backfill_embeddings(self, batch_size=0):
            self.processed = batch_size

    module.ToyVectorizer = ToyVectorizer
    module.ToyDB = ToyDB
    sys.modules["toy_plugin"] = module

    register_vectorizer(
        "toy", "toy_plugin", "ToyVectorizer", db_module="toy_plugin", db_class="ToyDB"
    )

    svc = SharedVectorService()
    assert not svc._handlers
    assert "toy" in svc._known_kinds
    assert svc.ready()
    assert svc.vectorise("toy", {"x": 3}) == [1.0, 2.0, 3.0]

    eb = EmbeddingBackfill(batch_size=5)
    calls = {}

    def fake_backfill(self, batch_size=0):
        calls["batch"] = batch_size

    monkeypatch.setattr(module.ToyDB, "backfill_embeddings", fake_backfill, raising=False)

    eb.run(db="toy")
    assert calls["batch"] == 5


def test_default_embedding_metadata_paths_match_db_filenames():
    cases = [
        (
            "code",
            CodeDB,
            Path(CodeDB.DB_FILE).with_suffix(".index").with_suffix(".json").name,
        ),
        (
            "error",
            ErrorDB,
            Path(ErrorDB.DEFAULT_VECTOR_INDEX_PATH).with_suffix(".json").name,
        ),
        (
            "workflow",
            WorkflowDB,
            Path(WorkflowDB.DEFAULT_VECTOR_INDEX_PATH).with_suffix(".json").name,
        ),
    ]

    for name, cls, expected in cases:
        _, metadata_path = _resolve_embedding_paths(name, cls)
        assert metadata_path is not None
        assert metadata_path.name == expected


def test_auto_discovered_vectorizer_scheduled(tmp_path, monkeypatch):
    """Vectorizers placed under vector_service.* are backfilled automatically."""

    toy_plugin = tmp_path / "toy_plugin.py"  # path-ignore
    toy_plugin.write_text(
        """
from vector_service.embedding_backfill import EmbeddableDBMixin


class ToyDB(EmbeddableDBMixin):
    @staticmethod
    def iter_records():
        return []

    @staticmethod
    def vector(record):
        return []

    def backfill_embeddings(self, batch_size=0):
        pass
"""
    )

    vs_extra = tmp_path / "extra_vs"
    vs_extra.mkdir()
    (vs_extra / "__init__.py").write_text("")  # path-ignore
    (vs_extra / "toy_vectorizer.py").write_text(  # path-ignore
        """
DB_MODULE = "toy_plugin"
DB_CLASS = "ToyDB"


class ToyVectorizer:
    def transform(self, record):
        return [0.0]
"""
    )

    sys.path.insert(0, str(tmp_path))
    import vector_service

    vector_service.__path__.append(str(vs_extra))

    import importlib
    import vector_service.registry as registry
    importlib.reload(registry)
    import vector_service.embedding_backfill as eb
    importlib.reload(eb)

    processed: list[str] = []

    def fake_run(self, *, db=None, **_):
        processed.append(db)

    monkeypatch.setattr(eb.EmbeddingBackfill, "run", fake_run)

    import asyncio

    asyncio.run(eb.schedule_backfill())
    assert "toy" in processed


def test_bootstrap_fast_defers_vector_store(monkeypatch):
    module = types.ModuleType("toy_plugin_bootstrap")

    class ToyVectorizer:
        def transform(self, record):
            return [float(record.get("x", 0))]

    module.ToyVectorizer = ToyVectorizer
    sys.modules["toy_plugin_bootstrap"] = module

    register_vectorizer("toy-bootstrap", "toy_plugin_bootstrap", "ToyVectorizer")

    svc = SharedVectorService(bootstrap_fast=True)
    vec = svc.vectorise_and_store("toy-bootstrap", "1", {"x": 2})

    assert vec == [2.0]
    assert svc.vector_store is None
