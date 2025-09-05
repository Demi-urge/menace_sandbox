import types
import sys

from vector_service.registry import register_vectorizer
from vector_service.vectorizer import SharedVectorService
from vector_service.embedding_backfill import EmbeddingBackfill, EmbeddableDBMixin


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
    assert svc.vectorise("toy", {"x": 3}) == [1.0, 2.0, 3.0]

    eb = EmbeddingBackfill(batch_size=5)
    calls = {}

    def fake_backfill(self, batch_size=0):
        calls["batch"] = batch_size

    monkeypatch.setattr(module.ToyDB, "backfill_embeddings", fake_backfill, raising=False)

    eb.run(db="toy")
    assert calls["batch"] == 5


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
