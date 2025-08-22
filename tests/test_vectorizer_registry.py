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
