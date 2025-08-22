import pytest

from vector_service.cognition_layer import CognitionLayer
from vector_service.context_builder import ContextBuilder
from vector_service.embedding_backfill import EmbeddingBackfill, EmbeddableDBMixin
from vector_service.patch_logger import PatchLogger
from patch_safety import PatchSafety
from vector_metrics_db import VectorMetricsDB


class DummyRetriever:
    def __init__(self):
        self.hits = [
            {
                "origin_db": "error",
                "record_id": "er1",
                "score": 0.5,
                "text": "error one",
                "metadata": {"message": "boom", "redacted": True},
            },
            {
                "origin_db": "bot",
                "record_id": "b1",
                "score": 0.5,
                "text": "bot one",
                "metadata": {"name": "bot-one", "redacted": True},
            },
            {
                "origin_db": "workflow",
                "record_id": "w1",
                "score": 0.5,
                "text": "workflow one",
                "metadata": {"title": "wf", "redacted": True},
            },
            {
                "origin_db": "enhancement",
                "record_id": "e1",
                "score": 0.5,
                "text": "enh one",
                "metadata": {"title": "enh", "redacted": True},
            },
        ]

    def search(self, query: str, top_k: int = 5, session_id: str = "", **_):
        return list(self.hits)


class DummyBus:
    def publish(self, *_, **__):
        pass


def _setup_layer(tmp_path):
    retriever = DummyRetriever()
    vec_db = VectorMetricsDB(tmp_path / "vec.db")
    builder = ContextBuilder(retriever=retriever)
    patch_logger = PatchLogger(
        vector_metrics=vec_db,
        roi_tracker=None,
        event_bus=DummyBus(),
        patch_safety=PatchSafety(failure_db_path=None),
    )
    layer = CognitionLayer(
        retriever=retriever,
        context_builder=builder,
        patch_logger=patch_logger,
        vector_metrics=vec_db,
        roi_tracker=None,
        event_bus=patch_logger.event_bus,
    )
    return layer, retriever, vec_db


def test_cognition_e2e(tmp_path, monkeypatch):
    # Seed sample databases and run EmbeddingBackfill
    class BotDB(EmbeddableDBMixin):
        records = {"b1": "bot one"}
        processed: list[str] = []

        def __init__(self, vector_backend="annoy"):
            self.vector_backend = vector_backend

        def backfill_embeddings(self, batch_size=0):
            self.__class__.processed.extend(self.records)

    class WorkflowDB(EmbeddableDBMixin):
        records = {"w1": "workflow one"}
        processed: list[str] = []

        def __init__(self, vector_backend="annoy"):
            self.vector_backend = vector_backend

        def backfill_embeddings(self, batch_size=0):
            self.__class__.processed.extend(self.records)

    class EnhancementDB(EmbeddableDBMixin):
        records = {"e1": "enh one"}
        processed: list[str] = []

        def __init__(self, vector_backend="annoy"):
            self.vector_backend = vector_backend

        def backfill_embeddings(self, batch_size=0):
            self.__class__.processed.extend(self.records)

    class ErrorDB(EmbeddableDBMixin):
        records = {"er1": "error one"}
        processed: list[str] = []

        def __init__(self, vector_backend="annoy"):
            self.vector_backend = vector_backend

        def backfill_embeddings(self, batch_size=0):
            self.__class__.processed.extend(self.records)

    monkeypatch.setattr(
        EmbeddingBackfill,
        "_load_known_dbs",
        lambda self, names=None: [BotDB, WorkflowDB, EnhancementDB, ErrorDB],
    )
    EmbeddingBackfill().run()
    assert BotDB.processed == ["b1"]
    assert WorkflowDB.processed == ["w1"]
    assert EnhancementDB.processed == ["e1"]
    assert ErrorDB.processed == ["er1"]

    layer, retriever, vec_db = _setup_layer(tmp_path)

    # Initial success so all weights start >0
    _ctx, sid = layer.query("hello", top_k=4)
    layer.record_patch_outcome(sid, True, contribution=2.0)
    weights = vec_db.get_db_weights()
    assert all(weights[o] > 0 for o in ("bot", "workflow", "enhancement", "error"))

    # Inject risk for bot and record failure
    retriever.hits[1]["metadata"]["risk_score"] = 2.0
    _ctx2, sid2 = layer.query("hello", top_k=4)
    layer.record_patch_outcome(sid2, False, contribution=0.0)
    weights2 = vec_db.get_db_weights()
    assert weights2["bot"] < weights2["workflow"]

    # Bot vectors should carry the lowest weight after failure
    assert weights2["bot"] == pytest.approx(0.0)
    assert weights2["bot"] < min(
        weights2["workflow"], weights2["enhancement"], weights2["error"]
    )

    # High ROI success for workflow only
    retriever.hits = [retriever.hits[2]]
    _ctx4, sid4 = layer.query("hello", top_k=1)
    layer.record_patch_outcome(sid4, True, contribution=3.0)
    retriever.hits = [
        {
            "origin_db": "error",
            "record_id": "er1",
            "score": 0.5,
            "text": "error one",
            "metadata": {"message": "boom", "redacted": True},
        },
        {
            "origin_db": "bot",
            "record_id": "b1",
            "score": 0.5,
            "text": "bot one",
            "metadata": {"name": "bot-one", "redacted": True, "risk_score": 2.0},
        },
        {
            "origin_db": "workflow",
            "record_id": "w1",
            "score": 0.5,
            "text": "workflow one",
            "metadata": {"title": "wf", "redacted": True},
        },
        {
            "origin_db": "enhancement",
            "record_id": "e1",
            "score": 0.5,
            "text": "enh one",
            "metadata": {"title": "enh", "redacted": True},
        },
    ]

    weights_final = vec_db.get_db_weights()
    assert weights_final["workflow"] > weights2["workflow"]
    assert vec_db.retriever_win_rate("workflow") > vec_db.retriever_win_rate("bot")
    assert vec_db.retriever_regret_rate("bot") > vec_db.retriever_regret_rate("workflow")

    assert weights_final["workflow"] > max(
        weights_final["error"], weights_final["enhancement"], weights_final.get("bot", 0.0)
    )
