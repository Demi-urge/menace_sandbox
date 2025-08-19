import logging
from types import SimpleNamespace

from vector_service import EmbeddingBackfill, EmbeddableDBMixin, PatchLogger


class DummyDB(EmbeddableDBMixin):
    instances = []

    def __init__(self, vector_backend=""):
        self.vector_backend = vector_backend
        self.processed = False
        DummyDB.instances.append(self)

    def backfill_embeddings(self, batch_size=0):
        self.processed = True
        self.batch_size = batch_size


def test_embedding_backfill_run(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    eb = EmbeddingBackfill(batch_size=10)
    monkeypatch.setattr(
        EmbeddingBackfill, "_load_known_dbs", lambda self, names=None: [DummyDB]
    )
    eb.run(session_id="s1")
    assert DummyDB.instances and DummyDB.instances[0].processed
    assert DummyDB.instances[0].batch_size == 10
    assert "Backfilling DummyDB" in caplog.text


class DummyMetricsDB:
    def __init__(self):
        self.calls = []

    def log_patch_outcome(self, patch_id, result, pairs, session_id=""):
        self.calls.append((patch_id, result, pairs, session_id))


class DummyVectorMetrics:
    def __init__(self):
        self.calls = []

    def update_outcome(self, session_id, pairs, contribution=0.0, patch_id="", win=False, regret=False):
        self.calls.append((session_id, pairs, contribution, patch_id, win, regret))


def test_patch_logger_tracks_metrics_db():
    mdb = DummyMetricsDB()
    logger = PatchLogger(metrics_db=mdb)
    logger.track_contributors(["db1:1", "2"], True, patch_id="p1", session_id="s1")
    assert mdb.calls == [("p1", True, [("db1", "1"), ("", "2")], "s1")]


def test_patch_logger_tracks_vector_metrics():
    vm = DummyVectorMetrics()
    logger = PatchLogger(metrics_db=None, vector_metrics=vm)
    logger.track_contributors(["a:3", "4"], False, patch_id="p2", session_id="s2")
    assert vm.calls == [("s2", [("a", "3"), ("", "4")], 0.0, "p2", False, True)]
