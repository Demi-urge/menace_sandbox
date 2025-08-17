import pytest
from semantic_service.patch_logger import PatchLogger
from semantic_service.embedding_backfill import EmbeddingBackfill, EmbeddableDBMixin
import semantic_service.decorators as dec


class DummyGauge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []
        self.label = ""

    def labels(self, label: str):
        self.label = label
        return self

    def inc(self, amount: float = 1.0) -> None:
        self.calls.append(("inc", self.label, amount))

    def set(self, value: float) -> None:
        self.calls.append(("set", self.label, value))


class DummyMetricsDB:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.called = False

    def log_patch_outcome(self, patch_id, result, pairs, session_id=""):
        self.called = True
        if self.fail:
            raise RuntimeError("db failure")


def patch_metrics(monkeypatch):
    call = DummyGauge()
    lat = DummyGauge()
    size = DummyGauge()
    monkeypatch.setattr(dec, "_CALL_COUNT", call)
    monkeypatch.setattr(dec, "_LATENCY_GAUGE", lat)
    monkeypatch.setattr(dec, "_RESULT_SIZE_GAUGE", size)
    return call, lat, size


# ---------------------------------------------------------------------------
# PatchLogger.track_contributors
# ---------------------------------------------------------------------------


def test_track_contributors_success_emits_metrics(monkeypatch):
    call, lat, size = patch_metrics(monkeypatch)
    db = DummyMetricsDB()
    pl = PatchLogger(metrics_db=db)
    pl.track_contributors(["a:b", "c"], True, patch_id="1", session_id="s")
    assert db.called
    assert ("inc", "PatchLogger.track_contributors", 1.0) in call.calls
    assert any(c[0] == "set" and c[1] == "PatchLogger.track_contributors" for c in lat.calls)
    assert any(c[0] == "set" and c[1] == "PatchLogger.track_contributors" for c in size.calls)


def test_track_contributors_db_failure_still_counts(monkeypatch):
    call, lat, size = patch_metrics(monkeypatch)
    db = DummyMetricsDB(fail=True)
    pl = PatchLogger(metrics_db=db)
    pl.track_contributors(["x"], False, patch_id="2")
    assert db.called
    assert ("inc", "PatchLogger.track_contributors", 1.0) in call.calls


def test_track_contributors_malformed_vector_ids_raise_and_count(monkeypatch):
    call, lat, size = patch_metrics(monkeypatch)
    pl = PatchLogger()
    with pytest.raises(TypeError):
        pl.track_contributors(["ok", None], True)
    assert ("inc", "PatchLogger.track_contributors", 1.0) in call.calls


# ---------------------------------------------------------------------------
# EmbeddingBackfill.run retry/skip behavior
# ---------------------------------------------------------------------------


_processed: list[type] = []


class RetryDB(EmbeddableDBMixin):
    def __init__(self, vector_backend: str | None = None) -> None:
        if vector_backend is not None:
            raise RuntimeError("no backend")

    def backfill_embeddings(self, batch_size: int = 0) -> None:
        _processed.append(self.__class__)


class FailDB(EmbeddableDBMixin):
    def __init__(self, vector_backend: str | None = None) -> None:
        raise RuntimeError("construction failed")

    def backfill_embeddings(self, batch_size: int = 0) -> None:
        _processed.append(self.__class__)


def test_embedding_backfill_run_retries_and_skips(monkeypatch):
    call, lat, size = patch_metrics(monkeypatch)
    _processed.clear()
    monkeypatch.setattr(
        EmbeddingBackfill, "_load_known_dbs", lambda self: [RetryDB, FailDB]
    )
    eb = EmbeddingBackfill(batch_size=5)
    eb.run(session_id="sid")
    assert _processed == [RetryDB]
    assert ("inc", "EmbeddingBackfill.run", 1.0) in call.calls
    assert ("inc", "EmbeddingBackfill._process_db", 1.0) in call.calls
