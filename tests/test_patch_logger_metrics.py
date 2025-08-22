import logging
import pytest
from typing import Any
from vector_service import EmbeddingBackfill, EmbeddableDBMixin, PatchLogger
import vector_service.decorators as dec


class DummyGauge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []
        self.label = ""
        self._labels: tuple[str, ...] = ()

    def labels(self, *labels: str):
        self._labels = labels
        self.label = labels[0] if labels else ""
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


class DummyVectorMetricsDB:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fb_calls: list[dict[str, Any]] = []

    def update_outcome(
        self,
        session_id,
        pairs,
        *,
        contribution,
        patch_id,
        win,
        regret,
    ):
        self.calls.append(
            {
                "session_id": session_id,
                "pairs": pairs,
                "contribution": contribution,
                "patch_id": patch_id,
                "win": win,
                "regret": regret,
            }
        )

    def log_retrieval_feedback(self, db, *, win=False, regret=False, roi=0.0):
        self.fb_calls.append({"db": db, "win": win, "regret": regret, "roi": roi})


class DummyROITracker:
    def __init__(self) -> None:
        self.metrics: dict[str, dict[str, float]] = {}

    def update_db_metrics(self, metrics: dict[str, dict[str, float]]) -> None:
        self.metrics.update(metrics)


class DummyPatchDB:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def record_vector_metrics(
        self,
        session_id,
        pairs,
        *,
        patch_id,
        contribution,
        win,
        regret,
    ):
        self.kwargs = {
            "session_id": session_id,
            "pairs": pairs,
            "patch_id": patch_id,
            "contribution": contribution,
            "win": win,
            "regret": regret,
        }


def patch_metrics(monkeypatch):
    call = DummyGauge()
    lat = DummyGauge()
    size = DummyGauge()
    monkeypatch.setattr(dec, "_CALL_COUNT", call)
    monkeypatch.setattr(dec, "_LATENCY_GAUGE", lat)
    monkeypatch.setattr(dec, "_RESULT_SIZE_GAUGE", size)
    import vector_service.patch_logger as pl_mod
    import vector_service.embedding_backfill as eb_mod
    outcome = DummyGauge()
    duration = DummyGauge()
    run_outcome = DummyGauge()
    run_duration = DummyGauge()
    monkeypatch.setattr(pl_mod, "_TRACK_OUTCOME", outcome)
    monkeypatch.setattr(pl_mod, "_TRACK_DURATION", duration)
    monkeypatch.setattr(eb_mod, "_RUN_OUTCOME", run_outcome)
    monkeypatch.setattr(eb_mod, "_RUN_DURATION", run_duration)
    return call, lat, size, outcome, duration, run_outcome, run_duration


# ---------------------------------------------------------------------------
# PatchLogger.track_contributors
# ---------------------------------------------------------------------------


def test_track_contributors_success_emits_metrics(monkeypatch):
    call, lat, size, outcome, duration, _, _ = patch_metrics(monkeypatch)
    db = DummyMetricsDB()
    pl = PatchLogger(metrics_db=db)
    pl.track_contributors(["a:b", "c"], True, patch_id="1", session_id="s")
    assert db.called
    assert ("inc", "PatchLogger.track_contributors", 1.0) in call.calls
    assert any(c[0] == "set" and c[1] == "PatchLogger.track_contributors" for c in lat.calls)
    assert any(c[0] == "set" and c[1] == "PatchLogger.track_contributors" for c in size.calls)
    assert ("inc", "success", 1.0) in outcome.calls
    assert any(c[0] == "set" for c in duration.calls)


def test_track_contributors_db_failure_still_counts(monkeypatch):
    call, lat, size, outcome, duration, _, _ = patch_metrics(monkeypatch)
    db = DummyMetricsDB(fail=True)
    pl = PatchLogger(metrics_db=db)
    pl.track_contributors(["x"], False, patch_id="2", session_id="s")
    assert db.called
    assert ("inc", "PatchLogger.track_contributors", 1.0) in call.calls
    assert ("inc", "failure", 1.0) in outcome.calls
    assert any(c[0] == "set" for c in duration.calls)


def test_track_contributors_malformed_vector_ids_raise_and_count(monkeypatch):
    call, lat, size, outcome, duration, _, _ = patch_metrics(monkeypatch)
    pl = PatchLogger()
    with pytest.raises(TypeError):
        pl.track_contributors(["ok", None], True, session_id="s")
    assert ("inc", "PatchLogger.track_contributors", 1.0) in call.calls
    assert ("inc", "error", 1.0) in outcome.calls
    assert any(c[0] == "set" for c in duration.calls)


def test_track_contributors_forwards_contribution_vector_metrics(monkeypatch):
    _, _, _, _, _, _, _ = patch_metrics(monkeypatch)
    vm = DummyVectorMetricsDB()
    pl = PatchLogger(vector_metrics=vm)
    pl.track_contributors(["v1"], True, patch_id="p", session_id="s", contribution=0.3)
    assert vm.calls and vm.calls[0]["contribution"] == 0.3


def test_track_contributors_forwards_contribution_patch_db(monkeypatch):
    _, _, _, _, _, _, _ = patch_metrics(monkeypatch)
    pdb = DummyPatchDB()
    pl = PatchLogger(patch_db=pdb)
    pl.track_contributors(["v2"], False, patch_id="7", session_id="s", contribution=0.8)
    assert pdb.kwargs and pdb.kwargs["contribution"] == 0.8


def test_track_contributors_forwards_roi_feedback(monkeypatch):
    _, _, _, _, _, _, _ = patch_metrics(monkeypatch)
    vm = DummyVectorMetricsDB()
    rt = DummyROITracker()
    pl = PatchLogger(vector_metrics=vm, roi_tracker=rt)
    pl.track_contributors(["db1:v1", "db1:v2", "db2:v3"], True, session_id="s", contribution=0.5)
    # aggregated ROI per origin
    fb = {c["db"]: c["roi"] for c in vm.fb_calls}
    assert fb["db1"] == pytest.approx(1.0)
    assert fb["db2"] == pytest.approx(0.5)
    assert rt.metrics["db1"]["roi"] == pytest.approx(1.0)


def test_track_contributors_forwards_roi_feedback_failure(monkeypatch):
    _, _, _, _, _, _, _ = patch_metrics(monkeypatch)
    vm = DummyVectorMetricsDB()
    rt = DummyROITracker()
    pl = PatchLogger(vector_metrics=vm, roi_tracker=rt)
    pl.track_contributors(["db1:v1", "db1:v2", "db2:v3"], False, session_id="s", contribution=0.5)
    fb = {c["db"]: c["roi"] for c in vm.fb_calls}
    assert fb["db1"] == pytest.approx(1.0)
    assert fb["db2"] == pytest.approx(0.5)
    assert rt.metrics["db1"]["win_rate"] == pytest.approx(0.0)
    assert rt.metrics["db1"]["regret_rate"] == pytest.approx(1.0)


def test_track_contributors_emits_safety_metrics(monkeypatch):
    _, _, _, _, _, _, _ = patch_metrics(monkeypatch)
    events = []

    class Bus:
        def publish(self, topic, payload):
            if topic == "patch_logger:outcome":
                events.append(payload)

    pl = PatchLogger(event_bus=Bus(), max_alert_severity=5.0)
    meta = {
        "db1:v1": {
            "alignment_severity": 2.0,
            "semantic_alerts": ["a1", "a2"],
        }
    }
    pl.track_contributors(["db1:v1"], True, session_id="s", retrieval_metadata=meta)
    assert events
    ev = events[0]
    assert ev["alignment_severity"] == pytest.approx(2.0)
    assert ev["semantic_alerts"] == ["a1", "a2"]
    assert ev["roi_metrics"]["db1"]["alignment_severity"] == pytest.approx(2.0)
    assert ev["roi_metrics"]["db1"]["semantic_alerts"] == ["a1", "a2"]


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
    call, lat, size, _, _, run_outcome, run_duration = patch_metrics(monkeypatch)
    _processed.clear()
    monkeypatch.setattr(
        EmbeddingBackfill, "_load_known_dbs", lambda self, names=None: [RetryDB, FailDB]
    )
    eb = EmbeddingBackfill(batch_size=5)
    eb.run(session_id="sid")
    assert _processed == [RetryDB]
    assert ("inc", "EmbeddingBackfill.run", 1.0) in call.calls
    assert ("inc", "EmbeddingBackfill._process_db", 1.0) in call.calls
    assert ("inc", "success", 1.0) in run_outcome.calls
    assert any(c[0] == "set" for c in run_duration.calls)


def test_embedding_backfill_license_skip_metric(monkeypatch):
    patch_metrics(monkeypatch)

    class MultiGauge:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[str, str], float]] = []
            self.label: tuple[str, str] = ("", "")

        def labels(self, *labels):
            self.label = labels  # type: ignore[assignment]
            return self

        def inc(self, amount: float = 1.0) -> None:
            self.calls.append(("inc", self.label, amount))

    gauge = MultiGauge()
    import vector_service.embedding_backfill as eb_mod

    monkeypatch.setattr(eb_mod, "_RUN_SKIPPED", gauge)
    monkeypatch.setattr(eb_mod, "license_check", lambda text: "GPL")
    monkeypatch.setattr(eb_mod, "license_fingerprint", lambda text: "hash")
    monkeypatch.setattr(eb_mod, "_log_violation", lambda *a, **k: None)

    added: list[str] = []

    class LicenseDB(EmbeddableDBMixin):
        def add_embedding(self, record_id, record, kind, *, source_id=""):
            added.append(record_id)

        def backfill_embeddings(self, batch_size: int = 0) -> None:
            self.add_embedding("1", "text", "kind")

    monkeypatch.setattr(
        EmbeddingBackfill, "_load_known_dbs", lambda self, names=None: [LicenseDB]
    )
    EmbeddingBackfill().run()
    assert added == []
    assert ("inc", ("LicenseDB", "GPL"), 1.0) in gauge.calls


def test_track_contributors_patch_db_failure_logs_and_raises(monkeypatch, caplog):
    patch_metrics(monkeypatch)

    class FailPatchDB:
        def record_vector_metrics(self, *a, **k):
            raise RuntimeError("boom")

    pl = PatchLogger(patch_db=FailPatchDB())
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError):
            pl.track_contributors(["v"], True, patch_id="1", session_id="s")
    assert "record_vector_metrics" in caplog.text


def test_track_contributors_event_bus_failure_logs(monkeypatch, caplog):
    patch_metrics(monkeypatch)

    class FailBus:
        def publish(self, *a, **k):
            raise RuntimeError("bus down")

    pl = PatchLogger(event_bus=FailBus())
    with caplog.at_level(logging.ERROR):
        pl.track_contributors(["v"], True, session_id="s")
    assert "patch_logger outcome publish failed" in caplog.text
