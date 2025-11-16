from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import menace.ranking_model_scheduler as rms
from menace.unified_event_bus import UnifiedEventBus
from vector_service.patch_logger import PatchLogger
import pytest


def test_patch_logger_triggers_scheduler_on_roi(tmp_path, monkeypatch):
    class DummyService:
        def __init__(self) -> None:
            self.model_path: Path | None = None
            self.reliability_reloaded = False
            self.context_builder = SimpleNamespace(
                refresh_db_weights=lambda *a, **k: None
            )

        def reload_ranker_model(self, path: Path) -> None:
            self.model_path = Path(path)

        def reload_reliability_scores(self) -> None:
            self.reliability_reloaded = True

    bus = UnifiedEventBus()
    svc = DummyService()
    tracker = SimpleNamespace(metrics=[])

    def _upd(m):
        tracker.metrics.append(m)

    tracker.update_db_metrics = _upd  # type: ignore[attr-defined]

    sched = rms.RankingModelScheduler(
        [svc],
        vector_db=tmp_path / "vec.db",
        metrics_db=tmp_path / "metrics.db",
        model_path=tmp_path / "model.json",
        interval=0,
        roi_tracker=tracker,  # type: ignore[arg-type]
        roi_signal_threshold=0.5,
        event_bus=bus,
    )

    monkeypatch.setattr(rms, "compute_retriever_stats", lambda m: None)
    dummy_tm = rms.rr.TrainedModel(object(), [])
    monkeypatch.setattr(rms.rr, "load_training_data", lambda **kw: object())
    monkeypatch.setattr(rms.rr, "train", lambda df: (dummy_tm, {}))
    monkeypatch.setattr(
        rms.rr, "save_model", lambda tm, p: Path(p).write_text("{}")
    )

    captured = []
    bus.subscribe("patch_logger:outcome", lambda t, e: captured.append(e))

    pl = PatchLogger(event_bus=bus)
    pl.track_contributors(["db:v1"], True, roi_delta=0.6)

    cfg = json.loads((tmp_path / "model.json").read_text())
    current = Path(cfg["current"])
    assert svc.model_path == current
    assert svc.reliability_reloaded
    assert tracker.metrics and "db" in tracker.metrics[0]
    assert captured and captured[0]["roi_metrics"]["db"]["roi"] == pytest.approx(0.6)


def test_patch_logger_does_not_trigger_when_below_threshold(tmp_path, monkeypatch):
    class DummyService:
        def __init__(self) -> None:
            self.model_path: Path | None = None
            self.reliability_reloaded = False
            self.context_builder = SimpleNamespace(
                refresh_db_weights=lambda *a, **k: None
            )

        def reload_ranker_model(self, path: Path) -> None:
            self.model_path = Path(path)

        def reload_reliability_scores(self) -> None:
            self.reliability_reloaded = True

    bus = UnifiedEventBus()
    svc = DummyService()
    tracker = SimpleNamespace()
    tracker.update_db_metrics = lambda m: None  # type: ignore[attr-defined]

    sched = rms.RankingModelScheduler(
        [svc],
        vector_db=tmp_path / "vec.db",
        metrics_db=tmp_path / "metrics.db",
        model_path=tmp_path / "model.json",
        interval=0,
        roi_tracker=tracker,  # type: ignore[arg-type]
        roi_signal_threshold=2.0,
        event_bus=bus,
    )

    monkeypatch.setattr(rms, "compute_retriever_stats", lambda m: None)
    dummy_tm = rms.rr.TrainedModel(object(), [])
    monkeypatch.setattr(rms.rr, "load_training_data", lambda **kw: object())
    monkeypatch.setattr(rms.rr, "train", lambda df: (dummy_tm, {}))
    monkeypatch.setattr(
        rms.rr, "save_model", lambda tm, p: Path(p).write_text("{}")
    )

    pl = PatchLogger(event_bus=bus)
    pl.track_contributors(["db:v1"], True, roi_delta=0.5)

    assert svc.model_path is None
    assert not svc.reliability_reloaded
