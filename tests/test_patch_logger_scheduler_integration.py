from __future__ import annotations

from pathlib import Path

import menace.ranking_model_scheduler as rms
from menace.unified_event_bus import UnifiedEventBus
from vector_service.patch_logger import PatchLogger


def test_patch_logger_triggers_scheduler(tmp_path, monkeypatch):
    class DummyService:
        def __init__(self) -> None:
            self.model_path: Path | None = None
            self.reliability_reloaded = False

        def reload_ranker_model(self, path: Path) -> None:
            self.model_path = Path(path)

        def reload_reliability_scores(self) -> None:
            self.reliability_reloaded = True

    bus = UnifiedEventBus()
    svc = DummyService()
    sched = rms.RankingModelScheduler(
        [svc],
        vector_db=tmp_path / "vec.db",
        metrics_db=tmp_path / "metrics.db",
        model_path=tmp_path / "model.json",
        interval=0,
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
    pl.track_contributors(["vec1"], True)

    assert svc.model_path == tmp_path / "model.json"
    assert svc.reliability_reloaded
