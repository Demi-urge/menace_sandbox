from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import menace.ranking_model_scheduler as rms


def test_scheduler_trains_and_reloads(tmp_path, monkeypatch):
    # dummy service recording reload calls
    class DummyService:
        def __init__(self) -> None:
            self.model_path: Path | None = None
            self.reliability_reloaded = False

        def reload_ranker_model(self, path: Path) -> None:
            self.model_path = Path(path)

        def reload_reliability_scores(self) -> None:
            self.reliability_reloaded = True

    svc = DummyService()

    sched = rms.RankingModelScheduler([svc],
                                      vector_db=tmp_path / "vec.db",
                                      metrics_db=tmp_path / "metrics.db",
                                      model_path=tmp_path / "model.json",
                                      interval=0)

    # stub heavy dependencies
    monkeypatch.setattr(rms, "VectorMetricsDB", lambda path: object())
    monkeypatch.setattr(rms.rr, "prepare_training_dataframe", lambda db: object())
    dummy_model = SimpleNamespace(coef_=[[1.0]], intercept_=[0.0], classes_=[0, 1])
    monkeypatch.setattr(rms.rr, "train_retrieval_ranker", lambda df: (dummy_model, ["x"]))
    monkeypatch.setattr(rms.rr, "save_model", lambda m, f, p: Path(p).write_text("{}"))

    stats_called: list[Path] = []
    monkeypatch.setattr(rms, "compute_retriever_stats", lambda m: stats_called.append(Path(m)))

    sched.retrain_and_reload()

    assert svc.model_path == tmp_path / "model.json"
    assert svc.reliability_reloaded
    assert stats_called == [tmp_path / "metrics.db"]
    assert (tmp_path / "model.json").exists()

