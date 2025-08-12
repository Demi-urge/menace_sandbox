import pytest
import menace.action_planner as ap
from menace.neuroplasticity import PathwayDB, PathwayRecord, Outcome
from menace.roi_tracker import ROITracker


class DummyROIDB:
    class _Col(list):
        def mean(self) -> float:  # pragma: no cover - minimal stub
            return 0.0

    class _DF:
        empty = True

        def __getitem__(self, key: str) -> "DummyROIDB._Col":  # pragma: no cover - minimal stub
            return DummyROIDB._Col([])

    def history(self, bot: str | None = None, limit: int = 50) -> "DummyROIDB._DF":
        return DummyROIDB._DF()


feature_map = {"A": 0.0, "B": 1.0, "C": 2.0}


def feature_fn(action: str) -> list[float]:
    return [feature_map[action]]


class Predictor:
    mapping = {1.0: (0.9, "linear"), 2.0: (0.1, "linear")}

    def predict(self, feats, horizon=None):  # pragma: no cover - simple stub
        feat = feats[0][0]
        return self.mapping.get(feat, (0.0, "marginal"))


def test_adaptive_roi_sequence(tmp_path):
    tracker = ROITracker()
    planner = ap.ActionPlanner(
        PathwayDB(tmp_path / "p.db"),
        DummyROIDB(),
        epsilon=0.0,
        reward_fn=lambda a, r: r.roi,
        feature_fn=feature_fn,
        roi_predictor=Predictor(),
        roi_tracker=tracker,
        use_adaptive_roi=True,
    )

    order_before = planner.plan_actions("A", ["B", "C"])
    assert order_before == ["B", "C"]

    rec = PathwayRecord(
        actions="A->B",
        inputs="",
        outputs="",
        exec_time=1.0,
        resources="",
        outcome=Outcome.FAILURE,
        roi=-5.0,
    )
    planner._update_from_record(rec)

    assert tracker.predicted_roi[-1] == pytest.approx(0.9)
    assert tracker.actual_roi[-1] == pytest.approx(-5.0)

    order_after = planner.plan_actions("A", ["B", "C"])
    assert order_after == ["C", "B"]
