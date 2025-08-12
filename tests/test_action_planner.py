import menace.action_planner as ap
from menace.neuroplasticity import PathwayDB, PathwayRecord, Outcome
from menace.unified_event_bus import UnifiedEventBus
from dataclasses import dataclass
from datetime import datetime
import pytest


@dataclass
class KPIRecord:
    bot: str
    revenue: float
    api_cost: float
    cpu_seconds: float
    success_rate: float
    ts: str = datetime.utcnow().isoformat()


class DummyROIDB:
    def __init__(self) -> None:
        self.records: list[KPIRecord] = []

    def add(self, rec: KPIRecord) -> int:
        self.records.append(rec)
        return len(self.records)

    class _Col(list):
        def mean(self) -> float:
            return sum(self) / len(self) if self else 0.0

    class _DF:
        def __init__(self, rows: list[dict]):
            self.rows = rows

        @property
        def empty(self) -> bool:
            return not self.rows

        def __getitem__(self, key: str) -> "DummyROIDB._Col":
            return DummyROIDB._Col([r[key] for r in self.rows])

    def history(self, bot: str | None = None, limit: int = 50) -> "DummyROIDB._DF":
        if bot:
            recs = [r.__dict__ for r in self.records if r.bot == bot]
        else:
            recs = [r.__dict__ for r in self.records]
        return DummyROIDB._DF(recs[-limit:])

    def future_roi(self, action: str, discount: float = 0.9) -> float:
        df = self.history(action, limit=5)
        if df.empty or len(df.rows) < 2:
            return 0.0
        rois = [
            (r["revenue"] - r["api_cost"]) / (r["cpu_seconds"] or 1.0) * r["success_rate"]
            for r in df.rows
        ]
        trend = rois[-1] - rois[0]
        return (rois[-1] + trend) * discount


def test_predict_next_action(tmp_path):
    pdb = PathwayDB(tmp_path / "p.db")
    roi = DummyROIDB()
    roi.add(KPIRecord(bot="B", revenue=10.0, api_cost=1.0, cpu_seconds=1.0, success_rate=1.0))
    roi.add(KPIRecord(bot="C", revenue=5.0, api_cost=1.0, cpu_seconds=1.0, success_rate=1.0))
    pdb.log(PathwayRecord(actions="A->B", inputs="", outputs="", exec_time=1.0, resources="", outcome=Outcome.SUCCESS, roi=1.0))
    pdb.log(PathwayRecord(actions="A->C", inputs="", outputs="", exec_time=1.0, resources="", outcome=Outcome.SUCCESS, roi=1.0))
    planner = ap.ActionPlanner(
        pdb,
        roi,
        epsilon=0.0,
        model_path=str(tmp_path / "model.pkl"),
    )
    assert planner.predict_next_action("A") == "B"


def test_roi_prediction_prioritizes_high_estimate(tmp_path):
    class Predictor:
        mapping = {1.0: (0.1, "linear"), 2.0: (0.9, "linear")}

        def predict(self, feats):
            feat = feats[0][0]
            return self.mapping.get(feat, (0.0, "marginal"))

    feature_map = {"B": 1.0, "C": 2.0}

    def feature_fn(action: str):
        return [feature_map[action]]

    pdb = PathwayDB(tmp_path / "p.db")
    roi = DummyROIDB()
    pdb.log(
        PathwayRecord(
            actions="A->B",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.SUCCESS,
            roi=1.0,
        )
    )
    pdb.log(
        PathwayRecord(
            actions="A->C",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.SUCCESS,
            roi=1.0,
        )
    )
    planner = ap.ActionPlanner(
        pdb,
        roi,
        epsilon=0.0,
        reward_fn=lambda a, r: 1.0,
        feature_fn=feature_fn,
        roi_predictor=Predictor(),
        use_adaptive_roi=True,
    )
    assert planner.predict_next_action("A") == "C"


def test_planner_updates_on_event(tmp_path):
    bus = UnifiedEventBus()
    pdb = PathwayDB(tmp_path / "p.db", event_bus=bus)
    roi = DummyROIDB()
    roi.add(KPIRecord(bot="B", revenue=1.0, api_cost=1.0, cpu_seconds=1.0, success_rate=1.0))
    roi.add(KPIRecord(bot="C", revenue=10.0, api_cost=1.0, cpu_seconds=1.0, success_rate=1.0))
    planner = ap.ActionPlanner(
        pdb,
        roi,
        event_bus=bus,
        epsilon=0.0,
        model_path=str(tmp_path / "model.pkl"),
    )
    pdb.log(PathwayRecord(actions="A->B", inputs="", outputs="", exec_time=1.0, resources="", outcome=Outcome.SUCCESS, roi=1.0))
    assert planner.predict_next_action("A") == "B"
    pdb.log(PathwayRecord(actions="A->C", inputs="", outputs="", exec_time=1.0, resources="", outcome=Outcome.SUCCESS, roi=1.0))
    assert planner.predict_next_action("A") == "C"


def test_reward_scaled_by_growth(tmp_path):
    class StubPredictor:
        def __init__(self) -> None:
            self.category = "exponential"

        def predict(self, feats):
            return 0.0, self.category

    pdb = PathwayDB(tmp_path / "p.db")
    roi = DummyROIDB()
    predictor = StubPredictor()
    planner = ap.ActionPlanner(
        pdb,
        roi,
        reward_fn=lambda a, r: 1.0,
        feature_fn=lambda a: [0.0],
        roi_predictor=predictor,
        use_adaptive_roi=True,
        growth_weighting=True,
        growth_multipliers={"exponential": 2.0, "linear": 1.0, "marginal": 0.5},
    )
    rec = PathwayRecord(
        actions="A->B",
        inputs="",
        outputs="",
        exec_time=1.0,
        resources="",
        outcome=Outcome.SUCCESS,
        roi=1.0,
    )
    reward_exp = planner._reward("B", rec)
    predictor.category = "marginal"
    reward_marg = planner._reward("B", rec)
    assert reward_exp == pytest.approx(2.0)
    assert reward_marg == pytest.approx(0.5)


def test_reward_scaled_by_growth_confidence(tmp_path):
    class StubPredictor:
        def __init__(self) -> None:
            self.conf = 0.5

        def predict(self, feats, horizon=None):
            return [0.0], "exponential", [], self.conf

    pdb = PathwayDB(tmp_path / "p.db")
    roi = DummyROIDB()
    predictor = StubPredictor()
    planner = ap.ActionPlanner(
        pdb,
        roi,
        reward_fn=lambda a, r: 1.0,
        feature_fn=lambda a: [0.0],
        roi_predictor=predictor,
        use_adaptive_roi=True,
        growth_weighting=True,
        growth_multipliers={"exponential": 2.0, "linear": 1.0, "marginal": 0.5},
    )
    rec = PathwayRecord(
        actions="A->B",
        inputs="",
        outputs="",
        exec_time=1.0,
        resources="",
        outcome=Outcome.SUCCESS,
        roi=1.0,
    )
    reward_conf = planner._reward("B", rec)
    predictor.conf = None
    reward_none = planner._reward("B", rec)
    assert reward_conf == pytest.approx(1.0)
    assert reward_none == pytest.approx(2.0)


def test_plan_actions_uses_roi_growth(tmp_path):
    class Predictor:
        def predict(self, feats, horizon=None):
            val = feats[0][0]
            if val == 1.0:
                return [1.0], "marginal", 1.0
            return [0.5], "exponential", 1.0

    feature_map = {"B": 1.0, "C": 2.0}

    def feature_fn(action: str):
        return [feature_map[action]]

    pdb = PathwayDB(tmp_path / "p.db")
    roi = DummyROIDB()
    planner = ap.ActionPlanner(
        pdb,
        roi,
        feature_fn=feature_fn,
        roi_predictor=Predictor(),
        use_adaptive_roi=True,
    )
    ranked = planner.plan_actions("A", ["B", "C"])
    assert ranked[0] == "C"
