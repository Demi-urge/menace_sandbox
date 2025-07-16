import menace.action_planner as ap
from menace.neuroplasticity import PathwayDB, PathwayRecord, Outcome
from menace.unified_event_bus import UnifiedEventBus
from dataclasses import dataclass
from datetime import datetime


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
