import sqlite3
import sys
import types
import pytest


class _StubDBRouter:
    def __init__(self, _name, local_path, _shared_path):
        self.path = local_path

    def get_connection(self, _name, operation: str | None = None):
        return sqlite3.connect(self.path)


sys.modules.setdefault(
    "menace_sandbox.db_router",
    types.SimpleNamespace(
        init_db_router=lambda *a, **k: None,
        DBRouter=_StubDBRouter,
        LOCAL_TABLES=set(),
    ),
)
sys.modules.setdefault("db_router", sys.modules["menace_sandbox.db_router"])

from menace_sandbox.db_router import init_db_router
from menace_sandbox.roi_results_db import ROIResultsDB
from menace_sandbox.composite_workflow_scorer import CompositeWorkflowScorer


class StubTracker:
    def __init__(self):
        self.roi_history = []
        self.module_deltas: dict[str, list[float]] = {}
        self.timings = {}
        self.scheduling_overhead = {}
        self.correlation_history = {}
        self._counter: dict[str, int] = {}

    def update(self, delta, roi_after, modules, metrics, profile_type):
        for mod in modules:
            count = self._counter.get(mod, 0) + 1
            self._counter[mod] = count
            self.roi_history.append(float(count))
            self.module_deltas.setdefault(mod, []).append(float(count))

    def cache_correlations(self, correlations):
        self.correlation_history.update(correlations)


from menace_sandbox.roi_calculator import ROIResult


class StubCalculator:
    profiles = {"default": {}}

    def calculate(self, metrics, profile_type):
        return ROIResult(0.0, False, [])


def test_score_workflow_records_and_updates(tmp_path):
    init_db_router(
        "test_scorer_db",
        local_db_path=str(tmp_path / "local.db"),
        shared_db_path=str(tmp_path / "shared.db"),
    )

    db_path = tmp_path / "roi.db"
    db = ROIResultsDB(db_path)
    tracker = StubTracker()
    scorer = CompositeWorkflowScorer(
        tracker=tracker,
        calculator_factory=lambda: StubCalculator(),
        results_db=db,
    )

    modules = {"alpha": lambda: True, "beta": lambda: True}
    scorer.score_workflow("wf", modules, run_id="r1")
    scorer.score_workflow("wf", modules, run_id="r2")

    cur = db.conn.cursor()
    cur.execute("SELECT COUNT(*) FROM workflow_results")
    assert cur.fetchone()[0] == 2

    cur.execute("SELECT COUNT(*) FROM workflow_module_deltas")
    assert cur.fetchone()[0] == 4

    cur.execute(
        """
        SELECT roi_delta_ma, roi_delta_var FROM workflow_module_deltas
        WHERE workflow_id=? AND module=? ORDER BY ts
        """,
        ("wf", "alpha"),
    )
    rows = cur.fetchall()
    assert rows[0][0] == pytest.approx(1.0)
    assert rows[0][1] == pytest.approx(0.0)
    assert rows[1][0] == pytest.approx(1.5)
    assert rows[1][1] == pytest.approx(0.25)
