import sqlite3
import sys
import types


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
        GLOBAL_ROUTER=None,
    ),
)
sys.modules.setdefault("db_router", sys.modules["menace_sandbox.db_router"])

from menace_sandbox.db_router import init_db_router
from menace_sandbox.roi_results_db import ROIResultsDB
import ast
from pathlib import Path
from typing import Any, Callable, Dict

ENG_PATH = Path(__file__).resolve().parents[1] / "self_improvement" / "engine.py"  # path-ignore
src = ENG_PATH.read_text()
tree = ast.parse(src)
future = ast.ImportFrom(module="__future__", names=[ast.alias("annotations", None)], level=0)
sie_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SelfImprovementEngine")
method = next(n for n in sie_cls.body if isinstance(n, ast.FunctionDef) and n.name == "_check_chain_stagnation")
engine_module = ast.Module([future, ast.ClassDef("SelfImprovementEngine", [], [], [method], [])], type_ignores=[])
engine_module = ast.fix_missing_locations(engine_module)
alerts: list[tuple] = []
ns: Dict[str, Any] = {
    "dispatch_alert": lambda *a, **k: alerts.append((a, k)),
}
exec(compile(engine_module, "<engine>", "exec"), ns)
sie = types.SimpleNamespace(SelfImprovementEngine=ns["SelfImprovementEngine"], dispatch_alert=ns["dispatch_alert"], alerts=alerts)


def _setup_db(tmp_path):
    init_db_router(
        "test_chain_roi",
        local_db_path=str(tmp_path / "local.db"),
        shared_db_path=str(tmp_path / "shared.db"),
    )
    return ROIResultsDB(tmp_path / "roi.db")


def _log(db: ROIResultsDB, roi: float, run: int) -> None:
    db.log_result(
        workflow_id="wf",
        run_id=f"r{run}",
        runtime=1.0,
        success_rate=1.0,
        roi_gain=roi,
        workflow_synergy_score=0.0,
        bottleneck_index=0.0,
        patchability_score=0.0,
    )


def _make_engine(db: ROIResultsDB, monkeypatch):
    eng = sie.SelfImprovementEngine.__new__(sie.SelfImprovementEngine)
    eng.roi_db = db
    eng.urgency_tier = 0
    eng.logger = types.SimpleNamespace(exception=lambda *a, **k: None)
    monkeypatch.setattr(sie, "dispatch_alert", sie.dispatch_alert)
    sie.alerts.clear()
    return eng, sie.alerts


def test_roi_drop_and_recovery(tmp_path, monkeypatch):
    db = _setup_db(tmp_path)
    _log(db, 1.0, 1)
    _log(db, 0.8, 2)
    _log(db, 0.7, 3)
    stats = db.fetch_chain_stats("wf")
    assert stats["non_positive_streak"] == 3

    eng, alerts = _make_engine(db, monkeypatch)
    eng._check_chain_stagnation()
    assert eng.urgency_tier == 1
    assert alerts and alerts[0][0][3]["workflow_id"] == "wf"

    _log(db, 1.2, 4)
    stats = db.fetch_chain_stats("wf")
    assert stats["non_positive_streak"] == 0
    alerts.clear()
    eng._check_chain_stagnation()
    assert eng.urgency_tier == 1
    assert alerts == []


def test_roi_stagnation_escalates(tmp_path, monkeypatch):
    db = _setup_db(tmp_path)
    _log(db, 1.0, 1)
    _log(db, 1.0, 2)
    _log(db, 1.0, 3)
    stats = db.fetch_chain_stats("wf")
    assert stats["non_positive_streak"] == 3
    eng, alerts = _make_engine(db, monkeypatch)
    eng._check_chain_stagnation()
    assert eng.urgency_tier == 1
    assert alerts and alerts[0][0][3]["workflow_id"] == "wf"
