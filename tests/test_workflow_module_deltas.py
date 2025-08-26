import os
from collections import defaultdict
import types
import sys

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

sys.modules.setdefault("menace_sandbox.self_test_service", types.ModuleType("self_test_service"))
sys.modules.setdefault("menace_sandbox.error_bot", types.ModuleType("error_bot"))
sys.modules.setdefault(
    "menace_sandbox.menace_memory_manager", types.ModuleType("menace_memory_manager")
)
sys.modules.setdefault(
    "menace_sandbox.chatgpt_enhancement_bot", types.ModuleType("chatgpt_enhancement_bot")
)
sys.modules.setdefault(
    "menace_sandbox.chatgpt_idea_bot", types.ModuleType("chatgpt_idea_bot")
)
run_auto = types.ModuleType("run_autonomous")
run_auto._verify_required_dependencies = lambda: None
sys.modules.setdefault("menace_sandbox.run_autonomous", run_auto)

import menace_sandbox.roi_scorer as rs  # noqa: E402
from menace_sandbox.roi_tracker import (  # noqa: E402
    load_workflow_module_deltas,
    apply_workflow_module_deltas,
)


class DummyMetricsDB:
    def __init__(self):
        self.data = defaultdict(list)
        self.router = None

    def log_eval(self, name, metric, value):
        self.data[name].append((len(self.data[name]), metric, value, None))

    def fetch_eval(self, name):
        return list(self.data.get(name, []))


class DummyPathwayDB:
    def __init__(self):
        self.router = None

    def log(self, rec):
        pass


class DummyTracker:
    def __init__(self):
        self.metrics_history = {}
        self.roi_history = []
        self.module_deltas = {}


def test_module_deltas_persistence(tmp_path, monkeypatch):
    tracker = DummyTracker()
    metrics_db = DummyMetricsDB()
    pathway_db = DummyPathwayDB()
    db_path = tmp_path / "roi_results.db"
    scorer = rs.CompositeWorkflowScorer(metrics_db, pathway_db, db_path=db_path, tracker=tracker)

    def stub_benchmark(func, metrics_db, pathway_db, name="wf"):
        metrics_db.log_eval(name, "alpha_runtime", 1.0)
        metrics_db.log_eval(name, "beta_time", 2.0)
        return func()

    import menace_sandbox.workflow_benchmark as wb
    monkeypatch.setattr(wb, "benchmark_workflow", stub_benchmark)

    def alpha() -> bool:
        tracker.module_deltas.setdefault("alpha", []).append(0.3)
        return True

    def beta() -> bool:
        tracker.module_deltas.setdefault("beta", []).append(-0.1)
        return True

    run_id, _ = scorer.score_workflow("wf", {"alpha": alpha, "beta": beta})
    assert scorer.module_deltas() == {"alpha": 0.3, "beta": -0.1}

    from menace_sandbox.db_router import DBRouter, LOCAL_TABLES

    LOCAL_TABLES.add("workflow_module_deltas")
    conn = DBRouter("test", str(db_path), str(db_path)).get_connection(
        "workflow_module_deltas", operation="read"
    )
    rows = conn.execute(
        (
            "SELECT module, runtime, roi_delta FROM workflow_module_deltas "
            "WHERE workflow_id=? AND run_id=? ORDER BY module"
        ),
        ("wf", run_id),
    ).fetchall()
    assert rows == [("alpha", 1.0, 0.3), ("beta", 2.0, -0.1)]

    deltas = load_workflow_module_deltas("wf", run_id, db_path)
    assert deltas == {"alpha": 0.3, "beta": -0.1}

    new_tracker = DummyTracker()
    apply_workflow_module_deltas(new_tracker, "wf", run_id, db_path)
    assert new_tracker.module_deltas == {"alpha": [0.3], "beta": [-0.1]}
