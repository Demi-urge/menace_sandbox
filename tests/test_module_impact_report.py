import sys
import types
from collections import defaultdict

os = __import__('os')

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

sys.modules.setdefault("menace_sandbox.self_test_service", types.ModuleType("self_test_service"))
sys.modules.setdefault("menace_sandbox.error_bot", types.ModuleType("error_bot"))
run_auto = types.ModuleType("run_autonomous")
run_auto._verify_required_dependencies = lambda: None
sys.modules.setdefault("menace_sandbox.run_autonomous", run_auto)

import menace_sandbox.roi_scorer as rs  # noqa: E402
from menace_sandbox.roi_results_db import module_impact_report  # noqa: E402


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

    def update(self, roi_before, roi_after, modules=None, **kwargs):
        if modules:
            delta = roi_after - roi_before
            for m in modules:
                self.module_deltas.setdefault(m, []).append(delta)
        return None, [], False, False


def test_module_impact_report(tmp_path, monkeypatch):
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
        return True

    def beta() -> bool:
        return True

    def calc_seq(vals):
        it = iter(vals)
        return lambda metrics, profile_type: (next(it), None, None)

    monkeypatch.setattr(scorer.calculator, "calculate", calc_seq([0.0, 0.1, 0.2]))
    run1, _ = scorer.score_workflow("wf", {"alpha": alpha, "beta": beta})

    monkeypatch.setattr(scorer.calculator, "calculate", calc_seq([0.0, 0.2, 0.1]))
    run2, _ = scorer.score_workflow("wf", {"alpha": alpha, "beta": beta})

    report = module_impact_report("wf", run2, db_path)
    assert report["improved"] == {"alpha": 0.1}
    assert report["regressed"] == {"beta": -0.1}
