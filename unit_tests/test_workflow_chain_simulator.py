import json
import sys
import types
from pathlib import Path


def test_simulate_chains_executes_and_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKFLOW_ROI_HISTORY_PATH", str(tmp_path / "roi_history.json"))

    wfm = types.ModuleType("workflow_evolution_manager")
    def _build_callable(sequence: str):
        modules = [m for m in sequence.split("-") if m]
        funcs = []
        for m in modules:
            try:
                mod = __import__(m)
                fn = getattr(mod, "main", getattr(mod, "run", lambda: True))
            except Exception:
                fn = lambda: True
            funcs.append(fn)
        def _wf():
            ok = True
            for fn in funcs:
                try:
                    ok = bool(fn()) and ok
                except Exception:
                    ok = False
            return ok
        return _wf
    wfm._build_callable = _build_callable
    sys.modules["workflow_evolution_manager"] = wfm

    class DummyResult:
        def __init__(self, success: bool):
            self.success_rate = 1.0 if success else 0.0
            self.roi_gain = 0.0
    class DummyScorer:
        def run(self, fn, wf_id, run_id=None):
            return DummyResult(bool(fn()))
    cws = types.ModuleType("composite_workflow_scorer")
    cws.CompositeWorkflowScorer = DummyScorer
    sys.modules["composite_workflow_scorer"] = cws

    wsc = types.ModuleType("workflow_synergy_comparator")
    def _entropy(spec):
        modules = [s.get("module") for s in spec.get("steps", [])]
        counts = {}
        for m in modules:
            counts[m] = counts.get(m, 0) + 1
        total = sum(counts.values())
        import math
        return -sum((c/total)*math.log(c/total, 2) for c in counts.values()) if total else 0.0
    wsc.WorkflowSynergyComparator = type("WSC", (), {"_entropy": staticmethod(_entropy)})
    sys.modules["workflow_synergy_comparator"] = wsc

    stab = types.ModuleType("workflow_stability_db")
    class DummyStabilityDB:
        def is_stable(self, *a, **k):
            return False
        def mark_stable(self, *a, **k):
            pass
    stab.WorkflowStabilityDB = DummyStabilityDB
    sys.modules["workflow_stability_db"] = stab

    sugg = types.ModuleType("workflow_chain_suggester")
    class DummySuggester:
        def suggest_chains(self, *a, **k):
            return []
    sugg.WorkflowChainSuggester = DummySuggester
    sys.modules["workflow_chain_suggester"] = sugg

    summary = types.ModuleType("workflow_run_summary")
    def record_run(*a, **k):
        pass
    summary.record_run = record_run
    sys.modules["workflow_run_summary"] = summary

    mod_a = types.ModuleType("mod_a")
    mod_a.main = lambda: True
    mod_b = types.ModuleType("mod_b")
    mod_b.main = lambda: False
    sys.modules["mod_a"] = mod_a
    sys.modules["mod_b"] = mod_b

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import workflow_chain_simulator as sim
    monkeypatch.setattr(sim, "RESULTS_PATH", tmp_path / "results.json")
    orig_persist = sim._persist_outcomes
    monkeypatch.setattr(sim, "_persist_outcomes", lambda outcomes, path=sim.RESULTS_PATH: orig_persist(outcomes, path))

    results = sim.simulate_chains([["mod_a"], ["mod_b"]])

    assert len(results) == 2
    assert results[0]["failure_rate"] == 0.0
    assert results[1]["failure_rate"] == 1.0

    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved[0]["chain"] == ["mod_a"]
    assert saved[1]["chain"] == ["mod_b"]
