import json
import sys
import types
from dynamic_path_router import resolve_path


def test_simulate_chains_executes_and_persists(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "WORKFLOW_ROI_HISTORY_PATH", str(resolve_path("roi_history.json", tmp_path))
    )

    wfm = types.ModuleType("workflow_evolution_manager")

    def _build_callable(sequence: str):
        modules = [m for m in sequence.split("-") if m]
        funcs = []
        for m in modules:
            def _default():
                return True

            try:
                mod = __import__(m)
                fn = getattr(mod, "main", getattr(mod, "run", _default))
            except Exception:
                fn = _default
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
    monkeypatch.setitem(sys.modules, "workflow_evolution_manager", wfm)

    class DummyResult:
        def __init__(self, success: bool):
            self.success_rate = 1.0 if success else 0.0
            self.roi_gain = 0.0

    class DummyScorer:
        def run(self, fn, wf_id, run_id=None):
            return DummyResult(bool(fn()))
    cws = types.ModuleType("composite_workflow_scorer")
    cws.CompositeWorkflowScorer = DummyScorer
    monkeypatch.setitem(sys.modules, "composite_workflow_scorer", cws)

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
    monkeypatch.setitem(sys.modules, "workflow_synergy_comparator", wsc)

    stab = types.ModuleType("workflow_stability_db")

    class DummyStabilityDB:
        def is_stable(self, *a, **k):
            return False

        def mark_stable(self, *a, **k):
            pass
    stab.WorkflowStabilityDB = DummyStabilityDB
    monkeypatch.setitem(sys.modules, "workflow_stability_db", stab)

    sugg = types.ModuleType("workflow_chain_suggester")

    class DummySuggester:
        def suggest_chains(self, *a, **k):
            return []
    sugg.WorkflowChainSuggester = DummySuggester
    monkeypatch.setitem(sys.modules, "workflow_chain_suggester", sugg)

    summary = types.ModuleType("workflow_run_summary")

    def record_run(*a, **k):
        pass
    summary.record_run = record_run
    monkeypatch.setitem(sys.modules, "workflow_run_summary", summary)

    mod_a = types.ModuleType("mod_a")
    mod_a.main = lambda: True
    mod_b = types.ModuleType("mod_b")
    mod_b.main = lambda: False
    monkeypatch.setitem(sys.modules, "mod_a", mod_a)
    monkeypatch.setitem(sys.modules, "mod_b", mod_b)

    sys.path.insert(0, str(resolve_path(".")))
    import importlib
    sys.modules.pop("workflow_chain_simulator", None)
    import workflow_chain_simulator as sim
    sim = importlib.reload(sim)
    monkeypatch.setattr(sim, "RESULTS_PATH", resolve_path("results.json", tmp_path))
    orig_persist = sim._persist_outcomes

    def _persist(outcomes, path=sim.RESULTS_PATH):
        return orig_persist(outcomes, path)

    monkeypatch.setattr(sim, "_persist_outcomes", _persist)

    results = sim.simulate_chains([["mod_a"], ["mod_b"]])

    assert len(results) == 2
    assert results[0]["failure_rate"] == 0.0
    assert results[1]["failure_rate"] == 1.0

    saved = json.loads(resolve_path("results.json", tmp_path).read_text())
    assert saved[0]["chain"] == ["mod_a"]
    assert saved[1]["chain"] == ["mod_b"]


def test_simulate_suggested_chains_handles_failed_runs(tmp_path, monkeypatch):
    """Chains suggested by the suggester are executed and failures recorded."""
    monkeypatch.setenv(
        "WORKFLOW_ROI_HISTORY_PATH", str(resolve_path("roi_history.json", tmp_path))
    )

    wfm = types.ModuleType("workflow_evolution_manager")

    def _build_callable(sequence: str):
        def _wf():
            if "error" in sequence:
                raise RuntimeError("boom")
            return True

        return _wf

    wfm._build_callable = _build_callable
    monkeypatch.setitem(sys.modules, "workflow_evolution_manager", wfm)

    class DummyResult:
        def __init__(self, ok: bool):
            self.success_rate = 1.0 if ok else 0.0
            self.roi_gain = 0.0

    class DummyScorer:
        def run(self, fn, wf_id, run_id=None):
            try:
                ok = bool(fn())
            except Exception:
                ok = False
            return DummyResult(ok)

    cws = types.ModuleType("composite_workflow_scorer")
    cws.CompositeWorkflowScorer = DummyScorer
    monkeypatch.setitem(sys.modules, "composite_workflow_scorer", cws)

    wsc = types.ModuleType("workflow_synergy_comparator")
    wsc.WorkflowSynergyComparator = type("WSC", (), {"_entropy": staticmethod(lambda spec: 0.0)})
    monkeypatch.setitem(sys.modules, "workflow_synergy_comparator", wsc)

    stab = types.ModuleType("workflow_stability_db")

    class DummyStabilityDB:
        def is_stable(self, *a, **k):
            return False

        def mark_stable(self, *a, **k):
            pass

    stab.WorkflowStabilityDB = DummyStabilityDB
    monkeypatch.setitem(sys.modules, "workflow_stability_db", stab)

    summary = types.ModuleType("workflow_run_summary")
    summary.record_run = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "workflow_run_summary", summary)

    mod_ok = types.ModuleType("mod_ok")
    mod_ok.main = lambda: True
    monkeypatch.setitem(sys.modules, "mod_ok", mod_ok)

    sys.path.insert(0, str(resolve_path(".")))
    import importlib
    import workflow_chain_simulator as sim
    sim = importlib.reload(sim)

    class DummySuggester:
        def suggest_chains(self, *a, **k):
            return [["mod_ok"], ["mod_error"]]

    monkeypatch.setattr(sim, "WorkflowChainSuggester", DummySuggester)
    monkeypatch.setattr(sim, "RESULTS_PATH", resolve_path("results.json", tmp_path))
    monkeypatch.setattr(sim, "_persist_outcomes", lambda *a, **k: None)

    results = sim.simulate_suggested_chains([0.0], top_k=2)

    assert len(results) == 2
    assert results[0]["failure_rate"] == 0.0
    assert results[1]["failure_rate"] == 1.0
