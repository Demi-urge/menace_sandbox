import json

from menace_sandbox.foresight_tracker import ForesightTracker
import asyncio
import pytest


class DummyROITracker:
    def __init__(self, deltas):
        self._deltas = iter(deltas)
        self.raroi_history = [0.0]
        self.confidence_history = [0.0]
        self.metrics_history = {"synergy_resilience": [0.0]}

    def next_delta(self):
        delta = next(self._deltas)
        self.raroi_history.append(self.raroi_history[-1] + delta / 2.0)
        return delta

    def scenario_degradation(self):
        return 0.0


class MiniSelfImprovementEngine:
    def __init__(self, tracker, foresight_tracker):
        self.tracker = tracker
        self.foresight_tracker = foresight_tracker
        self.workflow_ready = False

    def run_cycle(self, workflow_id="wf"):
        delta = self.tracker.next_delta()
        raroi_delta = self.tracker.raroi_history[-1] - self.tracker.raroi_history[-2]
        confidence = self.tracker.confidence_history[-1]
        resilience = self.tracker.metrics_history["synergy_resilience"][-1]
        scenario_deg = self.tracker.scenario_degradation()
        self.foresight_tracker.record_cycle_metrics(
            workflow_id,
            {
                "roi_delta": float(delta),
                "raroi_delta": float(raroi_delta),
                "confidence": float(confidence),
                "resilience": float(resilience),
                "scenario_degradation": float(scenario_deg),
            },
            compute_stability=True,
        )

    def attempt_promotion(self, workflow_id="wf"):
        risk = self.foresight_tracker.predict_roi_collapse(workflow_id)
        if risk.get("risk") == "Immediate collapse risk" or risk.get(
            "brittle"
        ):
            self.workflow_ready = False
        else:
            self.workflow_ready = True


def test_run_cycle_records_and_stability():
    ft = ForesightTracker(max_cycles=3, volatility_threshold=5.0)
    tracker = DummyROITracker([1.0, 2.0, 3.0, 0.0])
    eng = MiniSelfImprovementEngine(tracker, ft)

    for _ in range(3):
        eng.run_cycle()
    # initial positive trend
    assert ft.is_stable("wf")
    assert all("stability" in entry for entry in ft.history["wf"])

    eng.run_cycle()  # negative slope but low volatility
    history = ft.history["wf"]
    assert len(history) == 3
    assert [entry["roi_delta"] for entry in history] == [2.0, 3.0, 0.0]
    assert [entry["raroi_delta"] for entry in history] == [1.0, 1.5, 0.0]
    assert not ft.is_stable("wf")


def test_is_stable_reacts_to_high_volatility():
    ft = ForesightTracker(max_cycles=3, volatility_threshold=0.5)
    tracker = DummyROITracker([1.0, 5.0, 9.0])
    eng = MiniSelfImprovementEngine(tracker, ft)

    for _ in range(3):
        eng.run_cycle()
    assert not ft.is_stable("wf")


def test_metrics_persist_through_save_load(tmp_path):
    ft = ForesightTracker(max_cycles=3)
    tracker1 = DummyROITracker([1.0])
    eng1 = MiniSelfImprovementEngine(tracker1, ft)
    eng1.run_cycle()
    history_file = tmp_path / "foresight_history.json"
    with history_file.open("w", encoding="utf-8") as fh:
        json.dump(ft.to_dict(), fh, indent=2)

    with history_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    restored = ForesightTracker.from_dict(data)
    tracker2 = DummyROITracker([2.0])
    eng2 = MiniSelfImprovementEngine(tracker2, restored)
    eng2.run_cycle()
    with history_file.open("w", encoding="utf-8") as fh:
        json.dump(restored.to_dict(), fh, indent=2)

    with history_file.open("r", encoding="utf-8") as fh:
        final = json.load(fh)
    assert [e["roi_delta"] for e in final["history"]["wf"]] == [1.0, 2.0]


def test_promotion_blocked_by_risk_or_brittleness():
    ft = ForesightTracker()
    tracker = DummyROITracker([1.0])
    eng = MiniSelfImprovementEngine(tracker, ft)

    # Immediate collapse risk should block promotion
    ft.predict_roi_collapse = lambda wf: {
        "risk": "Immediate collapse risk",
        "brittle": False,
    }
    eng.attempt_promotion()
    assert not eng.workflow_ready

    # Brittleness alone should also block promotion
    ft.predict_roi_collapse = lambda wf: {"risk": "Stable", "brittle": True}
    eng.attempt_promotion()
    assert not eng.workflow_ready


def test_risky_workflow_not_promoted():
    ft = ForesightTracker()
    tracker = DummyROITracker([1.0, 0.0, -2.0])
    eng = MiniSelfImprovementEngine(tracker, ft)

    for _ in range(3):
        eng.run_cycle()

    info = ft.predict_roi_collapse("wf")
    assert info["risk"] == "Immediate collapse risk"

    eng.attempt_promotion()
    assert not eng.workflow_ready


def test_background_self_improvement_loop(monkeypatch):
    events: list[str] = []

    class DummyROI:
        def __init__(self):
            self.logged = []

        def log_result(self, **kw):
            self.logged.append(kw)
            events.append("roi")

    class DummyStability:
        def __init__(self):
            self.recorded = []

        def record_metrics(self, wf, roi, failures, entropy, roi_delta=None):
            self.recorded.append((wf, roi, entropy))
            events.append("stability")

    class DummyPlanner:
        def __init__(self):
            self.roi_db = DummyROI()
            self.stability_db = DummyStability()
            self.cluster_map: dict[tuple[str, ...], dict[str, object]] = {}

        def discover_and_persist(self, workflows, metrics_db=None):
            self.cluster_map[("a", "b")] = {"converged": False}
            return [{"chain": ["a", "b"], "roi_gain": 0.1, "failures": 0, "entropy": 0.0}]

        def mutate_pipeline(self, chain, workflows, **kwargs):
            events.append("mutate")
            return []

        def split_pipeline(self, chain, workflows, **kwargs):
            events.append("split")
            self.cluster_map[("a",)] = {"converged": True}
            self.cluster_map[("b",)] = {"converged": True}
            return [
                {"chain": ["a"], "roi_gain": 0.2, "failures": 0, "entropy": 0.0},
                {"chain": ["b"], "roi_gain": 0.2, "failures": 0, "entropy": 0.0},
            ]

        def remerge_pipelines(self, pipelines, workflows, **kwargs):
            events.append("remerge")
            self.cluster_map[("a", "b")] = {"converged": True}
            return [
                {"chain": ["a", "b"], "roi_gain": 0.5, "failures": 0, "entropy": 0.0}
            ]

    import sys, types

    dummy_mod = types.ModuleType("run_autonomous")
    dummy_mod.LOCAL_KNOWLEDGE_MODULE = None
    monkeypatch.setitem(sys.modules, "run_autonomous", dummy_mod)
    monkeypatch.setitem(sys.modules, "menace_sandbox.run_autonomous", dummy_mod)

    sandbox_pkg = types.ModuleType("sandbox_runner")
    env_mod = types.ModuleType("environment")
    orphan_mod = types.ModuleType("orphan_integration")
    orphan_mod.integrate_orphans = lambda *a, **k: None
    orphan_mod.post_round_orphan_scan = lambda *a, **k: None
    sandbox_pkg.environment = env_mod
    sandbox_pkg.orphan_integration = orphan_mod
    monkeypatch.setitem(sys.modules, "sandbox_runner", sandbox_pkg)
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)
    monkeypatch.setitem(sys.modules, "sandbox_runner.orphan_integration", orphan_mod)

    orphan_disc = types.ModuleType("orphan_discovery")
    orphan_disc.append_orphan_cache = lambda *a, **k: None
    orphan_disc.append_orphan_classifications = lambda *a, **k: None
    orphan_disc.prune_orphan_cache = lambda *a, **k: None
    orphan_disc.load_orphan_cache = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "orphan_discovery", orphan_disc)

    neuro = types.ModuleType("neurosales")
    neuro.add_message = lambda *a, **k: None
    neuro.get_recent_messages = lambda *a, **k: []
    neuro.push_chain = lambda *a, **k: None
    neuro.peek_chain = lambda *a, **k: []
    class _Dummy: ...
    neuro.MessageEntry = _Dummy
    neuro.CTAChain = _Dummy
    monkeypatch.setitem(sys.modules, "neurosales", neuro)

    light = types.ModuleType("light_bootstrap")
    monkeypatch.setitem(sys.modules, "light_bootstrap", light)
    env_boot = types.ModuleType("environment_bootstrap")
    monkeypatch.setitem(sys.modules, "environment_bootstrap", env_boot)
    embed_sched = types.ModuleType("vector_service.embedding_scheduler")
    monkeypatch.setitem(sys.modules, "vector_service.embedding_scheduler", embed_sched)
    unified = types.ModuleType("unified_event_bus")
    unified.AutomatedReviewer = object
    unified.UnifiedEventBus = object
    monkeypatch.setitem(sys.modules, "unified_event_bus", unified)
    auto_rev = types.ModuleType("automated_reviewer")
    auto_rev.AutomatedReviewer = object
    monkeypatch.setitem(sys.modules, "automated_reviewer", auto_rev)

    js = types.ModuleType("jsonschema")
    class _VE(Exception):
        pass
    js.ValidationError = _VE
    js.validate = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "jsonschema", js)

    from menace_sandbox import self_improvement_engine as sie  # delayed import

    planner = DummyPlanner()
    monkeypatch.setattr(sie, "MetaWorkflowPlanner", lambda: planner)

    async def run():
        task = asyncio.create_task(
            sie.self_improvement_cycle({"a": lambda: None, "b": lambda: None}, interval=0.01)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())

    assert {"mutate", "split", "remerge"}.issubset(events)
    assert planner.roi_db.logged
    assert planner.stability_db.recorded
