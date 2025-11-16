import json

from menace_sandbox.foresight_tracker import ForesightTracker
import asyncio
from collections import deque
from statistics import mean
import pytest
from pathlib import Path
import types
import shutil
import tempfile
import subprocess
import logging


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
    def __init__(self, tracker, foresight_tracker, window: int = 3):
        self.tracker = tracker
        self.foresight_tracker = foresight_tracker
        self.workflow_ready = False
        self.baseline: deque[float] = deque(maxlen=window)
        self.urgency_tier = 0

    def run_cycle(self, workflow_id="wf"):
        delta = self.tracker.next_delta()
        avg = mean(self.baseline) if self.baseline else 0.0
        if self.baseline and delta <= avg:
            self.urgency_tier += 1
        self.baseline.append(delta)
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
        if risk.get("risk") == "Immediate collapse risk" or risk.get("brittle"):
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
    assert eng.urgency_tier == 0

    eng.run_cycle()  # negative slope but low volatility
    history = ft.history["wf"]
    assert len(history) == 3
    assert [entry["roi_delta"] for entry in history] == [2.0, 3.0, 0.0]
    assert [entry["raroi_delta"] for entry in history] == [1.0, 1.5, 0.0]
    assert list(eng.baseline) == [2.0, 3.0, 0.0]
    assert eng.urgency_tier == 1
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
    assert eng.urgency_tier == 2
    assert list(eng.baseline) == [1.0, 0.0, -2.0]


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

    import sys
    import types

    dummy_mod = types.ModuleType("run_autonomous")
    dummy_mod.LOCAL_KNOWLEDGE_MODULE = None
    monkeypatch.setitem(sys.modules, "run_autonomous", dummy_mod)
    monkeypatch.setitem(sys.modules, "menace_sandbox.run_autonomous", dummy_mod)

    sandbox_pkg = types.ModuleType("sandbox_runner")
    env_mod = types.ModuleType("environment")
    orphan_mod = types.ModuleType("orphan_integration")
    orphan_mod.integrate_orphans = lambda *a, **k: None
    orphan_mod.post_round_orphan_scan = lambda *a, **k: None
    bootstrap_mod = types.ModuleType("bootstrap")
    bootstrap_mod.initialize_autonomous_sandbox = lambda *a, **k: None
    sandbox_pkg.environment = env_mod
    sandbox_pkg.orphan_integration = orphan_mod
    sandbox_pkg.bootstrap = bootstrap_mod
    monkeypatch.setitem(sys.modules, "sandbox_runner", sandbox_pkg)
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)
    monkeypatch.setitem(sys.modules, "sandbox_runner.orphan_integration", orphan_mod)
    monkeypatch.setitem(sys.modules, "sandbox_runner.bootstrap", bootstrap_mod)

    quick_fix = types.ModuleType("quick_fix_engine")
    monkeypatch.setitem(sys.modules, "quick_fix_engine", quick_fix)

    cws_mod = types.ModuleType("composite_workflow_scorer")
    cws_mod.CompositeWorkflowScorer = object
    monkeypatch.setitem(sys.modules, "composite_workflow_scorer", cws_mod)
    monkeypatch.setitem(
        sys.modules, "menace_sandbox.composite_workflow_scorer", cws_mod
    )
    wem = types.ModuleType("workflow_evolution_manager")
    wem.WorkflowEvolutionManager = object
    monkeypatch.setitem(sys.modules, "workflow_evolution_manager", wem)
    monkeypatch.setitem(
        sys.modules, "menace_sandbox.workflow_evolution_manager", wem
    )

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

    class _Dummy:
        ...
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

    data_bot = types.ModuleType("data_bot")
    data_bot.MetricsDB = object
    data_bot.DataBot = object
    data_bot.ErrorDB = object
    data_bot.ErrorLogger = object
    data_bot.KnowledgeGraph = object
    data_bot.MetricRecord = object
    monkeypatch.setitem(sys.modules, "data_bot", data_bot)
    monkeypatch.setitem(sys.modules, "menace_sandbox.data_bot", data_bot)

    sts_mod = types.ModuleType("self_test_service")
    sts_mod.SelfTestService = object
    monkeypatch.setitem(sys.modules, "self_test_service", sts_mod)
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_test_service", sts_mod)

    log_utils = types.ModuleType("logging_utils")
    log_utils.log_record = lambda **kw: kw
    log_utils.get_logger = lambda name: types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    log_utils.setup_logging = lambda: None
    log_utils.set_correlation_id = lambda _: None
    log_utils.LockedRotatingFileHandler = object
    log_utils.LockedTimedRotatingFileHandler = object
    monkeypatch.setitem(sys.modules, "logging_utils", log_utils)
    monkeypatch.setitem(sys.modules, "menace_sandbox.logging_utils", log_utils)

    js = types.ModuleType("jsonschema")

    class _VE(Exception):
        pass
    js.ValidationError = _VE
    js.validate = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "jsonschema", js)
    violation_logger = types.ModuleType("violation_logger")
    violation_logger.log_violation = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "violation_logger", violation_logger)

    info_bot = types.ModuleType("menace.information_synthesis_bot")
    info_bot.SynthesisTask = type("SynthesisTask", (), {})
    monkeypatch.setitem(sys.modules, "menace.information_synthesis_bot", info_bot)

    from menace_sandbox import self_improvement as sie  # delayed import
    monkeypatch.setattr(sie.init, "verify_dependencies", lambda auto_install=False: None)
    sie.init_self_improvement()

    planner = DummyPlanner()
    monkeypatch.setattr(sie.meta_planning, "MetaWorkflowPlanner", lambda: planner)

    async def run():
        task = asyncio.create_task(
            sie.self_improvement_cycle({"a": lambda: None, "b": lambda: None}, interval=0.01)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())

    assert {"roi", "stability"}.issubset(events)
    assert planner.roi_db.logged
    assert planner.stability_db.recorded


def test_target_region_escalation(monkeypatch, tmp_path, caplog):
    import sys
    stub_env = types.ModuleType("menace_sandbox.environment_bootstrap")
    stub_env.EnvironmentBootstrapper = object
    monkeypatch.setitem(sys.modules, "menace_sandbox.environment_bootstrap", stub_env)

    db_stub = types.ModuleType("menace_sandbox.data_bot")
    db_stub.MetricsDB = object
    db_stub.DataBot = object
    monkeypatch.setitem(sys.modules, "menace_sandbox.data_bot", db_stub)
    monkeypatch.setitem(sys.modules, "data_bot", db_stub)

    sce_stub = types.ModuleType("menace_sandbox.self_coding_engine")
    sce_stub.SelfCodingEngine = object
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_engine", sce_stub)

    mapl_stub = types.ModuleType("menace_sandbox.model_automation_pipeline")
    class AutomationResult:
        def __init__(self, package=None, roi=None):
            self.package = package
            self.roi = roi

    class ModelAutomationPipeline: ...

    mapl_stub.AutomationResult = AutomationResult
    mapl_stub.ModelAutomationPipeline = ModelAutomationPipeline
    monkeypatch.setitem(sys.modules, "menace_sandbox.model_automation_pipeline", mapl_stub)

    prb_stub = types.ModuleType("menace_sandbox.pre_execution_roi_bot")
    class ROIResult:
        def __init__(self, roi, errors, proi, perr, risk):
            self.roi = roi
            self.errors = errors
            self.predicted_roi = proi
            self.predicted_errors = perr
            self.risk = risk

    prb_stub.ROIResult = ROIResult
    monkeypatch.setitem(sys.modules, "menace_sandbox.pre_execution_roi_bot", prb_stub)

    monkeypatch.setitem(sys.modules, "menace", types.SimpleNamespace(RAISE_ERRORS=False))

    th_stub = types.ModuleType("menace_sandbox.sandbox_runner.test_harness")
    class DummyHarnessResult:
        def __init__(self, success, failure=None, stdout="", stderr="", duration=0.0):
            self.success = success
            self.failure = failure
            self.stdout = stdout
            self.stderr = stderr
            self.duration = duration

    th_stub.run_tests = lambda repo, changed, backend="venv": DummyHarnessResult(True)
    th_stub.TestHarnessResult = DummyHarnessResult
    sandbox_stub = types.ModuleType("menace_sandbox.sandbox_runner")
    sandbox_stub.test_harness = th_stub
    monkeypatch.setitem(sys.modules, "menace_sandbox.sandbox_runner", sandbox_stub)
    monkeypatch.setitem(sys.modules, "menace_sandbox.sandbox_runner.test_harness", th_stub)

    import menace_sandbox.self_coding_manager as scm
    class DummyBuilder:
        def query(self, desc, exclude_tags=None):
            return [], "sid"

    class DummyCognitionLayer:
        def __init__(self):
            self.context_builder = DummyBuilder()

        def record_patch_outcome(self, session_id, success, contribution=0.0):
            pass

    class DummyEngine:
        def __init__(self):
            self.regions = []
            self.cognition_layer = DummyCognitionLayer()
            self.last_prompt_text = ""

        def apply_patch(self, path: Path, desc: str, **kw):
            self.regions.append(kw.get("target_region"))
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("# patched\n")
            return len(self.regions), False, 0.0

    class DummyPipeline:
        def run(self, model, energy=1):
            return types.SimpleNamespace(roi=types.SimpleNamespace(confidence=1.0))

    engine = DummyEngine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(engine, pipeline, bot_name="bot")

    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("def foo():\n    return 1\n")

    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    outputs = [
        types.SimpleNamespace(
            success=False,
            failure={"strategy_tag": "x", "stack": "trace"},
            stdout="",
            stderr="",
            duration=0.0,
        )
        for _ in range(4)
    ]
    outputs.append(
        types.SimpleNamespace(success=True, failure=None, stdout="", stderr="", duration=0.0)
    )

    def run_tests_stub(repo, changed, backend="venv"):
        return outputs.pop(0)

    monkeypatch.setattr(scm, "run_tests", run_tests_stub)

    tr = scm.TargetRegion(file=file_path.name, start_line=1, end_line=2, function="foo")

    def parse_stub(trace):
        return {"trace": trace, "target_region": tr}

    monkeypatch.setattr(scm.ErrorParser, "parse", staticmethod(parse_stub))

    caplog.set_level(logging.INFO)
    mgr.run_patch(file_path, "desc", max_attempts=5)

    regions = engine.regions
    assert len(regions) == 5
    assert regions[0].start_line == 1 and regions[0].function == "foo"
    assert regions[2].start_line == 0 and regions[2].function == "foo"
    assert regions[-1].function == ""
    assert any("function scope" in r.message for r in caplog.records)
    assert any("module scope" in r.message for r in caplog.records)
