import argparse
import sys
import types
from pathlib import Path
import os
import asyncio
import json
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")


def _stub_module(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    pkg, _, sub = name.partition(".")
    pkg_mod = sys.modules.get(pkg)
    if pkg_mod and sub:
        setattr(pkg_mod, sub, mod)
    return mod


def _patch_adfuller(monkeypatch, pvalue: float) -> None:
    mod = types.ModuleType("statsmodels.tsa.stattools")

    def adfuller(vals, *a, **k):
        return (0.0, pvalue, 0, len(vals), {}, 0.0)

    mod.adfuller = adfuller
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.stattools", mod)
    tsa = types.ModuleType("tsa")
    tsa.stattools = mod
    monkeypatch.setitem(sys.modules, "statsmodels.tsa", tsa)
    root = types.ModuleType("statsmodels")
    root.tsa = tsa
    monkeypatch.setitem(sys.modules, "statsmodels", root)


def _patch_levene(monkeypatch, pvalue: float) -> None:
    stats_mod = types.ModuleType("scipy.stats")

    def levene(*a, **k):
        return types.SimpleNamespace(pvalue=pvalue)

    stats_mod.levene = levene
    stats_mod.pearsonr = lambda *a, **k: (0.0, 0.0)
    stats_mod.t = types.SimpleNamespace(cdf=lambda *a, **k: 0.5)
    monkeypatch.setitem(sys.modules, "scipy.stats", stats_mod)
    root = types.ModuleType("scipy")
    root.stats = stats_mod
    monkeypatch.setitem(sys.modules, "scipy", root)


class DummyBot:
    def __init__(self, *a, **k):
        pass


class DummyPolicy:
    def __init__(self, *a, **k):
        pass

    def save(self):
        pass


class DummyDataBot:
    def __init__(self, *a, **k):
        pass

    def collect(self, *a, **k):
        pass


@pytest.fixture(autouse=True)
def _patch_suggestion(monkeypatch):
    _stub_module(monkeypatch, "menace.patch_suggestion_db", PatchSuggestionDB=DummyBot)
    yield


class DummyTracker:
    instance = None

    def __init__(self, *a, **k):
        DummyTracker.instance = self
        self.records = []
        self.module_deltas = {}
        self.scenario_synergy = {}

    def update(self, prev, curr, modules=None, resources=None, metrics=None):
        self.records.append((modules, metrics))
        return 0.0, [], False

    def forecast(self):
        return 0.0, (0.0, 0.0)

    def forecast_metric(self, name):
        return 0.0, (0.0, 0.0)

    def diminishing(self):
        return 0.0

    def record_prediction(self, predicted, actual):
        pass

    def record_metric_prediction(self, metric, predicted, actual):
        pass

    def predict_all_metrics(self, manager, features):
        pass

    def rolling_mae(self, window=None):
        return 0.0

    def reliability(self):
        return 0.0

    def load_history(self, path):
        pass

    def save_history(self, path):
        pass

    def get_scenario_synergy(self, name):
        return self.scenario_synergy.get(name, [])


class DummyGPT:
    calls = []

    def __init__(self, *a, **k):
        pass

    def ask(self, msgs):
        self.calls.append(msgs)
        return {"choices": [{"message": {"content": "patch"}}]}


class DummyMetaLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.module_deltas = {}
        self.last_patch_id = 0

    def log_cycle(self, *a, **k):
        pass

    def rankings(self):
        return []

    def diminishing(self, threshold=None):
        return ["mod.py"]


class DummyEngine:
    def __init__(self, *a, **k):
        pass

    def apply_patch(self, path, suggestion):
        return 1, False, 0.0

    def rollback_patch(self, patch_id):
        pass


class DummyImprover:
    def run_cycle(self):
        class R:
            roi = types.SimpleNamespace(roi=0.0)

        return R()

    def _policy_state(self):
        return ()


class DummySandbox:
    def __init__(self, *a, **k):
        pass

    def analyse_and_fix(self):
        pass


class DummyTester:
    def __init__(self, *a, **k):
        pass

    def _run_once(self):
        pass


class DummyOrch:
    def create_oversight(self, *a, **k):
        pass

    def run_cycle(self, *a, **k):
        class R:
            roi = None

        return R()


class DummyBus:
    def __init__(self, persist_path=None, **kw):
        pass

    def close(self):
        pass


def test_repo_section_metrics(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBot)
    _stub_module(monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyPolicy
    )
    _stub_module(
        monkeypatch, "menace.self_improvement_engine", SelfImprovementEngine=DummyBot
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(
        monkeypatch, "menace.code_database", PatchHistoryDB=DummyBot, CodeDB=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.pre_execution_roi_bot", PreExecutionROIBot=DummyBot
    )
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)
    mod = types.ModuleType("menace.data_bot")
    mod.DataBot = DummyDataBot
    mod.MetricsDB = DummyBot
    mod.MetricsDB = DummyBot
    monkeypatch.setitem(sys.modules, "menace.data_bot", mod)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(Path(__file__).resolve().parents[1] / "sandbox_runner.py"),
        submodule_search_locations=[
            str(Path(__file__).resolve().parents[1] / "sandbox_runner")
        ],
    )
    sandbox_runner = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sandbox_runner
    spec.loader.exec_module(sandbox_runner)
    monkeypatch.setattr(
        sandbox_runner,
        "scan_repo_sections",
        lambda p: {"m.py": {"sec": ["pass"]}},
        raising=False,
    )

    code = "def a():\n    pass\n\ndef b():\n    pass\n"
    (tmp_path / "m.py").write_text(code)

    calls = []

    def fake_sim(code_str, stub=None):
        calls.append((code_str, stub))
        return {"risk_flags_triggered": ["x"]}

    import sandbox_runner.environment as env

    monkeypatch.setattr(env, "simulate_execution_environment", fake_sim)

    tracker = sandbox_runner.run_repo_section_simulations(
        str(tmp_path), input_stubs=[{"k": 1}], env_presets=[{"env": "prod"}]
    )

    assert len(calls) >= 3
    assert all(isinstance(m, dict) for _, m in tracker.records)


def test_gpt_trigger_on_diminishing(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "1")
    monkeypatch.setenv("SANDBOX_CYCLES", "1")
    monkeypatch.setenv("SANDBOX_ROI_TOLERANCE", "0")

    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyEngine)
    _stub_module(
        monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyPolicy
    )
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.pre_execution_roi_bot", PreExecutionROIBot=DummyBot
    )
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)
    mod = types.ModuleType("menace.data_bot")
    mod.DataBot = DummyDataBot
    mod.MetricsDB = DummyBot
    monkeypatch.setitem(sys.modules, "menace.data_bot", mod)
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBus)
    _stub_module(
        monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyOrch
    )
    _stub_module(
        monkeypatch,
        "menace.self_improvement_engine",
        SelfImprovementEngine=lambda *a, **k: DummyImprover(),
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyTester)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )
    _stub_module(monkeypatch, "menace.chatgpt_idea_bot", ChatGPTClient=DummyGPT)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(Path(__file__).resolve().parents[1] / "sandbox_runner.py"),
        submodule_search_locations=[
            str(Path(__file__).resolve().parents[1] / "sandbox_runner")
        ],
    )
    sandbox_runner = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sandbox_runner
    spec.loader.exec_module(sandbox_runner)

    monkeypatch.setattr(sandbox_runner, "MenaceOrchestrator", DummyOrch)
    monkeypatch.setattr(sandbox_runner, "_SandboxMetaLogger", DummyMetaLogger)

    sandbox_runner._run_sandbox(argparse.Namespace(sandbox_data_dir=str(tmp_path)))

    assert DummyGPT.calls


def test_section_loop_gpt_trigger(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "1")
    monkeypatch.setenv("SANDBOX_CYCLES", "1")
    monkeypatch.setenv("SANDBOX_ROI_TOLERANCE", "0")

    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyEngine)
    _stub_module(
        monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyPolicy
    )
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)
    mod = types.ModuleType("menace.data_bot")
    mod.DataBot = DummyDataBot
    mod.MetricsDB = DummyBot
    monkeypatch.setitem(sys.modules, "menace.data_bot", mod)
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBus)
    _stub_module(
        monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyOrch
    )
    _stub_module(
        monkeypatch,
        "menace.self_improvement_engine",
        SelfImprovementEngine=lambda *a, **k: DummyImprover(),
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyTester)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )
    _stub_module(monkeypatch, "menace.chatgpt_idea_bot", ChatGPTClient=DummyGPT)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(Path(__file__).resolve().parents[1] / "sandbox_runner.py"),
        submodule_search_locations=[
            str(Path(__file__).resolve().parents[1] / "sandbox_runner")
        ],
    )
    sandbox_runner = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sandbox_runner
    spec.loader.exec_module(sandbox_runner)

    monkeypatch.setattr(sandbox_runner, "MenaceOrchestrator", DummyOrch)
    monkeypatch.setattr(sandbox_runner, "_SandboxMetaLogger", DummyMetaLogger)
    monkeypatch.setattr(
        sandbox_runner,
        "scan_repo_sections",
        lambda path: {"mod.py": {"sec": ["pass"]}},
    )

    sandbox_runner._run_sandbox(argparse.Namespace(sandbox_data_dir=str(tmp_path)))

    assert DummyGPT.calls
    DummyGPT.calls.clear()


def test_metrics_db_records(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("SANDBOX_CYCLES", "1")

    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyEngine)
    _stub_module(
        monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyPolicy
    )
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)
    mod = types.ModuleType("menace.data_bot")
    mod.DataBot = DummyDataBot
    mod.MetricsDB = DummyBot
    mod.MetricsDB = DummyBot
    monkeypatch.setitem(sys.modules, "menace.data_bot", mod)
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBus)
    _stub_module(
        monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyOrch
    )
    _stub_module(
        monkeypatch,
        "menace.self_improvement_engine",
        SelfImprovementEngine=lambda *a, **k: DummyImprover(),
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyTester)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(Path(__file__).resolve().parents[1] / "sandbox_runner.py"),
        submodule_search_locations=[
            str(Path(__file__).resolve().parents[1] / "sandbox_runner")
        ],
    )
    sandbox_runner = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sandbox_runner
    spec.loader.exec_module(sandbox_runner)

    monkeypatch.setattr(sandbox_runner, "MenaceOrchestrator", DummyOrch)
    monkeypatch.setattr(sandbox_runner, "_SandboxMetaLogger", DummyMetaLogger)
    monkeypatch.setattr(
        sandbox_runner,
        "scan_repo_sections",
        lambda path: {"mod.py": {"sec": ["pass"]}},
    )

    sandbox_runner._run_sandbox(argparse.Namespace(sandbox_data_dir=str(tmp_path)))

    db = sandbox_runner.MetricsDB(tmp_path / "metrics.db")
    rows = db.fetch(limit=10)
    assert len(rows) > 0


class _PredTracker:
    def __init__(self, *a, **k):
        type(self).instance = self
        self.predicted_metrics = {}
        self.metrics_history = {"security_score": [0.0]}
        self.pred_calls = 0

    def update(self, *a, **k):
        return 0, [], False

    def forecast(self):
        return 0.0, (0.0, 0.0)

    def forecast_metric(self, name):
        return 0.0, (0.0, 0.0)

    def diminishing(self):
        return 0.0

    def predict_all_metrics(self, manager, features, bot_name=None):
        self.pred_calls += 1
        self.predicted_metrics.setdefault("security_score", []).append(1.0)
        return {"security_score": 1.0}

    def record_metric_prediction(self, metric, predicted, actual):
        self.predicted_metrics.setdefault(metric, []).append(predicted)


def test_metric_predictions_recorded(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("SANDBOX_CYCLES", "1")

    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=_PredTracker)
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyEngine)
    _stub_module(
        monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyPolicy
    )
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    mod = types.ModuleType("menace.data_bot")
    mod.DataBot = DummyDataBot
    mod.MetricsDB = DummyBot
    mod.MetricsDB = DummyBot
    monkeypatch.setitem(sys.modules, "menace.data_bot", mod)
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBus)
    _stub_module(
        monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyOrch
    )
    _stub_module(
        monkeypatch,
        "menace.self_improvement_engine",
        SelfImprovementEngine=lambda *a, **k: DummyImprover(),
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyTester)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )

    class DummyPreBot:
        instance = None

        def __init__(self, *a, **k):
            type(self).instance = self

    _stub_module(
        monkeypatch, "menace.pre_execution_roi_bot", PreExecutionROIBot=DummyPreBot
    )

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(Path(__file__).resolve().parents[1] / "sandbox_runner.py"),
        submodule_search_locations=[
            str(Path(__file__).resolve().parents[1] / "sandbox_runner")
        ],
    )
    sandbox_runner = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sandbox_runner
    spec.loader.exec_module(sandbox_runner)

    monkeypatch.setattr(sandbox_runner, "MenaceOrchestrator", DummyOrch)
    monkeypatch.setattr(sandbox_runner, "_SandboxMetaLogger", DummyMetaLogger)
    monkeypatch.setattr(
        sandbox_runner,
        "scan_repo_sections",
        lambda path: {"mod.py": {"sec": ["pass"]}},
    )

    ctx = sandbox_runner._sandbox_init(
        {}, argparse.Namespace(sandbox_data_dir=str(tmp_path))
    )
    sandbox_runner._sandbox_cycle_runner(ctx, None, None, ctx.tracker)
    sandbox_runner._sandbox_cleanup(ctx)
    tracker = ctx.tracker

    assert isinstance(tracker, _PredTracker)
    assert DummyPreBot.instance is not None
    assert getattr(DummyPreBot.instance, "prediction_manager", None) is not None
    assert tracker.pred_calls > 0
    assert tracker.predicted_metrics


def test_section_worker_netem(monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBus)
    _stub_module(monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyPolicy
    )
    _stub_module(
        monkeypatch, "menace.self_improvement_engine", SelfImprovementEngine=DummyBot
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyEngine)
    _stub_module(
        monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.pre_execution_roi_bot", PreExecutionROIBot=DummyBot
    )
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)
    mod = types.ModuleType("menace.data_bot")
    mod.DataBot = DummyDataBot
    mod.MetricsDB = DummyBot
    mod.MetricsDB = DummyBot
    monkeypatch.setitem(sys.modules, "menace.data_bot", mod)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(Path(__file__).resolve().parents[1] / "sandbox_runner.py"),
        submodule_search_locations=[
            str(Path(__file__).resolve().parents[1] / "sandbox_runner")
        ],
    )
    sandbox_runner = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sandbox_runner
    spec.loader.exec_module(sandbox_runner)

    class DummyIPRoute:
        calls = []

        def __init__(self):
            pass

        def bind(self, netns=None):
            DummyIPRoute.calls.append(("bind", netns))

        def link_lookup(self, ifname):
            return [1]

        def link(self, *a, **kw):
            DummyIPRoute.calls.append(("link", a, kw))

        def tc(self, action, parent, index, kind, **kw):
            DummyIPRoute.calls.append((action, kind, kw))

        def close(self):
            DummyIPRoute.calls.append(("close",))

    class DummyNetns:
        calls = []

        @staticmethod
        def create(name):
            DummyNetns.calls.append(("create", name))

        @staticmethod
        def remove(name):
            DummyNetns.calls.append(("remove", name))

    dummy_popen_calls = []

    def dummy_nspopen(ns, args, **kw):
        dummy_popen_calls.append((ns, args))
        return subprocess.Popen(args, **kw)

    monkeypatch.setattr(sandbox_runner.environment, "IPRoute", DummyIPRoute)
    monkeypatch.setattr(sandbox_runner.environment, "netns", DummyNetns)
    monkeypatch.setattr(sandbox_runner.environment, "NSPopen", dummy_nspopen)

    res, updates = asyncio.run(
        sandbox_runner._section_worker(
            "print('x')",
            {"NETWORK_LATENCY_MS": "50", "PACKET_LOSS": "1"},
            0.0,
        )
    )
    assert any(c[0] == "add" for c in DummyIPRoute.calls)
    assert any(c[0] == "del" for c in DummyIPRoute.calls)
    assert updates[-1][2]["netem_latency_ms"] == 50.0
    assert updates[-1][2]["netem_packet_loss"] == 1.0


def test_section_worker_netem_no_tc(monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBus)
    _stub_module(monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyPolicy
    )
    _stub_module(
        monkeypatch, "menace.self_improvement_engine", SelfImprovementEngine=DummyBot
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyEngine)
    _stub_module(
        monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.pre_execution_roi_bot", PreExecutionROIBot=DummyBot
    )
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)
    mod = types.ModuleType("menace.data_bot")
    mod.DataBot = DummyDataBot
    mod.MetricsDB = DummyBot
    mod.MetricsDB = DummyBot
    monkeypatch.setitem(sys.modules, "menace.data_bot", mod)

    import sandbox_runner

    called = {}

    async def fake_exec(code, env, **kw):
        called["code"] = code
        return {"exit_code": 0.0}

    monkeypatch.setattr(sandbox_runner.environment, "_execute_in_container", fake_exec)
    monkeypatch.setattr(sandbox_runner.environment, "IPRoute", None)
    monkeypatch.setattr(sandbox_runner.environment, "NSPopen", None)
    monkeypatch.setattr(sandbox_runner.environment, "netns", None)

    res, updates = asyncio.run(
        sandbox_runner._section_worker(
            "print('x')",
            {"NETWORK_LATENCY_MS": "50"},
            0.0,
        )
    )
    assert "pyroute2" in called.get("code", "")


def test_auto_prompt_selection(monkeypatch):
    import importlib

    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBus)
    _stub_module(monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyPolicy
    )
    _stub_module(
        monkeypatch, "menace.self_improvement_engine", SelfImprovementEngine=DummyBot
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyEngine)
    _stub_module(
        monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.pre_execution_roi_bot", PreExecutionROIBot=DummyBot
    )
    _stub_module(monkeypatch, "menace.patch_suggestion_db", PatchSuggestionDB=DummyBot)
    jinja_mod = types.ModuleType("jinja2")

    class DummyTemplate:
        def __init__(self, *a, **k):
            self.text = a[0] if a else ""

        def render(self, *a, **k):
            return self.text

    jinja_mod.Template = DummyTemplate
    monkeypatch.setitem(sys.modules, "jinja2", jinja_mod)
    mod = types.ModuleType("menace.data_bot")
    mod.DataBot = DummyDataBot
    mod.MetricsDB = DummyBot
    mod.MetricsDB = DummyBot
    monkeypatch.setitem(sys.modules, "menace.data_bot", mod)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(Path(__file__).resolve().parents[1] / "sandbox_runner.py"),
        submodule_search_locations=[
            str(Path(__file__).resolve().parents[1] / "sandbox_runner")
        ],
    )
    sandbox_runner = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sandbox_runner
    spec.loader.exec_module(sandbox_runner)

    class T:
        def __init__(self, sec_drop=False, eff_drop=False):
            base = [1.0, 0.9]
            self.roi_history = base
            self.module_deltas = {"a": [0.1, -0.1]}
            sec = [0.9, 0.7] if sec_drop else [0.9, 0.9]
            eff = [0.9, 0.5] if eff_drop else [0.9, 0.9]
            self.metrics_history = {"security_score": sec, "efficiency": eff}

        def diminishing(self):
            return 0.01

    importlib.reload(sandbox_runner)
    prompt = sandbox_runner.build_section_prompt("a", T(sec_drop=True))
    assert "SECURITY FOCUS" in prompt
    assert len(sandbox_runner._AUTO_TEMPLATES) >= 3
    cached = sandbox_runner._AUTO_TEMPLATES

    prompt = sandbox_runner.build_section_prompt("a", T(eff_drop=True))
    assert "EFFICIENCY FOCUS" in prompt
    assert sandbox_runner._AUTO_TEMPLATES is cached

    prompt = sandbox_runner.build_section_prompt("a", T())
    assert "ROI IMPROVEMENT" in prompt


def test_prompt_truncation_and_metrics(monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    os.environ.pop("GPT_SECTION_PROMPT_MAX_LENGTH", None)
    os.environ.pop("GPT_SECTION_SUMMARY_DEPTH", None)

    import importlib
    import importlib.util

    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBus)
    _stub_module(monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyPolicy
    )
    _stub_module(
        monkeypatch, "menace.self_improvement_engine", SelfImprovementEngine=DummyBot
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyEngine)
    _stub_module(
        monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.pre_execution_roi_bot", PreExecutionROIBot=DummyBot
    )
    mod = types.ModuleType("menace.data_bot")
    mod.DataBot = DummyDataBot
    mod.MetricsDB = DummyBot
    monkeypatch.setitem(sys.modules, "menace.data_bot", mod)

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(Path(__file__).resolve().parents[1] / "sandbox_runner.py"),
        submodule_search_locations=[
            str(Path(__file__).resolve().parents[1] / "sandbox_runner")
        ],
    )
    sandbox_runner = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sandbox_runner
    spec.loader.exec_module(sandbox_runner)

    class T:
        def __init__(self):
            self.roi_history = [1.0, 1.5, 0.9]
            self.metrics_history = {
                "security_score": [0.8, 0.9],
                "efficiency": [1.1, 1.2],
            }
            self.module_deltas = {"mod": [0.1, -0.1]}

        def diminishing(self):
            return 0.01

    snippet = "\n".join(f"line{i}" for i in range(20))
    prompt = sandbox_runner.build_section_prompt(
        "mod:sec",
        T(),
        snippet=snippet,
        max_length=50,
        summary_depth=1,
        max_prompt_length=200,
    )

    assert "ROI deltas" in prompt
    assert "efficiency=" in prompt
    assert "Metrics summary:" in prompt
    assert "# ..." in prompt


def test_prompt_synergy_and_length(monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    os.environ.pop("GPT_SECTION_PROMPT_MAX_LENGTH", None)
    os.environ.pop("GPT_SECTION_SUMMARY_DEPTH", None)

    import importlib
    import importlib.util

    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBus)
    _stub_module(monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyPolicy
    )
    _stub_module(
        monkeypatch, "menace.self_improvement_engine", SelfImprovementEngine=DummyBot
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyEngine)
    _stub_module(
        monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.pre_execution_roi_bot", PreExecutionROIBot=DummyBot
    )
    mod = types.ModuleType("menace.data_bot")
    mod.DataBot = DummyDataBot
    mod.MetricsDB = DummyBot
    monkeypatch.setitem(sys.modules, "menace.data_bot", mod)

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(Path(__file__).resolve().parents[1] / "sandbox_runner.py"),
        submodule_search_locations=[
            str(Path(__file__).resolve().parents[1] / "sandbox_runner")
        ],
    )
    sandbox_runner = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sandbox_runner
    spec.loader.exec_module(sandbox_runner)

    class T:
        def __init__(self):
            self.roi_history = [1.0, 1.2]
            self.metrics_history = {
                "security_score": [0.8, 0.9],
                "synergy_roi": [0.05],
                "synergy_security_score": [0.02],
            }
            self.module_deltas = {"mod": [0.1, -0.1]}

        def diminishing(self):
            return 0.01

    snippet = "\n".join(f"line{i}" for i in range(20))
    prompt = sandbox_runner.build_section_prompt(
        "mod:sec",
        T(),
        snippet=snippet,
        max_length=50,
        summary_depth=1,
        max_prompt_length=80,
    )

    assert "Synergy" in prompt
    assert "Synergy summary:" in prompt
    assert len(prompt) <= 80


def test_preset_adaptation(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("SANDBOX_CYCLES", "1")

    calls = {}

    def fake_adapt(tracker, presets):
        calls["tracker"] = tracker
        return [{"adapted": True}]

    _stub_module(monkeypatch, "menace.environment_generator", adapt_presets=fake_adapt)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBus)
    _stub_module(
        monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyOrch
    )
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyPolicy
    )
    _stub_module(
        monkeypatch,
        "menace.self_improvement_engine",
        SelfImprovementEngine=lambda *a, **k: DummyImprover(),
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyTester)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyEngine)
    _stub_module(
        monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.pre_execution_roi_bot", PreExecutionROIBot=DummyBot
    )
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)
    mod = types.ModuleType("menace.data_bot")
    mod.DataBot = DummyDataBot
    mod.MetricsDB = DummyBot
    monkeypatch.setitem(sys.modules, "menace.data_bot", mod)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(Path(__file__).resolve().parents[1] / "sandbox_runner.py"),
        submodule_search_locations=[
            str(Path(__file__).resolve().parents[1] / "sandbox_runner")
        ],
    )
    sandbox_runner = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sandbox_runner
    spec.loader.exec_module(sandbox_runner)

    monkeypatch.setattr(sandbox_runner, "MenaceOrchestrator", DummyOrch)
    monkeypatch.setattr(sandbox_runner, "_SandboxMetaLogger", DummyMetaLogger)
    monkeypatch.setattr(
        sandbox_runner,
        "scan_repo_sections",
        lambda path: {"mod.py": {"sec": ["pass"]}},
    )

    sandbox_runner.SANDBOX_ENV_PRESETS = [{"foo": "bar"}]
    ctx = sandbox_runner._sandbox_init(
        {}, argparse.Namespace(sandbox_data_dir=str(tmp_path))
    )
    sandbox_runner._sandbox_cycle_runner(ctx, "mod.py:sec", "pass", ctx.tracker)
    sandbox_runner._sandbox_cleanup(ctx)

    assert calls.get("tracker") is ctx.tracker
    assert sandbox_runner.SANDBOX_ENV_PRESETS == [{"adapted": True}]


def test_preset_persistence_across_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("SANDBOX_CYCLES", "1")

    count = {"val": 0}

    def fake_adapt(tracker, presets):
        count["val"] += 1
        cur = presets[0].get("num", 0)
        return [{"num": cur + 1}]

    _stub_module(monkeypatch, "menace.environment_generator", adapt_presets=fake_adapt)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBus)
    _stub_module(
        monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyOrch
    )
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyPolicy
    )
    _stub_module(
        monkeypatch,
        "menace.self_improvement_engine",
        SelfImprovementEngine=lambda *a, **k: DummyImprover(),
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyTester)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyEngine)
    _stub_module(
        monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.pre_execution_roi_bot", PreExecutionROIBot=DummyBot
    )
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)
    mod = types.ModuleType("menace.data_bot")
    mod.DataBot = DummyDataBot
    mod.MetricsDB = DummyBot
    monkeypatch.setitem(sys.modules, "menace.data_bot", mod)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(Path(__file__).resolve().parents[1] / "sandbox_runner.py"),
        submodule_search_locations=[
            str(Path(__file__).resolve().parents[1] / "sandbox_runner")
        ],
    )
    sandbox_runner = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sandbox_runner
    spec.loader.exec_module(sandbox_runner)

    monkeypatch.setattr(sandbox_runner, "MenaceOrchestrator", DummyOrch)
    monkeypatch.setattr(sandbox_runner, "_SandboxMetaLogger", DummyMetaLogger)
    monkeypatch.setattr(
        sandbox_runner,
        "scan_repo_sections",
        lambda path: {"mod.py": {"sec": ["pass"]}},
    )

    def fake_cycle(ctx, sec, snip, tracker, scenario=None):
        from menace.environment_generator import adapt_presets

        if getattr(ctx, "adapt_presets", True):
            sandbox_runner.SANDBOX_ENV_PRESETS = adapt_presets(
                tracker, sandbox_runner.SANDBOX_ENV_PRESETS
            )
            os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(
                sandbox_runner.SANDBOX_ENV_PRESETS
            )

    monkeypatch.setattr(sandbox_runner, "_sandbox_cycle_runner", fake_cycle)

    class DummyCtx:
        def __init__(self):
            self.adapt_presets = True

    sandbox_runner.SANDBOX_ENV_PRESETS = [{"num": 0}]
    tracker = DummyTracker()
    sandbox_runner._sandbox_cycle_runner(DummyCtx(), "mod.py:sec", "pass", tracker)
    assert sandbox_runner.SANDBOX_ENV_PRESETS == [{"num": 1}]

    tracker2 = DummyTracker()
    sandbox_runner._sandbox_cycle_runner(DummyCtx(), "mod.py:sec", "pass", tracker2)
    assert sandbox_runner.SANDBOX_ENV_PRESETS == [{"num": 2}]

    assert count["val"] == 2


def test_no_preset_adapt_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("SANDBOX_CYCLES", "1")

    def fake_adapt(tracker, presets):
        return [{"num": 1}]

    _stub_module(monkeypatch, "menace.environment_generator", adapt_presets=fake_adapt)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBus)
    _stub_module(
        monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyOrch
    )
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyPolicy
    )
    _stub_module(
        monkeypatch,
        "menace.self_improvement_engine",
        SelfImprovementEngine=lambda *a, **k: DummyImprover(),
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyTester)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyEngine)
    _stub_module(
        monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.pre_execution_roi_bot", PreExecutionROIBot=DummyBot
    )
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)
    mod = types.ModuleType("menace.data_bot")
    mod.DataBot = DummyDataBot
    mod.MetricsDB = DummyBot
    monkeypatch.setitem(sys.modules, "menace.data_bot", mod)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(Path(__file__).resolve().parents[1] / "sandbox_runner.py"),
        submodule_search_locations=[
            str(Path(__file__).resolve().parents[1] / "sandbox_runner")
        ],
    )
    sandbox_runner = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sandbox_runner
    spec.loader.exec_module(sandbox_runner)

    monkeypatch.setattr(sandbox_runner, "MenaceOrchestrator", DummyOrch)
    monkeypatch.setattr(sandbox_runner, "_SandboxMetaLogger", DummyMetaLogger)
    monkeypatch.setattr(
        sandbox_runner,
        "scan_repo_sections",
        lambda path: {"mod.py": {"sec": ["pass"]}},
    )

    def fake_cycle(ctx, sec, snip, tracker, scenario=None):
        from menace.environment_generator import adapt_presets

        if getattr(ctx, "adapt_presets", True):
            sandbox_runner.SANDBOX_ENV_PRESETS = adapt_presets(
                tracker, sandbox_runner.SANDBOX_ENV_PRESETS
            )
            os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(
                sandbox_runner.SANDBOX_ENV_PRESETS
            )

    monkeypatch.setattr(sandbox_runner, "_sandbox_cycle_runner", fake_cycle)

    class DummyCtx:
        def __init__(self):
            self.adapt_presets = False

    sandbox_runner.SANDBOX_ENV_PRESETS = [{"num": 0}]
    tracker = DummyTracker()
    sandbox_runner._sandbox_cycle_runner(DummyCtx(), "mod.py:sec", "pass", tracker)
    assert sandbox_runner.SANDBOX_ENV_PRESETS == [{"num": 0}]

    tracker2 = DummyTracker()
    sandbox_runner._sandbox_cycle_runner(DummyCtx(), "mod.py:sec", "pass", tracker2)
    assert sandbox_runner.SANDBOX_ENV_PRESETS == [{"num": 0}]


class _BrainstormTracker(DummyTracker):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.roi_history = [0.05]
        self.metrics_history = {"security_score": [1.0], "efficiency": [1.0]}

    def diminishing(self):
        return 0.1

    def register_metrics(self, *names):
        for name in names:
            self.metrics_history.setdefault(str(name), [0.0] * len(self.roi_history))


class _StaticImprover(DummyImprover):
    def run_cycle(self):
        class R:
            roi = types.SimpleNamespace(roi=0.05)

        return R()


class _NoFlagMetaLogger(DummyMetaLogger):
    def diminishing(self, threshold=None):
        return []


class _ResilienceImprover(DummyImprover):
    def __init__(self):
        self._vals = iter([0.3, 0.2])

    def run_cycle(self):
        val = next(self._vals)
        return types.SimpleNamespace(roi=types.SimpleNamespace(roi=val))


class _CaptureLogger(DummyMetaLogger):
    instance = None

    def __init__(self, path):
        super().__init__(path)
        _CaptureLogger.instance = self
        self.reasons = []

    def log_cycle(self, cycle, roi, modules, reason):
        self.reasons.append(reason)

    def diminishing(self, threshold=None):
        return []


def test_brainstorm_trigger_on_low_roi(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("SANDBOX_CYCLES", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "1")
    monkeypatch.setenv("SANDBOX_BRAINSTORM_INTERVAL", "1")
    monkeypatch.setenv("SANDBOX_BRAINSTORM_RETRIES", "1")
    monkeypatch.setenv("SANDBOX_ROI_TOLERANCE", "0")

    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=_BrainstormTracker)
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBus)
    _stub_module(
        monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyOrch
    )
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyPolicy
    )
    _stub_module(
        monkeypatch,
        "menace.self_improvement_engine",
        SelfImprovementEngine=lambda *a, **k: _StaticImprover(),
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyTester)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyEngine)
    _stub_module(
        monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.pre_execution_roi_bot", PreExecutionROIBot=DummyBot
    )
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)
    mod = types.ModuleType("menace.data_bot")
    mod.DataBot = DummyDataBot
    mod.MetricsDB = DummyBot
    monkeypatch.setitem(sys.modules, "menace.data_bot", mod)
    _stub_module(monkeypatch, "menace.chatgpt_idea_bot", ChatGPTClient=DummyGPT)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(Path(__file__).resolve().parents[1] / "sandbox_runner.py"),
        submodule_search_locations=[
            str(Path(__file__).resolve().parents[1] / "sandbox_runner")
        ],
    )
    sandbox_runner = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sandbox_runner
    spec.loader.exec_module(sandbox_runner)

    monkeypatch.setattr(sandbox_runner, "MenaceOrchestrator", DummyOrch)
    monkeypatch.setattr(sandbox_runner, "_SandboxMetaLogger", _NoFlagMetaLogger)
    monkeypatch.setattr(
        sandbox_runner,
        "build_section_prompt",
        lambda *a, **k: "Brainstorm high level improvements to increase ROI.",
    )
    monkeypatch.setattr(
        sandbox_runner,
        "scan_repo_sections",
        lambda path: {"mod.py": {"sec": ["pass"]}},
    )

    ctx = sandbox_runner._sandbox_init(
        {}, argparse.Namespace(sandbox_data_dir=str(tmp_path))
    )
    ctx.prev_roi = 0.05
    sandbox_runner._sandbox_cycle_runner(ctx, None, None, ctx.tracker)
    sandbox_runner._sandbox_cleanup(ctx)

    assert any(
        "Brainstorm high level improvements" in msg[0]["content"]
        for msg in DummyGPT.calls
    )
    DummyGPT.calls.clear()


def test_brainstorm_trigger_on_resilience_drop(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("SANDBOX_CYCLES", "2")
    monkeypatch.setenv("OPENAI_API_KEY", "1")
    monkeypatch.setenv("SANDBOX_BRAINSTORM_INTERVAL", "0")
    monkeypatch.setenv("SANDBOX_BRAINSTORM_RETRIES", "10")
    monkeypatch.setenv("SANDBOX_ROI_TOLERANCE", "0")

    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=_BrainstormTracker)
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBus)
    _stub_module(
        monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyOrch
    )
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyPolicy
    )
    _stub_module(
        monkeypatch,
        "menace.self_improvement_engine",
        SelfImprovementEngine=lambda *a, **k: _ResilienceImprover(),
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyTester)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyEngine)
    _stub_module(
        monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.pre_execution_roi_bot", PreExecutionROIBot=DummyBot
    )
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)
    mod = types.ModuleType("menace.data_bot")
    mod.DataBot = DummyDataBot
    mod.MetricsDB = DummyBot
    monkeypatch.setitem(sys.modules, "menace.data_bot", mod)
    _stub_module(monkeypatch, "menace.chatgpt_idea_bot", ChatGPTClient=DummyGPT)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(Path(__file__).resolve().parents[1] / "sandbox_runner.py"),
        submodule_search_locations=[
            str(Path(__file__).resolve().parents[1] / "sandbox_runner")
        ],
    )
    sandbox_runner = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sandbox_runner
    spec.loader.exec_module(sandbox_runner)

    monkeypatch.setattr(sandbox_runner, "MenaceOrchestrator", DummyOrch)
    monkeypatch.setattr(sandbox_runner, "_SandboxMetaLogger", _CaptureLogger)
    monkeypatch.setattr(
        sandbox_runner,
        "build_section_prompt",
        lambda *a, **k: "Brainstorm high level improvements to increase ROI.",
    )
    monkeypatch.setattr(
        sandbox_runner,
        "scan_repo_sections",
        lambda path: {"mod.py": {"sec": ["pass"]}},
    )

    ctx = sandbox_runner._sandbox_init(
        {}, argparse.Namespace(sandbox_data_dir=str(tmp_path))
    )
    sandbox_runner._sandbox_cycle_runner(ctx, None, None, ctx.tracker)
    sandbox_runner._sandbox_cleanup(ctx)

    assert any(
        "Brainstorm high level improvements" in msg[0]["content"]
        for msg in DummyGPT.calls
    )
    assert "resilience_brainstorm" in _CaptureLogger.instance.reasons
    DummyGPT.calls.clear()


def test_sandbox_prediction_mae_and_reliability(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("SANDBOX_CYCLES", "2")

    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyEngine)
    _stub_module(
        monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyPolicy
    )
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBus)
    _stub_module(
        monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyOrch
    )
    _stub_module(
        monkeypatch,
        "menace.self_improvement_engine",
        SelfImprovementEngine=lambda *a, **k: DummyImprover(),
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyTester)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )
    mod = types.ModuleType("menace.data_bot")
    mod.DataBot = DummyDataBot
    mod.MetricsDB = DummyBot
    monkeypatch.setitem(sys.modules, "menace.data_bot", mod)
    _stub_module(monkeypatch, "menace.pre_execution_roi_bot", PreExecutionROIBot=None)
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)

    import importlib
    import sandbox_runner
    import menace.roi_tracker as rt

    importlib.reload(sandbox_runner)

    monkeypatch.setattr(sandbox_runner, "MenaceOrchestrator", DummyOrch)
    monkeypatch.setattr(sandbox_runner, "_SandboxMetaLogger", DummyMetaLogger)
    monkeypatch.setattr(
        sandbox_runner, "scan_repo_sections", lambda path: {"m.py": {"sec": ["pass"]}}
    )
    monkeypatch.setattr(rt.ROITracker, "forecast", lambda self: (0.15, (0.0, 0.0)))

    ctx = sandbox_runner._sandbox_init(
        {}, argparse.Namespace(sandbox_data_dir=str(tmp_path))
    )

    class SeqImprover(DummyImprover):
        def __init__(self):
            self._vals = iter([0.1, 0.2])

        def run_cycle(self):
            val = next(self._vals)
            return types.SimpleNamespace(roi=types.SimpleNamespace(roi=val))

    ctx.improver = SeqImprover()
    sandbox_runner._sandbox_cycle_runner(ctx, None, None, ctx.tracker)
    sandbox_runner._sandbox_cleanup(ctx)

    tracker = ctx.tracker
    assert tracker.predicted_roi and tracker.actual_roi
    mae = tracker.rolling_mae()
    reliability = tracker.reliability()
    assert mae == pytest.approx(abs(tracker.predicted_roi[0] - tracker.actual_roi[0]))
    expected_rel = 1.0 / (1.0 + rt.ROITracker._ema([abs(tracker.predicted_roi[0] - tracker.actual_roi[0])]))
    assert reliability == pytest.approx(expected_rel)
    assert tracker.metrics_history["roi_reliability"][-1] == pytest.approx(reliability)


class _SynergyTracker:
    def __init__(self, *a, **k):
        self.metrics_history = {}
        self.roi_history = []
        self.synergy_history = []

    def update(self, prev, curr, modules=None, resources=None, metrics=None):
        if metrics:
            for m, v in metrics.items():
                self.metrics_history.setdefault(m, []).append(v)
        for name in self.metrics_history:
            if not metrics or name not in metrics:
                last = (
                    self.metrics_history[name][-1]
                    if self.metrics_history[name]
                    else 0.0
                )
                self.metrics_history[name].append(last)
        self.roi_history.append(curr)
        return 0, [], False

    def forecast(self):
        return 0.0, (0.0, 0.0)

    def forecast_metric(self, name):
        return 0.0, (0.0, 0.0)

    def diminishing(self):
        return 0.0

    def register_metrics(self, *names):
        for name in names:
            self.metrics_history.setdefault(str(name), [0.0] * len(self.roi_history))

    def record_prediction(self, predicted, actual):
        pass

    def record_metric_prediction(self, metric, predicted, actual):
        pass

    def predict_all_metrics(self, manager, features):
        pass

    def rolling_mae(self, window=None):
        return 0.0

    def reliability(self):
        return 0.0

    def load_history(self, path):
        pass

    def save_history(self, path):
        pass


def test_workflow_sim_synergy_metrics(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBot)
    _stub_module(monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.self_improvement_engine", SelfImprovementEngine=DummyBot
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyBot)
    _stub_module(
        monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox
    )
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(
        monkeypatch, "menace.code_database", PatchHistoryDB=DummyBot, CodeDB=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot
    )
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(
        monkeypatch,
        "menace.discrepancy_detection_bot",
        DiscrepancyDetectionBot=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot()
    )
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot, DataBot=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=_SynergyTracker)
    _stub_module(monkeypatch, "menace.metrics_dashboard", MetricsDashboard=DummyBot)
    jinja_mod = types.ModuleType("jinja2")
    jinja_mod.Template = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "jinja2", jinja_mod)

    from pathlib import Path as _P
    import importlib.util

    ROOT = _P(__file__).resolve().parents[1]
    pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
    pkg.__path__ = [str(ROOT)]

    spec = importlib.util.spec_from_file_location(
        "menace.task_handoff_bot",
        ROOT / "task_handoff_bot.py",
        submodule_search_locations=[str(ROOT)],
    )
    thb = importlib.util.module_from_spec(spec)
    sys.modules["menace.task_handoff_bot"] = thb
    spec.loader.exec_module(thb)
    wf_db = thb.WorkflowDB(tmp_path / "wf.db")
    wf_db.add(thb.WorkflowRecord(workflow=["simple_functions:print_ten"], title="t"))

    import sandbox_runner
    import sandbox_runner.environment as env

    monkeypatch.setattr(
        env,
        "SANDBOX_EXTRA_METRICS",
        {"shannon_entropy": 0.0, "flexibility": 0.0, "energy_consumption": 0.0},
    )

    def fake_sim(code_str, stub=None):
        return {"risk_flags_triggered": []}

    monkeypatch.setattr(sandbox_runner, "simulate_execution_environment", fake_sim)

    tracker = sandbox_runner.run_workflow_simulations(
        workflows_db=str(tmp_path / "wf.db"), env_presets=[{"env": "dev"}]
    )

    synergy_keys = [k for k in tracker.metrics_history if k.startswith("synergy_")]
    assert "synergy_roi" in synergy_keys
    for key in (
        "synergy_shannon_entropy",
        "synergy_flexibility",
        "synergy_energy_consumption",
    ):
        assert key in synergy_keys


def test_synergy_adaptation_increase(monkeypatch):
    tracker = _SynergyTracker()
    tracker.metrics_history = {
        "security_score": [70, 70, 70],
        "synergy_roi": [0.2, 0.3, 0.25],
        "synergy_security_score": [6.0, 5.5, 6.5],
    }
    presets = [{"THREAT_INTENSITY": 30, "SECURITY_LEVEL": 2}]
    import environment_generator as eg

    new = eg.adapt_presets(tracker, presets)
    assert new[0]["THREAT_INTENSITY"] > 30
    assert new[0]["SECURITY_LEVEL"] > 2


def test_synergy_adaptation_decrease(monkeypatch):
    tracker = _SynergyTracker()
    tracker.metrics_history = {
        "security_score": [70, 70, 70],
        "synergy_roi": [-0.3, -0.2, -0.25],
        "synergy_security_score": [-6.0, -5.5, -6.5],
    }
    presets = [{"THREAT_INTENSITY": 70, "SECURITY_LEVEL": 4}]
    import environment_generator as eg

    new = eg.adapt_presets(tracker, presets)
    assert new[0]["THREAT_INTENSITY"] < 70
    assert new[0]["SECURITY_LEVEL"] < 4


def test_synergy_adaptability_preset_adjust(monkeypatch):
    tracker = _SynergyTracker()
    tracker.metrics_history = {
        "security_score": [70, 70, 70],
        "synergy_adaptability": [0.06, 0.07, 0.05],
    }
    presets = [{"CPU_LIMIT": "2", "MEMORY_LIMIT": "512Mi"}]
    import environment_generator as eg

    new = eg.adapt_presets(tracker, presets)
    assert float(new[0]["CPU_LIMIT"]) < 2
    assert new[0]["MEMORY_LIMIT"] != "512Mi"


def test_synergy_converged_confidence_levels(monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    _stub_module(
        monkeypatch, "menace.metrics_dashboard", MetricsDashboard=lambda *a, **k: None
    )

    import importlib, sys

    sys.modules.pop("sandbox_runner.cli", None)
    cli = importlib.import_module("sandbox_runner.cli")

    hist = [
        {"synergy_roi": 0.001},
        {"synergy_roi": 0.001},
        {"synergy_roi": 0.001},
        {"synergy_roi": 0.001},
        {"synergy_roi": 0.001},
    ]

    ok, ema, conf = cli._synergy_converged(hist, 5, 0.01, confidence=0.95)
    assert ok is True

    ok, ema, conf = cli._synergy_converged(hist, 5, 0.01, confidence=0.99)
    assert ok is True


def test_synergy_variance_change_detection(monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    _stub_module(
        monkeypatch, "menace.metrics_dashboard", MetricsDashboard=lambda *a, **k: None
    )
    _patch_adfuller(monkeypatch, 0.01)
    _patch_levene(monkeypatch, 0.001)

    import importlib, sys

    sys.modules.pop("sandbox_runner.cli", None)
    cli = importlib.import_module("sandbox_runner.cli")

    hist = [
        {"synergy_roi": 0.001},
        {"synergy_roi": 0.0011},
        {"synergy_roi": 0.0012},
        {"synergy_roi": 0.0013},
        {"synergy_roi": 0.0014},
    ]

    ok, ema, conf = cli._synergy_converged(
        hist,
        5,
        0.01,
        variance_confidence=0.95,
    )
    assert ok is False


def test_synergy_variance_stable(monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    _stub_module(
        monkeypatch, "menace.metrics_dashboard", MetricsDashboard=lambda *a, **k: None
    )
    _patch_adfuller(monkeypatch, 0.01)
    _patch_levene(monkeypatch, 0.8)

    import importlib, sys

    sys.modules.pop("sandbox_runner.cli", None)
    cli = importlib.import_module("sandbox_runner.cli")

    hist = [
        {"synergy_roi": 0.001},
        {"synergy_roi": 0.0011},
        {"synergy_roi": 0.0012},
        {"synergy_roi": 0.0011},
        {"synergy_roi": 0.001},
    ]

    ok, ema, conf = cli._synergy_converged(
        hist,
        5,
        0.01,
        variance_confidence=0.95,
    )
    assert ok is False
