import types
import sys
from pathlib import Path


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


class DummyBot:
    def __init__(self, *a, **k):
        pass


class DummySandbox:
    calls = 0

    def __init__(self, *a, **k):
        pass

    def analyse_and_fix(self):
        DummySandbox.calls += 1


class DummyTracker:
    def __init__(self, *a, **k):
        self.roi_history = []
        self.metrics_history = {}
        self.scenario_synergy = {}

    def update(self, prev, curr, modules=None, resources=None, metrics=None):
        self.roi_history.append(curr)
        metrics = metrics or {}
        for m, v in metrics.items():
            self.metrics_history.setdefault(m, []).append(float(v))
        for m in list(self.metrics_history):
            if m not in metrics:
                last = self.metrics_history[m][-1]
                self.metrics_history[m].append(last)
        return 0.0, [], False

    def forecast(self):
        return 0.0, (0.0, 0.0)

    def forecast_metric(self, name):
        return 0.0, (0.0, 0.0)

    def diminishing(self):
        return 0.0

    def record_prediction(self, *a, **k):
        pass

    def record_metric_prediction(self, *a, **k):
        pass

    def register_metrics(self, *names):
        for n in names:
            self.metrics_history.setdefault(str(n), [0.0] * len(self.roi_history))

    def get_scenario_synergy(self, name):
        return self.scenario_synergy.get(name, [])


class FakeWorkflowRecord:
    def __init__(self, workflow, wid=1, title="t"):
        self.workflow = workflow
        self.wid = wid
        self.title = title


class FakeWorkflowDB:
    def __init__(self, path):
        self.path = Path(path)

    def fetch(self, limit=None):
        return [FakeWorkflowRecord(["simple_functions:print_ten"], wid=1)]


def _setup_runner(monkeypatch):
    ROOT = Path(__file__).resolve().parents[1]
    pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
    pkg.__path__ = [str(ROOT)]
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    _stub_module(monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox)
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot)
    _stub_module(monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)
    _stub_module(
        monkeypatch,
        "menace.task_handoff_bot",
        WorkflowDB=FakeWorkflowDB,
        WorkflowRecord=FakeWorkflowRecord,
    )


def test_profiles_results(monkeypatch):
    _setup_runner(monkeypatch)
    import sandbox_runner.environment as env

    async def fake_section_worker(snippet, env_input, threshold):
        return {"exit_code": 0}, [(0.0, 1.0, {"throughput": 1.0})]

    monkeypatch.setattr(env, "_section_worker", fake_section_worker)
    monkeypatch.setattr(env, "simulate_execution_environment", lambda *a, **k: {})
    from environment_generator import generate_presets

    profiles = [
        "high_latency_api",
        "hostile_input",
        "user_misuse",
        "concurrency_spike",
        "schema_drift",
        "flaky_upstream",
    ]
    presets = generate_presets(profiles=profiles)
    tracker = env.run_workflow_simulations("wf.db", env_presets=presets)
    for name in profiles:
        key = f"throughput:{name}"
        assert key in tracker.metrics_history


def test_canonical_profiles_autoload(monkeypatch):
    _setup_runner(monkeypatch)
    import sandbox_runner.environment as env
    monkeypatch.setattr(env, "simulate_execution_environment", lambda *a, **k: {})

    async def fake_section_worker(snippet, env_input, threshold):
        return {"exit_code": 0}, [(0.0, 1.0, {"throughput": 1.0})]

    monkeypatch.setattr(env, "_section_worker", fake_section_worker)
    canonical = {
        "high_latency_api": {
            "low": {"SCENARIO_NAME": "high_latency_api"},
            "high": {"SCENARIO_NAME": "high_latency_api"},
        },
        "hostile_input": {
            "low": {"SCENARIO_NAME": "hostile_input"},
            "high": {"SCENARIO_NAME": "hostile_input"},
        },
        "user_misuse": {
            "low": {"SCENARIO_NAME": "user_misuse"},
            "high": {"SCENARIO_NAME": "user_misuse"},
        },
        "concurrency_spike": {
            "low": {"SCENARIO_NAME": "concurrency_spike"},
            "high": {"SCENARIO_NAME": "concurrency_spike"},
        },
        "schema_drift": {
            "low": {"SCENARIO_NAME": "schema_drift"},
            "high": {"SCENARIO_NAME": "schema_drift"},
        },
        "flaky_upstream": {
            "low": {"SCENARIO_NAME": "flaky_upstream"},
            "high": {"SCENARIO_NAME": "flaky_upstream"},
        },
    }
    _stub_module(
        monkeypatch,
        "menace.environment_generator",
        generate_canonical_presets=lambda: canonical,
        generate_presets=lambda profiles=None: [],
    )
    tracker = env.run_workflow_simulations("wf.db")
    for name in [
        "high_latency_api",
        "hostile_input",
        "user_misuse",
        "concurrency_spike",
        "schema_drift",
        "flaky_upstream",
    ]:
        assert f"throughput:{name}" in tracker.metrics_history


def test_hostile_input_triggers_fix(monkeypatch):
    _setup_runner(monkeypatch)
    DummySandbox.calls = 0
    import sandbox_runner.environment as env

    counter = {"n": 0}

    def fake_sim(snippet, env_input):
        if env_input.get("SANDBOX_STUB_STRATEGY") == "hostile" and counter["n"] == 0:
            counter["n"] += 1
            return {"risk_flags_triggered": ["attack"]}
        return {"risk_flags_triggered": []}

    async def fake_section_worker(snippet, env_input, threshold):
        return {"exit_code": 0}, [(0.0, 1.0, {})]

    monkeypatch.setattr(env, "simulate_execution_environment", fake_sim)
    monkeypatch.setattr(env, "_section_worker", fake_section_worker)
    from environment_generator import generate_presets

    presets = generate_presets(profiles=["hostile_input"])
    tracker = env.run_workflow_simulations("wf.db", env_presets=presets)
    assert tracker.roi_history
    assert DummySandbox.calls > 0


def test_concurrency_spike_metrics(monkeypatch):
    _setup_runner(monkeypatch)
    import sandbox_runner.environment as env
    monkeypatch.setattr(env, "simulate_execution_environment", lambda *a, **k: {})

    async def fake_section_worker(snippet, env_input, threshold):
        throughput = float(env_input.get("THREAD_BURST", 0)) + float(
            env_input.get("ASYNC_TASK_BURST", 0)
        )
        return {"exit_code": 0}, [(0.0, 1.0, {"throughput": throughput})]

    monkeypatch.setattr(env, "_section_worker", fake_section_worker)
    from environment_generator import generate_presets

    presets = generate_presets(profiles=["concurrency_spike"])
    tracker = env.run_workflow_simulations("wf.db", env_presets=presets)
    thr = float(presets[0]["THREAD_BURST"]) + float(presets[0]["ASYNC_TASK_BURST"])
    assert tracker.metrics_history.get("throughput")
    assert thr in tracker.metrics_history["throughput"]


def test_new_scenario_specific_metrics(monkeypatch):
    _setup_runner(monkeypatch)
    from sandbox_runner.environment import _scenario_specific_metrics

    extra = _scenario_specific_metrics(
        "schema_drift", {"schema_mismatches": 2, "schema_checks": 10}
    )
    assert extra["schema_errors"] == 2
    assert extra["schema_mismatch_rate"] == 0.2

    extra = _scenario_specific_metrics(
        "flaky_upstream", {"upstream_failures": 1, "upstream_requests": 5}
    )
    assert extra["upstream_failures"] == 1
    assert extra["upstream_failure_rate"] == 0.2
