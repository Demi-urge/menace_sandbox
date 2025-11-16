import sys
import types
from pathlib import Path


from tests.test_sandbox_section_simulations import _stub_module
from tests.test_menace_master import _setup_mm_stubs


class DummyBot:
    def __init__(self, *a, **k):
        pass


class DummyKG:
    def __init__(self, *a, **k):
        pass


class DummyTracker:
    def __init__(self, *a, **k):
        self.calls = []
        self.roi_history = []

    def update(self, prev, curr, modules=None, resources=None, metrics=None):
        self.calls.append((modules, metrics))
        self.roi_history.append(curr)
        return 0.0, [], False

    def forecast(self):
        return 0.0, (0.0, 0.0)

    def diminishing(self):
        return 0.0

    def record_prediction(self, *a, **k):
        pass

    def record_metric_prediction(self, *a, **k):
        pass

    def rolling_mae(self, window=None):
        return 0.0

    def load_history(self, path):
        pass

    def save_history(self, path):
        pass


CANONICAL = {
    "high_latency_api": {"high": {"SCENARIO_NAME": "high_latency_api"}},
    "hostile_input": {"high": {"SCENARIO_NAME": "hostile_input"}},
    "user_misuse": {"high": {"SCENARIO_NAME": "user_misuse"}},
    "concurrency_spike": {"high": {"SCENARIO_NAME": "concurrency_spike"}},
}


def _common_stubs(monkeypatch):
    _setup_mm_stubs(monkeypatch)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(sys.modules["menace"], "__path__", [str(root)])
    _stub_module(monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyBot)
    _stub_module(monkeypatch, "menace.self_improvement", SelfImprovementEngine=DummyBot)
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyBot)
    _stub_module(monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummyBot)
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", PatchHistoryDB=DummyBot, CodeDB=DummyBot)
    _stub_module(monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot())
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot, DataBot=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker)
    _stub_module(monkeypatch, "knowledge_graph", KnowledgeGraph=DummyKG)
    _stub_module(monkeypatch, "networkx")
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)


def test_missing_canonical_scenario_added(monkeypatch, tmp_path):
    _common_stubs(monkeypatch)

    (tmp_path / "m.py").write_text("def f():\n    return 1\n")  # path-ignore

    import sandbox_runner
    import sandbox_runner.environment as env

    monkeypatch.setattr(
        sandbox_runner,
        "scan_repo_sections",
        lambda p, modules=None: {"m.py": {"sec": ["pass"]}},  # path-ignore
        raising=False,
    )
    monkeypatch.setattr(
        sandbox_runner,
        "simulate_execution_environment",
        lambda code_str, env_input: {},
        raising=False,
    )

    async def fake_worker(snippet, env_input, threshold):
        return {"exit_code": 0}, [(0.0, 1.0, {"profitability": 1.0})]

    monkeypatch.setattr(env, "_section_worker", fake_worker)

    eg_mod = types.ModuleType("menace.environment_generator")
    eg_mod.generate_presets = lambda *a, **k: [
        {"SCENARIO_NAME": "high_latency_api"},
        {"SCENARIO_NAME": "hostile_input"},
        {"SCENARIO_NAME": "user_misuse"},
    ]
    eg_mod.generate_canonical_presets = lambda: CANONICAL
    eg_mod.suggest_profiles_for_module = lambda module: []
    monkeypatch.setitem(sys.modules, "menace.environment_generator", eg_mod)

    tracker, details = sandbox_runner.run_repo_section_simulations(
        str(tmp_path), input_stubs=[{}], return_details=True
    )

    expected = {"high_latency_api", "hostile_input", "user_misuse", "concurrency_spike"}
    assert set(details["m.py"].keys()) == expected  # path-ignore


def test_module_specific_presets(monkeypatch, tmp_path):
    _common_stubs(monkeypatch)

    (tmp_path / "a.py").write_text("def f():\n    return 1\n")  # path-ignore
    (tmp_path / "b.py").write_text("def g():\n    return 2\n")  # path-ignore

    import sandbox_runner
    import sandbox_runner.environment as env

    monkeypatch.setattr(
        sandbox_runner,
        "scan_repo_sections",
        lambda p, modules=None: {"a.py": {"s": ["pass"]}, "b.py": {"s": ["pass"]}},  # path-ignore
        raising=False,
    )
    monkeypatch.setattr(
        sandbox_runner,
        "simulate_execution_environment",
        lambda code_str, env_input: {},
        raising=False,
    )

    async def fake_worker(snippet, env_input, threshold):
        return {"exit_code": 0}, [(0.0, 1.0, {"profitability": 1.0})]

    monkeypatch.setattr(env, "_section_worker", fake_worker)

    eg_mod = types.ModuleType("menace.environment_generator")
    eg_mod.generate_presets = lambda *a, **k: []
    eg_mod.generate_canonical_presets = lambda: CANONICAL
    eg_mod.suggest_profiles_for_module = lambda m: []
    monkeypatch.setitem(sys.modules, "menace.environment_generator", eg_mod)

    tracker, details = sandbox_runner.run_repo_section_simulations(
        str(tmp_path),
        input_stubs=[{}],
        env_presets=[{"SCENARIO_NAME": "high_latency_api"}],
        return_details=True,
    )

    lst_a = details["a.py"]["high_latency_api"]  # path-ignore
    lst_b = details["b.py"]["high_latency_api"]  # path-ignore
    assert lst_a is not lst_b
