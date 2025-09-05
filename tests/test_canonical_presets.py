import sys
from pathlib import Path

import pytest

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
        self.metrics_history = {}
        self.synergy_metrics_history = {}
        self.scenario_synergy = {}

    def register_metrics(self, *names):
        for n in names:
            target = (
                self.synergy_metrics_history if str(n).startswith("synergy_") else self.metrics_history
            )
            target.setdefault(str(n), [])

    def update(self, prev, curr, modules=None, resources=None, metrics=None):
        self.calls.append({"modules": modules, "metrics": metrics})
        self.roi_history.append(curr)
        for k, v in (metrics or {}).items():
            hist = self.synergy_metrics_history if k.startswith("synergy_") else self.metrics_history
            hist.setdefault(k, []).append(v)
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

    def get_scenario_synergy(self, name):
        return self.scenario_synergy.get(name, [])


def test_generate_canonical_presets():
    from menace import environment_generator as eg

    presets = eg.generate_canonical_presets()
    assert set(presets) == {
        "high_latency_api",
        "hostile_input",
        "user_misuse",
        "concurrency_spike",
    }
    for levels in presets.values():
        assert set(levels) == {"low", "high"}
        for lvl, data in levels.items():
            assert data["SCENARIO_NAME"] in presets
    # deterministic
    assert presets == eg.generate_canonical_presets()


def test_run_repo_section_simulations_canonical(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(sys.modules["menace"], "__path__", [str(root)])

    (tmp_path / "m.py").write_text("def f():\n    return 1\n")  # path-ignore

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

    monkeypatch.setenv("SANDBOX_PRESET_MODE", "canonical")

    import sandbox_runner
    import sandbox_runner.environment as env

    monkeypatch.setattr(
        sandbox_runner,
        "scan_repo_sections",
        lambda p, modules=None: {"m.py": {"sec": ["pass"]}},  # path-ignore
        raising=False,
    )

    calls = []
    counts = {name: 0 for name in ["high_latency_api", "hostile_input", "user_misuse", "concurrency_spike"]}

    async def fake_worker(snippet, env_input, threshold):
        name = env_input.get("SCENARIO_NAME")
        counts[name] += 1
        base = {
            "high_latency_api": 1.0,
            "hostile_input": 2.0,
            "user_misuse": 3.0,
            "concurrency_spike": 4.0,
        }[name]
        roi = base if counts[name] == 1 else base + 0.5
        metrics = {"profitability": roi}
        calls.append(name)
        return {"exit_code": 0}, [(0.0, roi, metrics)]

    monkeypatch.setattr(env, "_section_worker", fake_worker)

    tracker = sandbox_runner.run_repo_section_simulations(str(tmp_path), input_stubs=[{}])

    expected = {"high_latency_api", "hostile_input", "user_misuse", "concurrency_spike"}
    assert set(calls) == expected
    assert set(tracker.scenario_synergy) == expected
    for name in expected:
        assert counts[name] == 2
        assert "synergy_roi" in tracker.scenario_synergy[name][0]
