from hypothesis import given, strategies as st, settings, HealthCheck
import types, sys

from tests.test_menace_master import _setup_mm_stubs
from tests.test_sandbox_section_simulations import _stub_module, DummyBot, DummyTracker2

import sandbox_runner
import sandbox_runner.environment as env


@settings(max_examples=3, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(payload=st.text(min_size=0, max_size=5))
def test_hostile_input_scenario(monkeypatch, tmp_path, payload):
    _setup_mm_stubs(monkeypatch)
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    (tmp_path / "m.py").write_text("def f(x):\n    return x\n")  # path-ignore

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
    _stub_module(monkeypatch, "networkx")
    sqla = types.ModuleType("sqlalchemy")
    sqla_engine = types.ModuleType("sqlalchemy.engine")
    sqla_engine.Engine = object
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqla)
    monkeypatch.setitem(sys.modules, "sqlalchemy.engine", sqla_engine)
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker2)

    monkeypatch.setattr(
        sandbox_runner,
        "scan_repo_sections",
        lambda p, modules=None: {"m.py": {"sec": ["pass"]}},  # path-ignore
        raising=False,
    )
    monkeypatch.setattr(sandbox_runner, "simulate_execution_environment", lambda *a, **k: {"risk_flags_triggered": []})

    presets = [{"SCENARIO_NAME": "hostile", "FAILURE_MODES": ["hostile_input"]}]
    tracker = sandbox_runner.run_repo_section_simulations(
        str(tmp_path), input_stubs=[{"x": payload}], env_presets=presets
    )
    assert any(call["modules"][1] == "hostile" for call in tracker.calls)
