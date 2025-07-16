import argparse
import sys
import json
from pathlib import Path
import types
import subprocess
import pytest

from tests.test_sandbox_runner_metrics import (
    _stub_module,
    DummyTracker,
    DummyBot,
    DummyPolicy,
    DummyDataBot,
    DummySandbox,
    DummyTester,
    DummyOrch,
    DummyMetaLogger,
    DummyEngine,
    DummyImprover,
    DummyBus,
    _SynergyTracker,
)


class _Tracker(DummyTracker):
    def register_metrics(self, *names):
        pass


def test_bandit_and_coverage_metrics(monkeypatch, tmp_path):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=_Tracker)
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBus)
    _stub_module(monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyOrch)
    _stub_module(monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyPolicy)
    _stub_module(monkeypatch, "menace.self_improvement_engine", SelfImprovementEngine=lambda *a, **k: DummyImprover())
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyTester)
    _stub_module(monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox)
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyEngine)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot, PatchHistoryDB=DummyBot)
    _stub_module(monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot())
    _stub_module(monkeypatch, "menace.discrepancy_detection_bot", DiscrepancyDetectionBot=DummyBot)
    _stub_module(monkeypatch, "menace.pre_execution_roi_bot", PreExecutionROIBot=DummyBot)
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)
    data_mod = types.ModuleType("menace.data_bot")
    data_mod.DataBot = DummyDataBot
    data_mod.MetricsDB = DummyBot
    monkeypatch.setitem(sys.modules, "menace.data_bot", data_mod)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sandbox_runner",
        str(Path(__file__).resolve().parents[1] / "sandbox_runner.py"),
        submodule_search_locations=[str(Path(__file__).resolve().parents[1] / "sandbox_runner")],
    )
    sandbox_runner = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_runner"] = sandbox_runner
    spec.loader.exec_module(sandbox_runner)

    monkeypatch.setattr(sandbox_runner, "MenaceOrchestrator", DummyOrch)
    monkeypatch.setattr(sandbox_runner, "_SandboxMetaLogger", DummyMetaLogger)
    monkeypatch.setattr(sandbox_runner, "scan_repo_sections", lambda path: {"mod.py": {"sec": ["pass"]}})
    monkeypatch.setattr(sandbox_runner.cycle, "psutil", None)
    monkeypatch.setattr(sandbox_runner.cycle.shutil, "which", lambda n: None)
    monkeypatch.setattr(sandbox_runner.cycle, "mi_visit", None)
    monkeypatch.setattr(sandbox_runner.cycle, "PylintRun", None)
    monkeypatch.setattr(sandbox_runner.cycle, "TextReporter", None)

    def fake_run(cmd, capture_output=True, text=True, check=False):
        if cmd and cmd[0] == "bandit":
            out = json.dumps({"results": [{"issue_severity": "HIGH"}, {"issue_severity": "LOW"}]})
            return subprocess.CompletedProcess(cmd, 0, out, "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sandbox_runner.subprocess, "run", fake_run)

    (tmp_path / "mod.py").write_text("print('x')\nprint('y')\n")
    from coverage import CoverageData
    cov = CoverageData(basename=str(tmp_path / ".coverage"))
    cov.add_lines({str(tmp_path / "mod.py"): {1, 2}})
    cov.write()

    ctx = sandbox_runner._sandbox_init({}, argparse.Namespace(sandbox_data_dir=str(tmp_path)))
    ctx.repo = tmp_path
    ctx.changed_modules = lambda last_id: (["mod.py"], last_id)
    sandbox_runner._sandbox_cycle_runner(ctx, None, None, ctx.tracker)
    sandbox_runner._sandbox_cleanup(ctx)

    metrics = ctx.tracker.records[-1][1]
    assert metrics["security_score"] == pytest.approx(80.0)
    assert metrics["adaptability"] == pytest.approx(100.0)
    assert metrics["flexibility"] == pytest.approx(1.0)
