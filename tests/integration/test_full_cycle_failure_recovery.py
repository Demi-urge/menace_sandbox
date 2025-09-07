import json
import logging
import subprocess
import types

import pytest

from tests.test_self_test_service import mod as sts
from tests.test_self_debugger_sandbox import sds, DummyTelem, DummyTrail, DummyBuilder
from tests.test_self_debugger_patch_flow import FlowEngine


class DummyDB:
    def __init__(self):
        self.entries: list[tuple[str, int]] = []

    def add_test_result(self, path: str, failed: int) -> None:
        self.entries.append((path, failed))


class DummyTracker:
    def __init__(self):
        self.metrics_history = {"synergy_roi": [0.0]}

    def advance(self):
        self.metrics_history["synergy_roi"].append(self.metrics_history["synergy_roi"][-1] + 0.1)


class Pipeline:
    def __init__(self):
        self.calls = 0

    def run(self, model: str, energy: int = 1):
        self.calls += 1
        if self.calls == 1:
            raise subprocess.CalledProcessError(1, ["cmd"], stderr=b"fail")
        return types.SimpleNamespace(package=None, roi=types.SimpleNamespace(roi=0.0))


class StubImprovementEngine:
    def __init__(self, pipeline: Pipeline, tracker: DummyTracker):
        self.pipeline = pipeline
        self.tracker = tracker
        self.synergy_weight_roi = 1.0

    def run_cycle(self):
        self.pipeline.run("menace")
        self.synergy_weight_roi += 0.1


def test_full_cycle_failure_and_recovery(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.ERROR)

    # ---- SelfTestService: missing dependency then recovery ----
    db = DummyDB()
    svc = sts.SelfTestService(db=db, history_path=tmp_path / "hist.json")
    mod = tmp_path / "test_mod.py"  # path-ignore
    mod.write_text("def test_a():\n    assert True\n")

    def run_missing(cmd, **kwargs):
        raise FileNotFoundError("pytest missing")

    monkeypatch.setattr(subprocess, "run", run_missing)
    passed, warnings, metrics = svc._run_module_harness(mod.as_posix())
    assert not passed

    def run_ok(cmd, **kwargs):
        class P:
            returncode = 0
            stdout = b""
            stderr = b""
        return P()

    monkeypatch.setattr(subprocess, "run", run_ok)
    passed, warnings, metrics = svc._run_module_harness(mod.as_posix())
    assert passed
    assert "coverage" in metrics

    # ---- SelfDebuggerSandbox: failing subprocess then recovery ----
    engine = FlowEngine()
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), engine, context_builder=DummyBuilder(), audit_trail=trail
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dbg, "_generate_tests", lambda logs: ["def test_ok():\n    assert True\n"])
    monkeypatch.setattr(dbg, "_test_flakiness", lambda p, env=None, *, runs=None: 0.0)
    monkeypatch.setattr(dbg, "_code_complexity", lambda p: 0.0)

    fail = {"first": True}

    def run_tests(path, env=None):
        if fail["first"]:
            fail["first"] = False
            raise sds.CoverageSubprocessError("boom")
        return 80.0, 0.0

    monkeypatch.setattr(dbg, "_run_tests", run_tests)

    dbg.analyse_and_fix()
    assert "sandbox tests failed" in caplog.text
    caplog.clear()
    dbg.analyse_and_fix()
    recs = [json.loads(r) for r in trail.records]
    assert recs[-1]["result"] == "success"

    # ---- SelfImprovementEngine: failing pipeline then recovery ----
    tracker = DummyTracker()
    eng = StubImprovementEngine(Pipeline(), tracker)

    with pytest.raises(subprocess.CalledProcessError):
        eng.run_cycle()
    caplog.clear()
    tracker.advance()
    eng.run_cycle()
    assert eng.synergy_weight_roi > 1.0

