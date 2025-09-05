import logging
import subprocess
from pathlib import Path
import pytest

# Attempt to import SelfDebuggerSandbox; skip tests if unavailable
import logging
import subprocess
from pathlib import Path

import pytest
import sys
import types

run_auto = types.ModuleType("run_autonomous")
run_auto.LOCAL_KNOWLEDGE_MODULE = None
sys.modules.setdefault("run_autonomous", run_auto)
sys.modules.setdefault("menace_sandbox.run_autonomous", run_auto)

try:
    from menace_sandbox.self_debugger_sandbox import SelfDebuggerSandbox
except BaseException:  # pragma: no cover - import may fail due to heavy deps
    SelfDebuggerSandbox = None
try:
    from menace_sandbox.self_test_service import SelfTestService
except BaseException:  # pragma: no cover
    SelfTestService = None
try:
    from menace_sandbox.self_improvement import SelfImprovementEngine
except BaseException:  # pragma: no cover
    SelfImprovementEngine = None


# -------- self_debugger_sandbox --------

@pytest.mark.skipif(SelfDebuggerSandbox is None, reason="self_debugger_sandbox unavailable")
def test_sandbox_subprocess_failure(monkeypatch, caplog):
    sandbox = SelfDebuggerSandbox.__new__(SelfDebuggerSandbox)
    sandbox.logger = logging.getLogger("test")
    sandbox.engine = type("E", (), {"patch_file": lambda self, p, m: None})()
    sandbox._bad_hashes = set()
    sandbox.flakiness_runs = 1
    sandbox._test_timeout = 1
    sandbox._record_exception = lambda exc: None
    sandbox._recent_logs = lambda: ["log"]
    sandbox._generate_tests = lambda logs: ["print('hi')"]
    sandbox.preemptive_fix_high_risk_modules = lambda: None
    sandbox._test_flakiness = lambda *a, **k: 0.0
    sandbox._code_complexity = lambda p: 0.0
    sandbox._run_tests = lambda p: (0.0, 0.0)
    sandbox._recent_synergy_metrics = lambda tracker: (0.0, 0.0, 0.0, 0.0)
    sandbox._composite_score = lambda *a, **k: 0.0
    sandbox._log_patch = lambda *a, **k: None
    sandbox.state_getter = lambda: ()
    sandbox.policy = None

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with caplog.at_level(logging.ERROR):
        sandbox.analyse_and_fix()
    assert "sandbox tests failed" in caplog.text


@pytest.mark.skipif(SelfDebuggerSandbox is None, reason="self_debugger_sandbox unavailable")
def test_sandbox_subprocess_timeout(monkeypatch, caplog):
    sandbox = SelfDebuggerSandbox.__new__(SelfDebuggerSandbox)
    sandbox.logger = logging.getLogger("test")
    sandbox.engine = type("E", (), {"patch_file": lambda self, p, m: None})()
    sandbox._bad_hashes = set()
    sandbox.flakiness_runs = 1
    sandbox._test_timeout = 1
    sandbox._record_exception = lambda exc: None
    sandbox._recent_logs = lambda: ["log"]
    sandbox._generate_tests = lambda logs: ["print('hi')"]
    sandbox.preemptive_fix_high_risk_modules = lambda: None
    sandbox._test_flakiness = lambda *a, **k: 0.0
    sandbox._code_complexity = lambda p: 0.0
    sandbox._run_tests = lambda p: (0.0, 0.0)
    sandbox._recent_synergy_metrics = lambda tracker: (0.0, 0.0, 0.0, 0.0)
    sandbox._composite_score = lambda *a, **k: 0.0
    sandbox._log_patch = lambda *a, **k: None
    sandbox.state_getter = lambda: ()
    sandbox.policy = None

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout=1, stderr="late")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with caplog.at_level(logging.ERROR):
        sandbox.analyse_and_fix()
    assert "sandbox tests timed out" in caplog.text


# -------- self_test_service --------

def _make_service(tmp_path):
    svc = SelfTestService.__new__(SelfTestService)
    svc.logger = logging.getLogger("test")
    svc.test_runner = "pytest"
    svc.stub_scenarios = {}
    svc.orphan_traces = {}
    svc._has_pytest_file = lambda mod: True
    svc.container_timeout = 1
    return svc


@pytest.mark.skipif(SelfTestService is None, reason="self_test_service unavailable")
def test_service_subprocess_failure(monkeypatch, tmp_path, caplog):
    svc = _make_service(tmp_path)
    mod = tmp_path / "test_mod.py"  # path-ignore
    mod.write_text("def test_a():\n    assert True\n")

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="bad")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with caplog.at_level(logging.ERROR):
        passed, warnings, metrics = svc._run_module_harness(mod.as_posix())
    assert not passed and warnings == [] and metrics == {}
    assert "module harness failed" in caplog.text


@pytest.mark.skipif(SelfTestService is None, reason="self_test_service unavailable")
def test_service_subprocess_timeout(monkeypatch, tmp_path, caplog):
    svc = _make_service(tmp_path)
    mod = tmp_path / "test_mod.py"  # path-ignore
    mod.write_text("def test_a():\n    assert True\n")

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout=1, stderr="slow")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with caplog.at_level(logging.ERROR):
        passed, warnings, metrics = svc._run_module_harness(mod.as_posix())
    assert not passed and warnings == [] and metrics == {}
    assert "module harness timed out" in caplog.text


# -------- self_improvement_engine --------

def _make_engine():
    eng = SelfImprovementEngine.__new__(SelfImprovementEngine)
    eng.logger = logging.getLogger("test")
    eng.alignment_flagger = type("F", (), {"flag_patch": lambda self, patch, ctx: {}})()
    return eng


@pytest.mark.skipif(SelfImprovementEngine is None, reason="self_improvement_engine unavailable")
def test_improvement_engine_subprocess_failure(monkeypatch, caplog):
    eng = _make_engine()

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="err")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with caplog.at_level(logging.ERROR):
        eng._alignment_review_last_commit("desc")
    assert "git command failed" in caplog.text


@pytest.mark.skipif(SelfImprovementEngine is None, reason="self_improvement_engine unavailable")
def test_improvement_engine_subprocess_timeout(monkeypatch, caplog):
    eng = _make_engine()

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout=1, stderr="late")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with caplog.at_level(logging.ERROR):
        eng._alignment_review_last_commit("desc")
    assert "git command timed out" in caplog.text


@pytest.mark.skipif(SelfImprovementEngine is None, reason="self_improvement_engine unavailable")
def test_improvement_engine_subprocess_unexpected(monkeypatch, caplog):
    eng = _make_engine()

    def fake_run(cmd, **kwargs):
        raise OSError("bad")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with caplog.at_level(logging.ERROR):
        eng._alignment_review_last_commit("desc")
    assert "git command unexpected failure" in caplog.text


@pytest.mark.skipif(SelfImprovementEngine is None, reason="self_improvement_engine unavailable")
def test_improvement_engine_flagger_failure(monkeypatch, caplog):
    eng = _make_engine()

    def fake_run(cmd, **kwargs):
        return types.SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    class BadFlagger:
        def flag_patch(self, patch, ctx):
            raise RuntimeError("boom")

    eng.alignment_flagger = BadFlagger()
    with caplog.at_level(logging.ERROR):
        eng._alignment_review_last_commit("desc")
    assert "alignment flagger failed" in caplog.text


@pytest.mark.skipif(SelfImprovementEngine is None, reason="self_improvement_engine unavailable")
def test_flag_patch_alignment_subprocess_unexpected(monkeypatch, caplog):
    eng = _make_engine()

    def fake_run(cmd, **kwargs):
        raise OSError("bad")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with caplog.at_level(logging.ERROR):
        eng._flag_patch_alignment(1, {})
    assert "git command unexpected failure" in caplog.text


@pytest.mark.skipif(SelfImprovementEngine is None, reason="self_improvement_engine unavailable")
def test_flag_patch_alignment_flagger_failure(monkeypatch, caplog):
    eng = _make_engine()

    def fake_run(cmd, **kwargs):
        return types.SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    class BadFlagger:
        def flag_patch(self, patch, ctx):
            raise RuntimeError("boom")

    eng.alignment_flagger = BadFlagger()
    with caplog.at_level(logging.ERROR):
        eng._flag_patch_alignment(1, {})
    assert "alignment flagging failed" in caplog.text
