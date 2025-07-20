import json
from pathlib import Path

from tests.test_self_debugger_sandbox import sds, DummyTelem, DummyEngine, DummyTrail


class RB:
    def __init__(self):
        self.calls = []

    def rollback(self, pid):
        self.calls.append(pid)


class FlowEngine(DummyEngine):
    def __init__(self):
        super().__init__()
        self.rollback_mgr = RB()
        self.rollback_patch_calls = []
        self.patched = []

    def apply_patch(self, path: Path, desc: str):
        pid = len(self.patched) + 1
        self.patched.append(pid)
        return pid, False, 0.0

    def rollback_patch(self, pid: str) -> None:
        self.rollback_patch_calls.append(pid)


def _setup_dbg(monkeypatch, tmp_path, coverage_vals, error_vals):
    engine = FlowEngine()
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), engine, audit_trail=trail)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dbg, "_generate_tests", lambda logs: ["def test_ok():\n    pass\n"])
    monkeypatch.setattr(sds.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(dbg, "_test_flakiness", lambda p, env=None, *, runs=None: 0.0)
    monkeypatch.setattr(dbg, "_code_complexity", lambda p: 0.0)
    monkeypatch.setattr(dbg, "_run_tests", lambda p, env=None: (coverage_vals.pop(0), 0.0))
    monkeypatch.setattr(engine, "_current_errors", lambda: error_vals.pop(0), raising=False)
    return dbg, engine, trail


def test_good_patch_kept(monkeypatch, tmp_path):
    cov = [60.0, 80.0, 60.0, 80.0]
    errs = [1, 0, 1, 0]
    dbg, engine, trail = _setup_dbg(monkeypatch, tmp_path, cov, errs)
    dbg.analyse_and_fix()

    assert engine.rollback_mgr.calls == []
    assert engine.rollback_patch_calls == ["1"]
    recs = [json.loads(r) for r in trail.records]
    assert recs[-1]["result"] == "success"
    assert "score" in recs[-1]
    assert recs[-1]["score"] > 0.5


def test_bad_patch_rolled_back(monkeypatch, tmp_path):
    cov = [60.0, 50.0, 60.0, 50.0]
    errs = [0, 1, 0, 1]
    dbg, engine, trail = _setup_dbg(monkeypatch, tmp_path, cov, errs)
    dbg.analyse_and_fix()

    assert engine.rollback_mgr.calls == ["1"]
    assert engine.rollback_patch_calls == []
    recs = [json.loads(r) for r in trail.records]
    assert recs[-1]["result"] == "reverted"
    assert "score" in recs[-1]
    assert recs[-1]["score"] < 0.5


def test_composite_score_prefers_better(monkeypatch):
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), DummyEngine())
    score_low = dbg._composite_score(0.05, 0.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0)
    score_high = dbg._composite_score(0.2, 0.2, 0.3, 0.0, 0.0, 0.1, 0.0, 0.0)
    assert score_high > score_low
