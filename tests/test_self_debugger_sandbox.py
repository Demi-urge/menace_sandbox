from pathlib import Path
import logging
import sys
import types

sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519", types.ModuleType("ed25519")
)
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///:memory:"))
sys.modules.setdefault("httpx", types.ModuleType("httpx"))

import importlib.util
from pathlib import Path as _P

_spec = importlib.util.spec_from_file_location(
    "menace.self_debugger_sandbox",
    _P(__file__).resolve().parents[1] / "self_debugger_sandbox.py",
)
sds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sds)
sys.modules["menace.self_debugger_sandbox"] = sds

class DummyTelem:
    def recent_errors(self, limit: int = 5):
        return ["dummy"]

class DummyEngine:
    def __init__(self):
        self.applied = False
        self.rollback_mgr = None

    def generate_helper(self, desc: str) -> str:
        return "def helper():\n    pass\n"

    def patch_file(self, path: Path, desc: str) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(self.generate_helper(desc))

    def apply_patch(self, path: Path, desc: str):
        self.applied = True
        return 1, False, 0.0


class DummyTrail:
    def __init__(self, path=None):
        self.records = []

    def record(self, msg):
        self.records.append(msg)


def test_sandbox_failing_patch(monkeypatch, tmp_path):
    engine = DummyEngine()
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), engine)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dbg, "_generate_tests", lambda logs: ["def test_fail():\n    assert False\n"])

    def run_fail(cmd, cwd=None, check=False, env=None):
        raise RuntimeError("fail")

    monkeypatch.setattr(sds.subprocess, "run", run_fail)
    dbg.analyse_and_fix()
    assert not engine.applied


def test_sandbox_success(monkeypatch, tmp_path):
    engine = DummyEngine()
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), engine, audit_trail=trail)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dbg, "_generate_tests", lambda logs: ["def test_ok():\n    assert True\n"])
    monkeypatch.setattr(sds.subprocess, "run", lambda *a, **k: None)
    dbg.analyse_and_fix()
    assert engine.applied
    assert any("success" in r for r in trail.records)


def test_sandbox_failed_audit(monkeypatch, tmp_path):
    class FailEngine(DummyEngine):
        def apply_patch(self, path: Path, desc: str):
            raise RuntimeError("boom")

    engine = FailEngine()
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), engine, audit_trail=trail)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dbg, "_generate_tests", lambda logs: ["def test_ok():\n   assert True\n"])
    monkeypatch.setattr(sds.subprocess, "run", lambda *a, **k: None)
    dbg.analyse_and_fix()
    assert not engine.applied
    assert any("failed" in r for r in trail.records)


def test_log_patch_logs_audit_error(caplog):
    class FailTrail(DummyTrail):
        def record(self, msg):
            raise RuntimeError("boom")

    dbg = sds.SelfDebuggerSandbox(DummyTelem(), DummyEngine(), audit_trail=FailTrail())
    caplog.set_level(logging.ERROR)
    dbg._log_patch("desc", "res")
    assert "audit trail logging failed" in caplog.text
