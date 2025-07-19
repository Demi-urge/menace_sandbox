from pathlib import Path
import logging
import sys
import types
import importlib
import json
import asyncio
import asyncio
import asyncio

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
sys.modules.setdefault("sqlalchemy", types.ModuleType("sqlalchemy"))
sys.modules.setdefault("sqlalchemy.engine", types.ModuleType("engine"))
cov_mod = types.ModuleType("coverage")
cov_mod.Coverage = object
sys.modules.setdefault("coverage", cov_mod)
import os
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

# Create a lightweight menace package with stub modules used during import
menace_pkg = types.ModuleType("menace")
menace_pkg.__path__ = []
sys.modules.setdefault("menace", menace_pkg)

auto_dbg = types.ModuleType("menace.automated_debugger")
class AutomatedDebugger:
    def __init__(self, telemetry_db, engine):
        import logging as _logging
        self.telemetry_db = telemetry_db
        self.engine = engine
        self.logger = _logging.getLogger("AutomatedDebugger")

    def _generate_tests(self, logs):
        return []

    def _recent_logs(self, limit: int = 5):
        return ["dummy"]
auto_dbg.AutomatedDebugger = AutomatedDebugger
sys.modules.setdefault("menace.automated_debugger", auto_dbg)

sce_mod = types.ModuleType("menace.self_coding_engine")
class SelfCodingEngine: ...
sce_mod.SelfCodingEngine = SelfCodingEngine
sys.modules.setdefault("menace.self_coding_engine", sce_mod)

audit_mod = types.ModuleType("menace.audit_trail")
class AuditTrail: ...
audit_mod.AuditTrail = AuditTrail
sys.modules.setdefault("menace.audit_trail", audit_mod)

cd_mod = types.ModuleType("menace.code_database")
def _hash_code(data: bytes) -> str: return "x"
cd_mod._hash_code = _hash_code
class PatchHistoryDB: ...
cd_mod.PatchHistoryDB = PatchHistoryDB
sys.modules.setdefault("menace.code_database", cd_mod)

pol_mod = types.ModuleType("menace.self_improvement_policy")
class SelfImprovementPolicy: ...
pol_mod.SelfImprovementPolicy = SelfImprovementPolicy
sys.modules.setdefault("menace.self_improvement_policy", pol_mod)

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
    monkeypatch.setattr(dbg, "_coverage_percent", lambda p, env=None: 50.0)
    monkeypatch.setattr(dbg, "_test_flakiness", lambda p, env=None: 0.0)
    monkeypatch.setattr(dbg, "_code_complexity", lambda p: 0.0)

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

    async def fake_exec(*a, **k):
        class P:
            returncode = 0

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(sds.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def fake_exec(*a, **k):
        class P:
            returncode = 0

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(sds.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def fake_exec(*a, **k):
        class P:
            returncode = 0

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(sds.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def fake_exec(*a, **k):
        class P:
            returncode = 0

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(sds.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def fake_exec(*a, **k):
        class P:
            returncode = 0

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(sds.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(dbg, "_test_flakiness", lambda p, env=None: 0.0)
    monkeypatch.setattr(dbg, "_code_complexity", lambda p: 0.0)
    monkeypatch.setattr(dbg, "_coverage_percent", lambda p, env=None: 80.0)
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
    monkeypatch.setattr(dbg, "_coverage_percent", lambda p, env=None: 80.0)
    monkeypatch.setattr(dbg, "_test_flakiness", lambda p, env=None: 0.0)
    monkeypatch.setattr(dbg, "_code_complexity", lambda p: 0.0)
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


def test_log_patch_records_extra_fields():
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), DummyEngine(), audit_trail=trail)
    dbg._log_patch(
        "desc",
        "success",
        1.0,
        1.0,
        coverage_delta=0.1,
        error_delta=0.0,
        roi_delta=0.0,
        score=0.9,
        flakiness=0.2,
        runtime_impact=0.05,
        complexity=1.0,
    )
    rec = json.loads(trail.records[-1])
    assert rec["flakiness"] == 0.2
    assert rec["runtime_impact"] == 0.05
    assert rec["complexity"] == 1.0


def test_coverage_drop_reverts(monkeypatch, tmp_path):
    class RB:
        def __init__(self):
            self.calls = []

        def rollback(self, pid):
            self.calls.append(pid)

    class CovEngine(DummyEngine):
        def __init__(self):
            super().__init__()
            self.rollback_mgr = RB()

    engine = CovEngine()
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), engine, audit_trail=trail)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dbg, "_generate_tests", lambda logs: ["def test_ok():\n    assert True\n"])
    monkeypatch.setattr(sds.subprocess, "run", lambda *a, **k: None)

    coverage_vals = [80.0, 50.0]

    class FakeCov:
        def start(self):
            pass

        def stop(self):
            pass

        def combine(self, files):
            pass

        def report(self, include=None, file=None):
            return coverage_vals.pop(0)

        def xml_report(self, outfile=None, include=None):
            return 0

    monkeypatch.setattr(sds, "Coverage", lambda *a, **k: FakeCov())

    dbg.analyse_and_fix()

    assert engine.rollback_mgr.calls == ["1"]
    assert any("reverted" in r for r in trail.records)


def test_select_best_patch(monkeypatch, tmp_path):
    class ScoreEngine(DummyEngine):
        def __init__(self):
            super().__init__()
            self.deltas = [0.1, 0.5, 0.5]
            self.calls = []

        def apply_patch(self, path: Path, desc: str):
            delta = self.deltas.pop(0)
            self.calls.append(delta)
            return len(self.calls), False, delta

        def rollback_patch(self, pid: str) -> None:
            pass

    engine = ScoreEngine()
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), engine)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dbg, "_generate_tests", lambda logs: ["def a():\n    pass\n", "def b():\n    pass\n"])
    monkeypatch.setattr(sds.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(dbg, "_test_flakiness", lambda p, env=None: 0.0)
    monkeypatch.setattr(dbg, "_code_complexity", lambda p: 0.0)

    cov_vals = [50.0, 60.0, 50.0, 70.0, 50.0, 80.0]
    monkeypatch.setattr(dbg, "_coverage_percent", lambda p, env=None: cov_vals.pop(0))

    dbg.analyse_and_fix()

    assert engine.calls[-1] == 0.5


def test_run_tests_includes_telemetry(monkeypatch, tmp_path):
    engine = DummyEngine()
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), engine)
    monkeypatch.chdir(tmp_path)

    base = Path("test_base.py")
    base.write_text("def test_base():\n    assert True\n")

    called = {}

    def fake_cov(paths, env=None):
        called["paths"] = [Path(p) for p in paths]
        return 100.0

    monkeypatch.setattr(dbg, "_coverage_percent", fake_cov)
    monkeypatch.setattr(dbg, "_recent_logs", lambda limit=5: ["dummy"])
    monkeypatch.setattr(dbg, "_generate_tests", lambda logs: ["def test_extra():\n    pass\n"])

    dbg._run_tests(base)

    assert len(called.get("paths", [])) == 2
    assert any("telemetry" in p.name for p in called["paths"])


def _stub_proc():
    class P:
        returncode = 0

        async def wait(self):
            return None

    return P()


def test_coverage_xml_report_failure(monkeypatch, caplog):
    engine = DummyEngine()
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), engine)
    caplog.set_level(logging.ERROR)

    async def fake_exec(*a, **k):
        return _stub_proc()

    monkeypatch.setattr(sds.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    class Cov:
        def __init__(self, *a, **k):
            pass

        def combine(self, files):
            pass

        def xml_report(self, outfile=None, include=None):
            raise RuntimeError("boom")

        def report(self, include=None, file=None):
            return 100.0

    monkeypatch.setattr(sds, "Coverage", Cov)

    cov = dbg._coverage_percent([Path("dummy.py")])
    assert cov == 0.0
    assert "coverage generation failed" in caplog.text


def test_coverage_report_failure(monkeypatch, caplog):
    engine = DummyEngine()
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), engine)
    caplog.set_level(logging.ERROR)

    async def fake_exec(*a, **k):
        return _stub_proc()

    monkeypatch.setattr(sds.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    class Cov:
        def __init__(self, *a, **k):
            pass

        def combine(self, files):
            pass

        def xml_report(self, outfile=None, include=None):
            return 0

        def report(self, include=None, file=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(sds, "Coverage", Cov)

    cov = dbg._coverage_percent([Path("dummy.py")])
    assert cov == 0.0
    assert "coverage generation failed" in caplog.text
