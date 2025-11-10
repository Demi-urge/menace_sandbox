from pathlib import Path
import logging
import sys
import types
import importlib
import pytest
import json
import asyncio
import time
import dynamic_path_router
import subprocess
from contextlib import contextmanager

sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
)
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
sys.modules.setdefault(
    "env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///:memory:")
)
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
sys.modules.setdefault("sqlalchemy", types.ModuleType("sqlalchemy"))
sys.modules.setdefault("sqlalchemy.engine", types.ModuleType("engine"))
cov_mod = types.ModuleType("coverage")
cov_mod.Coverage = object
sys.modules.setdefault("coverage", cov_mod)
sklearn_mod = types.ModuleType("sklearn")
linear_mod = types.ModuleType("linear_model")


class DummyLR:
    def fit(self, *a, **k):
        return self

    def predict(self, X):
        return [0.0 for _ in range(len(X))]


linear_mod.LinearRegression = DummyLR
pre_mod = types.ModuleType("preprocessing")


class DummyPF:
    def __init__(self, *a, **k):
        pass


pre_mod.PolynomialFeatures = DummyPF
sklearn_mod.linear_model = linear_mod
sklearn_mod.preprocessing = pre_mod
sys.modules.setdefault("sklearn", sklearn_mod)
sys.modules.setdefault("sklearn.linear_model", linear_mod)
sys.modules.setdefault("sklearn.preprocessing", pre_mod)
import os

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

# Stub external dependency used during sandbox imports


class _NeuroStub(types.SimpleNamespace):
    def __getattr__(self, name):  # pragma: no cover - simple default
        return lambda *a, **k: [] if name.startswith("get") else None


sys.modules.setdefault("neurosales", _NeuroStub())

# Additional stubs to avoid heavy imports
sys.modules.setdefault(
    "environment_bootstrap", types.SimpleNamespace(EnvironmentBootstrapper=object)
)
sys.modules.setdefault(
    "light_bootstrap", types.SimpleNamespace(EnvironmentBootstrapper=object)
)
sys.modules.setdefault(
    "vector_service.embedding_scheduler",
    types.SimpleNamespace(start_scheduler_from_env=lambda *a, **k: None),
)
sys.modules.setdefault(
    "unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object)
)
sys.modules.setdefault(
    "automated_reviewer", types.SimpleNamespace(AutomatedReviewer=object)
)
sys.modules.setdefault(
    "jsonschema", types.SimpleNamespace(ValidationError=Exception, validate=lambda *a, **k: None)
)
sys.modules.setdefault(
    "quick_fix_engine", types.SimpleNamespace(generate_patch=lambda *a, **k: None)
)
sys.modules.setdefault("menace.quick_fix_engine", sys.modules["quick_fix_engine"])
import patch_score_backend as _psb
sys.modules.setdefault("menace.patch_score_backend", _psb)

# Create a lightweight menace package with stub modules used during import
menace_pkg = types.ModuleType("menace")
menace_pkg.__path__ = []
sys.modules.setdefault("menace", menace_pkg)
import logging_utils as _logging_utils
sys.modules.setdefault("menace.logging_utils", _logging_utils)


@contextmanager
def _fake_env(workdir, *, context_builder):
    def _run(cmd, *, env=None, capture_output=False, text=False, **_kw):
        stdout = ""
        if (
            capture_output
            and text
            and len(cmd) >= 2
            and cmd[0] in {"python", sys.executable}
            and cmd[1].startswith("-c")
        ):
            stdout = sys.executable
        return subprocess.CompletedProcess(cmd, 0, stdout, "")

    yield workdir, _run, sys.executable


cfg_stub = types.SimpleNamespace(
    get_impact_severity=lambda *a, **k: {},
    impact_severity_map=lambda *a, **k: {},
)
sys.modules.setdefault("menace.config_loader", cfg_stub)
sys.modules.setdefault("config_loader", cfg_stub)
sys.modules.setdefault(
    "menace.workflow_run_summary",
    types.SimpleNamespace(record_run=lambda *a, **k: None, save_all_summaries=lambda *a, **k: None),
)
sys.modules.setdefault(
    "workflow_run_summary",
    types.SimpleNamespace(record_run=lambda *a, **k: None, save_all_summaries=lambda *a, **k: None),
)
sys.modules.setdefault("menace.telemetry_backend", types.SimpleNamespace(TelemetryBackend=object))
sys.modules.setdefault("telemetry_backend", types.SimpleNamespace(TelemetryBackend=object))
sys.modules.setdefault("menace.borderline_bucket", types.SimpleNamespace(BorderlineBucket=object))
sys.modules.setdefault("borderline_bucket", types.SimpleNamespace(BorderlineBucket=object))
sys.modules.setdefault("menace.truth_adapter", types.SimpleNamespace(TruthAdapter=object))
sys.modules.setdefault("truth_adapter", types.SimpleNamespace(TruthAdapter=object))
sys.modules.setdefault(
    "menace.roi_calculator", types.SimpleNamespace(ROICalculator=object, propose_fix=lambda *a, **k: None)
)
sys.modules.setdefault(
    "roi_calculator", types.SimpleNamespace(ROICalculator=object, propose_fix=lambda *a, **k: None)
)
sys.modules.setdefault("menace.readiness_index", types.SimpleNamespace(compute_readiness=lambda *a, **k: 0.0))
sys.modules.setdefault(
    "readiness_index", types.SimpleNamespace(compute_readiness=lambda *a, **k: 0.0)
)

rt = types.ModuleType("roi_tracker")
sys.modules.setdefault("roi_tracker", rt)
sys.modules.setdefault("menace.roi_tracker", rt)

auto_dbg = types.ModuleType("menace.automated_debugger")


class AutomatedDebugger:
    def __init__(self, telemetry_db, engine, context_builder):
        import logging as _logging

        self.telemetry_db = telemetry_db
        self.engine = engine
        self.context_builder = context_builder
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


def _hash_code(data: bytes) -> str:
    return "x"


cd_mod._hash_code = _hash_code


class PatchHistoryDB:
    def __init__(self, path=None):
        self.path = Path(path or "patch.db")
        self.records = []
        self.flaky = []
        self.weights = None

    def add(self, rec):
        self.records.append(rec)

    def filter(self, filename=None, reverted=None):
        return list(self.records)

    def by_hash(self, code_hash):
        return [r for r in self.records if getattr(r, "code_hash", None) == code_hash]

    def record_flakiness(self, filename: str, flakiness: float) -> None:
        self.flaky.append((filename, flakiness))

    def average_flakiness(self, filename: str, limit: int = 20) -> float:
        vals = [f for f_name, f in self.flaky if f_name == filename]
        return float(sum(vals) / len(vals)) if vals else 0.0

    def store_weights(self, weights):
        self.weights = tuple(weights)

    def get_weights(self):
        return self.weights


cd_mod.PatchHistoryDB = PatchHistoryDB
sys.modules.setdefault("menace.code_database", cd_mod)
sys.modules.setdefault("code_database", cd_mod)

pol_mod = types.ModuleType("menace.self_improvement_policy")


class SelfImprovementPolicy: ...


pol_mod.SelfImprovementPolicy = SelfImprovementPolicy
sys.modules.setdefault("menace.self_improvement_policy", pol_mod)

import importlib.util
from pathlib import Path as _P

_spec = importlib.util.spec_from_file_location(
    "menace.self_debugger_sandbox",
    dynamic_path_router.path_for_prompt("self_debugger_sandbox.py"),  # path-ignore
)
sds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sds)
sys.modules["menace.self_debugger_sandbox"] = sds

class DummyManager:
    def run_patch(self, *a, **k):
        pass

from functools import partial

sds.SelfDebuggerSandbox = partial(sds.SelfDebuggerSandbox, manager=DummyManager())

# Provide a minimal error logger stub to avoid heavy dependencies during tests
sds.ErrorLogger = lambda *a, **k: types.SimpleNamespace(record=lambda *a, **k: None)
sds.create_ephemeral_env = _fake_env
sds.generate_edge_cases = lambda: {}


class DummyBuilder:
    def build_context(self, query: str, **kwargs):
        return {}
    def exclude_failed_strategies(self, tags):
        pass
    def query(self, *a, **k):
        return [], {}
    def refresh_db_weights(self):
        pass


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

    def apply_patch(self, path: Path, desc: str, **_: object):
        self.applied = True
        return 1, False, 0.0


class DummyTrail:
    def __init__(self, path=None):
        self.records = []

    def record(self, msg):
        self.records.append(msg)


def test_no_global_builder():
    assert not hasattr(sds, 'CONTEXT_BUILDER')


def test_builder_injected_and_refreshed():
    class B(DummyBuilder):
        def __init__(self):
            self.refreshed = False

        def refresh_db_weights(self):
            self.refreshed = True

    b = B()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=b
    )
    assert dbg.context_builder is b
    assert b.refreshed


def test_baseline_config_from_env():
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(),
        DummyEngine(),
        context_builder=DummyBuilder(),
        baseline_window=7,
        stagnation_iters=3,
        delta_margin=0.8,
        score_weights=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    )
    assert dbg.delta_margin == 0.8
    assert dbg._baseline_tracker.composite_history.maxlen == 7
    assert dbg.stagnation_iters == 3
    assert dbg.score_weights == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)


def test_sandbox_failing_patch(monkeypatch, tmp_path):
    engine = DummyEngine()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), engine, context_builder=DummyBuilder()
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        dbg, "_generate_tests", lambda logs: ["def test_fail():\n    assert False\n"]
    )

    async def fake_cov(p, env=None, **kwargs):
        return 50.0, {}

    monkeypatch.setattr(dbg, "_coverage_percent", fake_cov)
    monkeypatch.setattr(dbg, "_test_flakiness", lambda p, env=None, *, runs=None: 0.0)
    monkeypatch.setattr(dbg, "_code_complexity", lambda p: 0.0)

    def run_fail(cmd, cwd=None, check=False, env=None):
        raise RuntimeError("fail")

    monkeypatch.setattr(sds.subprocess, "run", run_fail)
    dbg.analyse_and_fix()
    assert not engine.applied


def test_sandbox_success(monkeypatch, tmp_path):
    engine = DummyEngine()
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), engine, context_builder=DummyBuilder(), audit_trail=trail
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        dbg, "_generate_tests", lambda logs: ["def test_ok():\n    assert True\n"]
    )
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
    monkeypatch.setattr(dbg, "_test_flakiness", lambda p, env=None, *, runs=None: 0.0)
    monkeypatch.setattr(dbg, "_code_complexity", lambda p: 0.0)

    async def fake_cov_ok(p, env=None, **kwargs):
        return 80.0, {}

    monkeypatch.setattr(dbg, "_coverage_percent", fake_cov_ok)
    dbg.analyse_and_fix()
    assert engine.applied
    assert any("success" in r for r in trail.records)


def test_sandbox_failed_audit(monkeypatch, tmp_path):
    class FailEngine(DummyEngine):
        def apply_patch(self, path: Path, desc: str, **_: object):
            raise RuntimeError("boom")

    engine = FailEngine()
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), engine, context_builder=DummyBuilder(), audit_trail=trail
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        dbg, "_generate_tests", lambda logs: ["def test_ok():\n   assert True\n"]
    )
    monkeypatch.setattr(sds.subprocess, "run", lambda *a, **k: None)

    async def fake_cov_ok(p, env=None, **kwargs):
        return 80.0, {}

    monkeypatch.setattr(dbg, "_coverage_percent", fake_cov_ok)
    monkeypatch.setattr(dbg, "_test_flakiness", lambda p, env=None, *, runs=None: 0.0)
    monkeypatch.setattr(dbg, "_code_complexity", lambda p: 0.0)
    dbg.analyse_and_fix()
    assert not engine.applied
    assert any("failed" in r for r in trail.records)


def test_log_patch_logs_audit_error(caplog):
    class FailTrail(DummyTrail):
        def record(self, msg):
            raise RuntimeError("boom")

    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder(), audit_trail=FailTrail()
    )
    caplog.set_level(logging.ERROR)
    dbg._log_patch("desc", "res")
    assert "audit trail logging failed" in caplog.text


def test_log_patch_records_extra_fields():
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder(), audit_trail=trail
    )
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
        synergy_resilience=0.3,
        synergy_antifragility=0.4,
    )
    rec = json.loads(trail.records[-1])
    assert rec["flakiness"] == 0.2
    assert rec["runtime_impact"] == 0.05
    assert rec["complexity"] == 1.0
    assert rec["synergy_resilience"] == 0.3
    assert rec["synergy_antifragility"] == 0.4


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
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), engine, context_builder=DummyBuilder(), audit_trail=trail
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        dbg, "_generate_tests", lambda logs: ["def test_ok():\n    assert True\n"]
    )
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

        def apply_patch(self, path: Path, desc: str, **_: object):
            delta = self.deltas.pop(0)
            self.calls.append(delta)
            return len(self.calls), False, delta

        def rollback_patch(self, pid: str) -> None:
            pass

    engine = ScoreEngine()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), engine, context_builder=DummyBuilder()
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        dbg,
        "_generate_tests",
        lambda logs: ["def a():\n    pass\n", "def b():\n    pass\n"],
    )
    monkeypatch.setattr(sds.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(dbg, "_test_flakiness", lambda p, env=None, *, runs=None: 0.0)
    monkeypatch.setattr(dbg, "_code_complexity", lambda p: 0.0)

    cov_vals = [50.0, 60.0, 50.0, 70.0, 50.0, 80.0]

    async def fake_cov(p, env=None, **kwargs):
        return cov_vals.pop(0), {}

    monkeypatch.setattr(dbg, "_coverage_percent", fake_cov)

    dbg.analyse_and_fix()

    assert engine.calls[-1] == 0.5


def test_run_tests_includes_telemetry(monkeypatch, tmp_path):
    engine = DummyEngine()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), engine, context_builder=DummyBuilder()
    )
    monkeypatch.chdir(tmp_path)

    base = Path("test_base.py")  # path-ignore
    base.write_text("def test_base():\n    assert True\n")

    called = {}

    async def fake_cov(paths, env=None, **kwargs):
        called["paths"] = [Path(p) for p in paths]
        return 100.0, {}

    monkeypatch.setattr(dbg, "_coverage_percent", fake_cov)
    monkeypatch.setattr(dbg, "_recent_logs", lambda limit=5: ["dummy"])
    monkeypatch.setattr(
        dbg, "_generate_tests", lambda logs: ["def test_extra():\n    pass\n"]
    )

    dbg._run_tests(base)

    assert len(called.get("paths", [])) == 2
    assert any("telemetry" in p.name for p in called["paths"])


def _stub_proc(rc=0, out=b"", err=b""):
    class P:
        def __init__(self):
            self.returncode = rc

        async def communicate(self):
            return out, err

    return P()


def test_coverage_xml_report_failure(monkeypatch, caplog):
    engine = DummyEngine()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), engine, context_builder=DummyBuilder()
    )
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

    percent, _ = asyncio.run(dbg._coverage_percent([Path("dummy.py")]))  # path-ignore
    assert percent == 0.0
    assert "coverage generation failed" in caplog.text
    assert "boom" in caplog.text


def test_coverage_report_failure(monkeypatch, caplog):
    engine = DummyEngine()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), engine, context_builder=DummyBuilder()
    )
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

    percent, _ = asyncio.run(dbg._coverage_percent([Path("dummy.py")]))  # path-ignore
    assert percent == 0.0
    assert "coverage generation failed" in caplog.text


def test_coverage_records_executed_functions(monkeypatch, tmp_path):
    engine = DummyEngine()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), engine, context_builder=DummyBuilder()
    )

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
            Path(outfile).write_text(
                """<coverage><packages><package><classes><class filename='foo.py'>"  # path-ignore
                "<methods>"
                "<method name='run'><lines><line number='1' hits='1'/></lines></method>"
                "<method name='skip'><lines><line number='2' hits='0'/></lines></method>"
                "</methods></class></classes></package></packages></coverage>"""
            )

        def report(self, include=None, file=None):
            return 100.0

    monkeypatch.setattr(sds, "Coverage", Cov)
    captured = {}
    monkeypatch.setattr(
        sds,
        "record_run",
        lambda *a, **k: captured.update(a[1] if len(a) > 1 else a[0]),
    )

    percent, _ = asyncio.run(dbg._coverage_percent([Path("dummy.py")]))  # path-ignore
    assert percent == 100.0
    assert captured.get("executed_functions") == ["foo.py:run"]  # path-ignore


def test_coverage_subprocess_failure(monkeypatch):
    engine = DummyEngine()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), engine, context_builder=DummyBuilder()
    )

    async def fail_exec(*a, **k):
        return _stub_proc(1, b"boom", b"err")

    monkeypatch.setattr(sds.asyncio, "create_subprocess_exec", fail_exec)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_exec)

    with pytest.raises(sds.CoverageSubprocessError) as exc:
        asyncio.run(dbg._coverage_percent([Path("dummy.py")]))  # path-ignore
    assert "boom" in str(exc.value)


def test_flakiness_deterministic(monkeypatch):
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder()
    )
    monkeypatch.setattr(dbg, "_run_tests", lambda p, env=None: (80.0, 0.0))
    assert dbg._test_flakiness(Path("dummy.py")) == 0.0  # path-ignore


def test_flakiness_variable(monkeypatch):
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder(), flakiness_runs=5
    )
    cov_vals = [50.0, 60.0, 70.0, 80.0, 90.0]

    def fake_run(path, env=None):
        return cov_vals.pop(0), 0.0

    monkeypatch.setattr(dbg, "_run_tests", fake_run)
    flakiness = dbg._test_flakiness(Path("dummy.py"))  # path-ignore
    assert abs(flakiness - 6.324) < 0.001


def test_flakiness_handles_failed_runs(monkeypatch):
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder(), flakiness_runs=5
    )
    vals = [50.0, RuntimeError("boom"), 70.0, RuntimeError("boom"), 50.0]

    def flaky_run(path, env=None):
        v = vals.pop(0)
        if isinstance(v, Exception):
            raise v
        return v, 0.0

    monkeypatch.setattr(dbg, "_run_tests", flaky_run)
    flakiness = dbg._test_flakiness(Path("dummy.py"))  # path-ignore
    assert flakiness > 10.0


def test_run_tests_retries_on_subprocess_failure(monkeypatch):
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder()
    )

    calls = {"n": 0}

    async def fake_cov(paths, env=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sds.CoverageSubprocessError("boom")
        return 80.0, {}

    monkeypatch.setattr(dbg, "_generate_tests", lambda logs: [])
    monkeypatch.setattr(dbg, "_recent_logs", lambda limit=5: [])
    monkeypatch.setattr(dbg, "_coverage_percent", fake_cov)
    dbg._test_retries = 2
    cov, _ = dbg._run_tests(Path("dummy.py"))  # path-ignore
    assert cov == 80.0
    assert calls["n"] == 2


def test_backend_unreachable_fails_fast(monkeypatch):
    from menace.patch_score_backend import PatchScoreBackend

    class BadBackend(PatchScoreBackend):
        def store(self, record):
            pass

        def fetch_recent(self, limit: int = 20):
            raise RuntimeError("nope")

    mod = types.SimpleNamespace(BadBackend=BadBackend)
    monkeypatch.setitem(sys.modules, "bad_backend", mod)
    monkeypatch.setenv("PATCH_SCORE_BACKEND", "bad_backend:BadBackend")
    with pytest.raises(RuntimeError):
        sds.SelfDebuggerSandbox(
            DummyTelem(), DummyEngine(), context_builder=DummyBuilder()
        )


def test_score_weights_evolve_from_audit():
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(),
        DummyEngine(),
        context_builder=DummyBuilder(),
        audit_trail=trail,
        smoothing_factor=0.5,
    )

    dbg._log_patch(
        "p1",
        "success",
        coverage_delta=0.1,
        error_delta=0.0,
        roi_delta=0.3,
    )
    dbg._composite_score(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    first = dbg.score_weights

    dbg._log_patch(
        "p2",
        "success",
        coverage_delta=0.2,
        error_delta=0.1,
        roi_delta=0.1,
    )
    dbg._composite_score(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    second = dbg.score_weights

    dbg._log_patch(
        "p3",
        "success",
        coverage_delta=0.0,
        error_delta=0.2,
        roi_delta=0.2,
    )
    dbg._composite_score(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    third = dbg.score_weights

    assert first != second
    assert second != third
    assert abs(sum(first) - 6.0) < 1e-6
    assert abs(sum(second) - 6.0) < 1e-6
    assert abs(sum(third) - 6.0) < 1e-6
    assert second[4] > first[4]
    assert third[4] >= second[4]


def test_composite_score_ema_smooth():
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(),
        DummyEngine(),
        context_builder=DummyBuilder(),
        audit_trail=trail,
        smoothing_factor=0.5,
    )

    patches = [
        {"coverage_delta": 0.1, "error_delta": 0.0, "roi_delta": 0.3},
        {"coverage_delta": 0.2, "error_delta": 0.1, "roi_delta": 0.1},
        {"coverage_delta": 0.0, "error_delta": 0.2, "roi_delta": 0.2},
    ]
    scores = []
    for p in patches:
        dbg._log_patch("p", "success", **p)
        score = dbg._composite_score(
            p["coverage_delta"],
            p["error_delta"],
            p["roi_delta"],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        scores.append(score)

    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] >= scores[1] >= scores[2]


def test_weights_update_from_patch_db(tmp_path):
    patch_db = sds.PatchHistoryDB(tmp_path / "p.db")
    engine = DummyEngine()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), engine, context_builder=DummyBuilder()
    )

    patch_db.add(
        types.SimpleNamespace(
            ts="1",
            errors_before=5,
            errors_after=2,
            roi_delta=0.2,
            complexity_delta=0.1,
            synergy_roi=0.05,
            synergy_efficiency=0.1,
            coverage_delta=0.0,
        )
    )
    patch_db.add(
        types.SimpleNamespace(
            ts="2",
            errors_before=6,
            errors_after=3,
            roi_delta=0.2,
            complexity_delta=0.2,
            synergy_roi=0.05,
            synergy_efficiency=0.2,
            coverage_delta=0.0,
        )
    )

    dbg._update_score_weights(patch_db)
    before = dbg.score_weights[4]

    patch_db.add(
        types.SimpleNamespace(
            ts="3",
            errors_before=6,
            errors_after=1,
            roi_delta=0.4,
            complexity_delta=0.3,
            synergy_roi=2.0,
            synergy_efficiency=2.0,
            coverage_delta=0.0,
        )
    )

    dbg._update_score_weights(patch_db)
    after = dbg.score_weights[4]
    assert after > before

    assert patch_db.get_weights() == dbg.score_weights


def test_weight_updates_affect_score(tmp_path):
    patch_db = sds.PatchHistoryDB(tmp_path / "p.db")
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder()
    )

    base = dbg._composite_score(0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5)

    patch_db.add(
        types.SimpleNamespace(
            ts="1",
            errors_before=5,
            errors_after=1,
            roi_delta=0.5,
            complexity_delta=0.1,
            synergy_roi=1.0,
            synergy_efficiency=1.0,
            coverage_delta=0.0,
        )
    )

    dbg._update_score_weights(patch_db)
    dbg._score_db = patch_db
    updated = dbg._composite_score(0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5)

    assert updated > base


def test_weights_persist_across_instances(tmp_path):
    patch_db = sds.PatchHistoryDB(tmp_path / "p.db")
    dbg1 = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder()
    )
    patch_db.add(
        types.SimpleNamespace(
            ts="1",
            errors_before=4,
            errors_after=2,
            roi_delta=0.3,
            complexity_delta=0.1,
            synergy_roi=0.6,
            synergy_efficiency=0.6,
            coverage_delta=0.0,
        )
    )
    dbg1._update_score_weights(patch_db)
    saved = dbg1.score_weights

    dbg2 = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder()
    )
    dbg2._update_score_weights(patch_db)
    assert dbg2.score_weights == saved


def test_composite_score_uses_tracker_synergy():
    tracker = types.SimpleNamespace(
        synergy_metrics_history={"synergy_roi": [0.5], "synergy_efficiency": [0.4]},
        metrics_history={},
    )
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder()
    )
    base = dbg._composite_score(0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0)
    with_tracker = dbg._composite_score(
        0.1,
        0.0,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        tracker=tracker,
    )
    assert with_tracker > base


def test_composite_score_custom_weights():
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder()
    )
    base = dbg._composite_score(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
    weighted = dbg._composite_score(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        weights={"roi": 0.0, "efficiency": 0.0},
    )
    assert weighted < base


def test_synergy_metrics_affect_patch_acceptance(monkeypatch, tmp_path):
    engine = DummyEngine()
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(),
        engine,
        context_builder=DummyBuilder(),
        audit_trail=trail,
        delta_margin=0.6,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        dbg, "_generate_tests", lambda logs: ["def test_ok():\n    assert True\n"]
    )
    monkeypatch.setattr(sds.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(dbg, "_run_tests", lambda p, env=None: (50.0, 0.0))
    monkeypatch.setattr(dbg, "_test_flakiness", lambda p, env=None, *, runs=None: 0.0)
    monkeypatch.setattr(dbg, "_code_complexity", lambda p: 0.0)

    dbg.analyse_and_fix()
    assert not engine.applied

    tracker = types.SimpleNamespace(
        synergy_metrics_history={"synergy_roi": [0.5], "synergy_efficiency": [0.5]},
        metrics_history={},
        update=lambda *a, **k: None,
    )
    dbg.analyse_and_fix(tracker=tracker)
    assert engine.applied


def test_candidates_evaluated_concurrently(monkeypatch, tmp_path):
    engine = DummyEngine()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), engine, context_builder=DummyBuilder()
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        dbg,
        "_generate_tests",
        lambda logs: ["def a():\n    pass\n", "def b():\n    pass\n"],
    )
    monkeypatch.setattr(sds.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(dbg, "_test_flakiness", lambda p, env=None, *, runs=None: 0.0)
    monkeypatch.setattr(dbg, "_code_complexity", lambda p: 0.0)

    def fake_run(path, env=None):
        time.sleep(0.1)
        return 50.0, 0.0

    monkeypatch.setattr(dbg, "_run_tests", fake_run)

    start = time.perf_counter()
    dbg.analyse_and_fix()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5


def test_run_tests_logs_output(monkeypatch, tmp_path):
    engine = DummyEngine()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), engine, context_builder=DummyBuilder()
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))

    async def fake_cov(paths, env=None, **kwargs):
        raise sds.CoverageSubprocessError("boom")

    monkeypatch.setattr(dbg, "_coverage_percent", fake_cov)
    monkeypatch.setattr(
        dbg,
        "_record_exception",
        lambda exc: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError):
        dbg._run_tests(Path("test_dummy.py"))  # path-ignore

    assert dbg._last_test_log is not None
    assert dbg._last_test_log.exists()
    assert dbg._last_test_log.parent == tmp_path


def test_log_patch_includes_log_path(monkeypatch, tmp_path):
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder(), audit_trail=trail
    )
    monkeypatch.chdir(tmp_path)

    log_file = tmp_path / "fail.log"
    log_file.write_text("boom")
    dbg._last_test_log = log_file

    dbg._log_patch("desc", "failed")

    rec = json.loads(trail.records[-1])
    assert rec["log_path"] == str(log_file)


def test_analyse_and_fix_aborts_on_run_error(monkeypatch, tmp_path):
    engine = DummyEngine()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), engine, context_builder=DummyBuilder()
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        dbg, "_generate_tests", lambda logs: ["def test_ok():\n    assert True\n"]
    )
    monkeypatch.setattr(dbg, "_test_flakiness", lambda *a, **k: 0.0)
    monkeypatch.setattr(dbg, "_code_complexity", lambda p: 0.0)
    monkeypatch.setattr(sds.subprocess, "run", lambda *a, **k: None)

    def fail_run(path, env=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(dbg, "_run_tests", fail_run)

    dbg.analyse_and_fix()
    assert not engine.applied


def test_flakiness_runs_env(monkeypatch):
    monkeypatch.setenv("FLAKINESS_RUNS", "7")
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder(), flakiness_runs=3
    )
    assert dbg.flakiness_runs == 7


def test_flakiness_history_affects_score(tmp_path):
    patch_db = sds.PatchHistoryDB(tmp_path / "p.db")
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder()
    )
    dbg._score_db = patch_db
    patch_db.record_flakiness("a.py", 1.0)  # path-ignore
    base = dbg._composite_score(0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0)
    penalized = dbg._composite_score(0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, filename="a.py")  # path-ignore
    assert penalized < base


def test_patch_scores_persist_and_retrieve(monkeypatch, tmp_path):
    db_path = tmp_path / "scores.db"
    monkeypatch.setenv("SANDBOX_SCORE_DB", str(db_path))
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder(), audit_trail=trail
    )
    dbg._log_patch(
        "desc",
        "success",
        coverage_delta=0.1,
        error_delta=0.0,
        roi_delta=0.2,
        score=0.8,
        flakiness=0.1,
        runtime_impact=0.05,
        complexity=1.0,
        synergy_roi=0.3,
        synergy_efficiency=0.4,
    )
    rows = dbg.recent_scores(1)
    assert len(rows) == 1
    row = rows[0]
    assert row[0] == "desc"
    assert row[1] == "success"
    assert abs(row[12] - 0.8) < 1e-6


def test_recent_scores_limit(monkeypatch, tmp_path):
    db_path = tmp_path / "scores.db"
    monkeypatch.setenv("SANDBOX_SCORE_DB", str(db_path))
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder(), audit_trail=DummyTrail()
    )
    for i in range(3):
        dbg._log_patch(f"p{i}", "success", score=float(i))
    rows = dbg.recent_scores(2)
    assert len(rows) == 2
    assert rows[0][0] == "p2"
    assert rows[1][0] == "p1"


def test_parallel_evaluation_records_scores(monkeypatch, tmp_path):
    patch_db = sds.PatchHistoryDB(tmp_path / "p.db")
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(),
        DummyEngine(),
        context_builder=DummyBuilder(),
        audit_trail=trail,
        flakiness_runs=1,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        dbg,
        "_generate_tests",
        lambda logs: ["def a():\n    pass\n", "def b():\n    pass\n"],
    )
    monkeypatch.setattr(sds.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(dbg, "_code_complexity", lambda p: 0.0)

    cov_vals = [50.0, 60.0, 50.0, 50.0, 70.0, 50.0, 50.0, 80.0, 50.0]

    def fake_run(path, env=None):
        return cov_vals.pop(0), 0.0

    monkeypatch.setattr(dbg, "_run_tests", fake_run)

    dbg.analyse_and_fix(patch_db=patch_db)

    rec = json.loads(trail.records[-1])
    assert rec["score"] > 0
    assert len(patch_db.flaky) >= 1


def test_parallel_candidate_scoring_runtime(monkeypatch, tmp_path):
    patch_db = sds.PatchHistoryDB(tmp_path / "p.db")
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(),
        DummyEngine(),
        context_builder=DummyBuilder(),
        audit_trail=trail,
        flakiness_runs=1,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        dbg,
        "_generate_tests",
        lambda logs: ["def a():\n    pass\n", "def b():\n    pass\n"],
    )
    monkeypatch.setattr(sds.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(dbg, "_code_complexity", lambda p: 0.0)

    cov_vals = [50.0, 60.0, 50.0, 50.0, 70.0, 50.0, 50.0, 80.0, 50.0]

    def fake_run(path, env=None):
        time.sleep(0.1)
        return cov_vals.pop(0), 0.0

    monkeypatch.setattr(dbg, "_run_tests", fake_run)

    start = time.perf_counter()
    dbg.analyse_and_fix(patch_db=patch_db)
    elapsed = time.perf_counter() - start

    rec = json.loads(trail.records[-1])
    assert rec["score"] > 0
    assert len(patch_db.flaky) >= 1
    assert elapsed < 0.5


def test_coverage_revert_records_history(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_SCORE_DB", str(tmp_path / "scores.db"))

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
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), engine, context_builder=DummyBuilder(), audit_trail=trail
    )
    monkeypatch.setattr(
        dbg, "_generate_tests", lambda logs: ["def test_ok():\n    assert True\n"]
    )
    monkeypatch.setattr(sds.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(dbg, "_test_flakiness", lambda p, env=None, *, runs=None: 0.0)
    monkeypatch.setattr(dbg, "_code_complexity", lambda p: 0.0)

    cov_vals = [80.0, 50.0]

    async def fake_cov(p, env=None, **kwargs):
        return cov_vals.pop(0), {}

    monkeypatch.setattr(dbg, "_coverage_percent", fake_cov)

    dbg.analyse_and_fix()

    assert engine.rollback_mgr.calls == ["1"]
    rows = dbg.recent_scores(2)
    assert any(r[1] == "reverted" for r in rows)


def test_analyse_and_fix_records_history(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_SCORE_DB", str(tmp_path / "scores.db"))
    patch_db = sds.PatchHistoryDB(tmp_path / "p.db")
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(),
        DummyEngine(),
        context_builder=DummyBuilder(),
        audit_trail=trail,
        flakiness_runs=2,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        dbg, "_generate_tests", lambda logs: ["def test_ok():\n    assert True\n"]
    )
    monkeypatch.setattr(sds.subprocess, "run", lambda *a, **k: None)

    def fake_run(path, env=None):
        return 80.0, 0.0

    monkeypatch.setattr(dbg, "_run_tests", fake_run)
    monkeypatch.setattr(dbg, "_test_flakiness", lambda p, env=None, *, runs=None: 0.0)
    monkeypatch.setattr(dbg, "_code_complexity", lambda p: 0.0)

    tracker = types.SimpleNamespace(
        synergy_metrics_history={"synergy_roi": [0.5], "synergy_efficiency": [0.4]},
        metrics_history={},
    )

    dbg.analyse_and_fix(patch_db=patch_db, tracker=tracker)

    assert patch_db.flaky

    cur = dbg._history_conn.execute(
        "SELECT filename, flakiness FROM flakiness_history ORDER BY id DESC LIMIT 1"
    )
    row = cur.fetchone()
    assert row is not None
    assert "test_auto.py" in row[0]  # path-ignore
    assert row[1] >= 0.0

    rows = dbg.recent_scores(1)
    assert len(rows) == 1
    r = rows[0]
    assert r[0] == "auto_debug"
    assert r[6] > 0.0
    assert r[7] > 0.0


def test_roi_metrics_change_between_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_SCORE_DB", str(tmp_path / "scores.db"))
    patch_db = sds.PatchHistoryDB(tmp_path / "p.db")
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder(), flakiness_runs=1
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        dbg, "_generate_tests", lambda logs: ["def test_ok():\n    assert True\n"]
    )
    monkeypatch.setattr(sds.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(dbg, "_run_tests", lambda p, env=None: (75.0, 0.0))
    monkeypatch.setattr(dbg, "_test_flakiness", lambda p, env=None, *, runs=None: 0.0)
    monkeypatch.setattr(dbg, "_code_complexity", lambda p: 0.0)

    tracker = types.SimpleNamespace(
        synergy_metrics_history={"synergy_roi": [0.2], "synergy_efficiency": [0.3]},
        metrics_history={},
    )
    dbg.analyse_and_fix(patch_db=patch_db, tracker=tracker)

    tracker.synergy_metrics_history["synergy_roi"].append(0.7)
    tracker.synergy_metrics_history["synergy_efficiency"].append(0.6)
    dbg.analyse_and_fix(patch_db=patch_db, tracker=tracker)

    rows = dbg.recent_scores(2)
    assert len(rows) >= 2
    assert rows[0][6] != rows[1][6]
    assert rows[0][7] != rows[1][7]
