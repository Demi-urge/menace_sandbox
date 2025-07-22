from pathlib import Path
import logging
import sys
import types
import importlib
import pytest
import json
import asyncio
import time

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

# Create a lightweight menace package with stub modules used during import
menace_pkg = types.ModuleType("menace")
menace_pkg.__path__ = []
sys.modules.setdefault("menace", menace_pkg)
import logging_utils as _logging_utils
sys.modules.setdefault("menace.logging_utils", _logging_utils)

rt_spec = importlib.util.spec_from_file_location(
    "menace.roi_tracker", Path(__file__).resolve().parents[1] / "roi_tracker.py"
)
rt = importlib.util.module_from_spec(rt_spec)
assert rt_spec.loader is not None
rt_spec.loader.exec_module(rt)
sys.modules.setdefault("roi_tracker", rt)
sys.modules.setdefault("menace.roi_tracker", rt)

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
    monkeypatch.setattr(
        dbg, "_generate_tests", lambda logs: ["def test_fail():\n    assert False\n"]
    )

    async def fake_cov(p, env=None):
        return 50.0

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
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), engine, audit_trail=trail)
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

    async def fake_cov_ok(p, env=None):
        return 80.0

    monkeypatch.setattr(dbg, "_coverage_percent", fake_cov_ok)
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
    monkeypatch.setattr(
        dbg, "_generate_tests", lambda logs: ["def test_ok():\n   assert True\n"]
    )
    monkeypatch.setattr(sds.subprocess, "run", lambda *a, **k: None)

    async def fake_cov_ok(p, env=None):
        return 80.0

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
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), engine, audit_trail=trail)
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

        def apply_patch(self, path: Path, desc: str):
            delta = self.deltas.pop(0)
            self.calls.append(delta)
            return len(self.calls), False, delta

        def rollback_patch(self, pid: str) -> None:
            pass

    engine = ScoreEngine()
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), engine)
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

    async def fake_cov(p, env=None):
        return cov_vals.pop(0)

    monkeypatch.setattr(dbg, "_coverage_percent", fake_cov)

    dbg.analyse_and_fix()

    assert engine.calls[-1] == 0.5


def test_run_tests_includes_telemetry(monkeypatch, tmp_path):
    engine = DummyEngine()
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), engine)
    monkeypatch.chdir(tmp_path)

    base = Path("test_base.py")
    base.write_text("def test_base():\n    assert True\n")

    called = {}

    async def fake_cov(paths, env=None):
        called["paths"] = [Path(p) for p in paths]
        return 100.0

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

    cov = asyncio.run(dbg._coverage_percent([Path("dummy.py")]))
    assert cov == 0.0
    assert "coverage generation failed" in caplog.text
    assert "boom" in caplog.text


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

    cov = asyncio.run(dbg._coverage_percent([Path("dummy.py")]))
    assert cov == 0.0
    assert "coverage generation failed" in caplog.text


def test_coverage_subprocess_failure(monkeypatch):
    engine = DummyEngine()
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), engine)

    async def fail_exec(*a, **k):
        return _stub_proc(1, b"boom", b"err")

    monkeypatch.setattr(sds.asyncio, "create_subprocess_exec", fail_exec)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_exec)

    with pytest.raises(RuntimeError) as exc:
        asyncio.run(dbg._coverage_percent([Path("dummy.py")]))
    assert "boom" in str(exc.value)


def test_flakiness_deterministic(monkeypatch):
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), DummyEngine())
    monkeypatch.setattr(dbg, "_run_tests", lambda p, env=None: (80.0, 0.0))
    assert dbg._test_flakiness(Path("dummy.py")) == 0.0


def test_flakiness_variable(monkeypatch):
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), DummyEngine(), flakiness_runs=5)
    cov_vals = [50.0, 60.0, 70.0, 80.0, 90.0]

    def fake_run(path, env=None):
        return cov_vals.pop(0), 0.0

    monkeypatch.setattr(dbg, "_run_tests", fake_run)
    flakiness = dbg._test_flakiness(Path("dummy.py"))
    assert abs(flakiness - 6.324) < 0.001


def test_score_weights_evolve_from_audit():
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), audit_trail=trail, smoothing_factor=0.5
    )

    dbg._log_patch(
        "p1",
        "success",
        coverage_delta=0.1,
        error_delta=0.0,
        roi_delta=0.3,
    )
    dbg._composite_score(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    first = dbg.score_weights

    dbg._log_patch(
        "p2",
        "success",
        coverage_delta=0.2,
        error_delta=0.1,
        roi_delta=0.1,
    )
    dbg._composite_score(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    second = dbg.score_weights

    dbg._log_patch(
        "p3",
        "success",
        coverage_delta=0.0,
        error_delta=0.2,
        roi_delta=0.2,
    )
    dbg._composite_score(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
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
        DummyTelem(), DummyEngine(), audit_trail=trail, smoothing_factor=0.5
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
        )
        scores.append(score)

    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] >= scores[1] >= scores[2]


def test_weights_update_from_patch_db(tmp_path):
    patch_db = sds.PatchHistoryDB(tmp_path / "p.db")
    engine = DummyEngine()
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), engine)

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
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), DummyEngine())

    base = dbg._composite_score(0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.5, 0.5)

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
    updated = dbg._composite_score(0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.5, 0.5)

    assert updated > base


def test_weights_persist_across_instances(tmp_path):
    patch_db = sds.PatchHistoryDB(tmp_path / "p.db")
    dbg1 = sds.SelfDebuggerSandbox(DummyTelem(), DummyEngine())
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

    dbg2 = sds.SelfDebuggerSandbox(DummyTelem(), DummyEngine())
    dbg2._update_score_weights(patch_db)
    assert dbg2.score_weights == saved


def test_composite_score_uses_tracker_synergy():
    tracker = rt.ROITracker()
    tracker.update(0.0, 0.1, metrics={"synergy_roi": 0.5, "synergy_efficiency": 0.4})
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), DummyEngine())
    base = dbg._composite_score(0.1, 0.0, 0.1, 0.0, 0.0, 0.0)
    with_tracker = dbg._composite_score(
        0.1,
        0.0,
        0.1,
        0.0,
        0.0,
        0.0,
        tracker=tracker,
    )
    assert with_tracker > base


def test_synergy_metrics_affect_patch_acceptance(monkeypatch, tmp_path):
    engine = DummyEngine()
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), engine, audit_trail=trail, score_threshold=0.6
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
    rec_no_tracker = json.loads(trail.records[-1])
    assert rec_no_tracker["result"] == "reverted"

    tracker = rt.ROITracker()
    tracker.update(0.0, 0.1, metrics={"synergy_roi": 0.5, "synergy_efficiency": 0.5})
    trail.records.clear()
    dbg.analyse_and_fix(tracker=tracker)
    rec_with_tracker = json.loads(trail.records[-1])
    assert rec_with_tracker["result"] == "success"
    assert rec_with_tracker["synergy_roi"] > 0.0


def test_candidates_evaluated_concurrently(monkeypatch, tmp_path):
    engine = DummyEngine()
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), engine)
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
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), engine)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))

    def fake_run(cmd, capture_output=True, text=True, env=None):
        class R:
            returncode = 1
            stdout = "out"
            stderr = "err"

        return R()

    class Cov:
        def __init__(self, *a, **k):
            pass

        def load(self):
            pass

        def report(self, include=None, file=None):
            return 0.0

        def xml_report(self, outfile=None, include=None):
            return 0

    monkeypatch.setattr(sds.subprocess, "run", fake_run)
    monkeypatch.setattr(sds, "Coverage", Cov)

    dbg._run_tests(Path("test_dummy.py"))

    assert dbg._last_test_log is not None
    assert dbg._last_test_log.exists()
    assert dbg._last_test_log.parent == tmp_path


def test_log_patch_includes_log_path(monkeypatch, tmp_path):
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), DummyEngine(), audit_trail=trail)
    monkeypatch.chdir(tmp_path)

    log_file = tmp_path / "fail.log"
    log_file.write_text("boom")
    dbg._last_test_log = log_file

    dbg._log_patch("desc", "failed")

    rec = json.loads(trail.records[-1])
    assert rec["log_path"] == str(log_file)


def test_analyse_and_fix_aborts_on_run_error(monkeypatch, tmp_path):
    engine = DummyEngine()
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), engine)
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
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), DummyEngine(), flakiness_runs=3)
    assert dbg.flakiness_runs == 7


def test_flakiness_history_affects_score(tmp_path):
    patch_db = sds.PatchHistoryDB(tmp_path / "p.db")
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), DummyEngine())
    dbg._score_db = patch_db
    patch_db.record_flakiness("a.py", 1.0)
    base = dbg._composite_score(0.0, 0.0, 0.5, 0.0, 0.0, 0.0)
    penalized = dbg._composite_score(0.0, 0.0, 0.5, 0.0, 0.0, 0.0, filename="a.py")
    assert penalized < base


def test_patch_scores_persist_and_retrieve(monkeypatch, tmp_path):
    db_path = tmp_path / "scores.db"
    monkeypatch.setenv("SANDBOX_SCORE_DB", str(db_path))
    trail = DummyTrail()
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), DummyEngine(), audit_trail=trail)
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
    dbg = sds.SelfDebuggerSandbox(DummyTelem(), DummyEngine(), audit_trail=DummyTrail())
    for i in range(3):
        dbg._log_patch(f"p{i}", "success", score=float(i))
    rows = dbg.recent_scores(2)
    assert len(rows) == 2
    assert rows[0][0] == "p2"
    assert rows[1][0] == "p1"
