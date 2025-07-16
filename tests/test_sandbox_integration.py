import importlib.util
import sys
import types
from pathlib import Path
import sandbox_runner

from tests.test_menace_master import _setup_mm_stubs, DummyBot, _stub_module


def _load_master():
    path = Path(__file__).resolve().parents[1] / "menace_master.py"
    spec = importlib.util.spec_from_file_location("menace_master", path)
    mm = importlib.util.module_from_spec(spec)
    sys.modules["menace_master"] = mm
    spec.loader.exec_module(mm)
    return mm


def test_sandbox_integration(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    mm = _load_master()

    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=DummyBot)
    _stub_module(monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot)

    monkeypatch.setattr(mm, "_init_unused_bots", lambda: None)
    monkeypatch.setattr(mm, "_start_dependency_watchdog", lambda event_bus=None: None)
    monkeypatch.setattr(mm, "run_once", lambda models: None)
    # override policy manager stub with run_continuous method
    class DummyService:
        def __init__(self, *a, **k):
            pass

        def run_continuous(self, *a, **k):
            class T:
                def join(self, timeout=None):
                    pass

            return T()

        def adjust(self):
            pass
    monkeypatch.setattr(sys.modules["menace.override_policy"], "OverridePolicyManager", DummyService, raising=False)
    monkeypatch.setattr(sys.modules["menace.self_service_override"], "SelfServiceOverride", DummyService, raising=False)
    monkeypatch.setattr(mm, "SelfServiceOverride", DummyService)

    # stub repo clone and copy
    monkeypatch.setattr(mm.subprocess, "run", lambda cmd, check=False: Path(cmd[-1]).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(mm.shutil, "copy2", lambda *a, **k: None)

    # simple stubs
    class DummyBus:
        def __init__(self, persist_path=None, **kw):
            pass
        def close(self):
            pass
    monkeypatch.setattr(mm, "UnifiedEventBus", DummyBus)

    class DummyOrch:
        def create_oversight(self, *a, **k):
            pass

        def start_scheduled_jobs(self):
            pass

        def run_cycle(self, *a, **k):
            return {}
    monkeypatch.setattr(mm, "MenaceOrchestrator", DummyOrch)

    class DummySandbox:
        def __init__(self, *a, **k):
            pass
        def analyse_and_fix(self):
            pass
    monkeypatch.setattr(sys.modules["menace.self_debugger_sandbox"], "SelfDebuggerSandbox", DummySandbox)

    class DummyTester:
        def _run_once(self):
            pass
    monkeypatch.setattr(mm, "SelfTestService", DummyTester)

    disc_calls = []
    eff_calls = []

    class DummyDiscrepancy:
        def scan(self):
            disc_calls.append(True)
            return [("x", 1.0, None)]

    class DummyEfficiency:
        def assess_efficiency(self):
            eff_calls.append(True)
            return {}


    class DummyImprover:
        def __init__(self):
            self.rois = iter([0.1, 0.15, 0.151])
            self.calls = 0
            self.disc = DummyDiscrepancy()
            self.eff = DummyEfficiency()

        def run_cycle(self):
            self.calls += 1
            self.disc.scan()
            self.eff.assess_efficiency()
            try:
                val = next(self.rois)
            except StopIteration:
                val = 0.151
            return types.SimpleNamespace(roi=types.SimpleNamespace(roi=val))

    improver = DummyImprover()
    monkeypatch.setattr(mm, "SelfImprovementEngine", lambda *a, **k: improver)

    monkeypatch.setattr(sys.modules["menace.error_bot"], "ErrorDB", lambda p: DummyBot(), raising=False)
    monkeypatch.setattr(sys.modules["menace.error_bot"], "ErrorBot", DummyBot, raising=False)
    monkeypatch.setattr(sys.modules["menace.data_bot"], "MetricsDB", DummyBot, raising=False)

    flag = tmp_path / "first.flag"
    monkeypatch.setenv("MENACE_FIRST_RUN_FILE", str(flag))
    monkeypatch.setenv("AUTO_SANDBOX", "1")
    monkeypatch.setenv("AUTO_BOOTSTRAP", "0")
    monkeypatch.setenv("AUTO_UPDATE", "0")
    monkeypatch.setenv("AUTO_BACKUP", "0")
    monkeypatch.setenv("RUN_CYCLES", "1")
    monkeypatch.setenv("SLEEP_SECONDS", "0")
    monkeypatch.setenv("SANDBOX_ROI_TOLERANCE", "0.01")
    monkeypatch.setenv("SANDBOX_CYCLES", "5")

    mm.main([])

    assert flag.exists()
    assert improver.calls == 3
    assert disc_calls
    assert eff_calls


def test_sandbox_error_no_flag(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    mm = _load_master()

    monkeypatch.setattr(mm, "_init_unused_bots", lambda: None)
    monkeypatch.setattr(mm, "_start_dependency_watchdog", lambda event_bus=None: None)

    class DummyOrch:
        def create_oversight(self, *a, **k):
            pass

        def start_scheduled_jobs(self):
            pass

        def run_cycle(self, *a, **k):
            return {}

    monkeypatch.setattr(mm, "MenaceOrchestrator", DummyOrch)

    class DummyService:
        def __init__(self, *a, **k):
            pass

        def run_continuous(self, *a, **k):
            class T:
                def join(self, timeout=None):
                    pass
            return T()

        def adjust(self):
            pass

    monkeypatch.setattr(sys.modules["menace.override_policy"], "OverridePolicyManager", DummyService, raising=False)
    monkeypatch.setattr(sys.modules["menace.self_service_override"], "SelfServiceOverride", DummyService, raising=False)
    monkeypatch.setattr(mm, "SelfServiceOverride", DummyService)

    monkeypatch.setattr(mm, "run_once", lambda models: None)
    monkeypatch.setattr(sandbox_runner, "_run_sandbox", lambda args: (_ for _ in ()).throw(RuntimeError("boom")))

    flag = tmp_path / "first.flag"
    monkeypatch.setenv("MENACE_FIRST_RUN_FILE", str(flag))
    monkeypatch.setenv("AUTO_SANDBOX", "1")
    monkeypatch.setenv("AUTO_BOOTSTRAP", "0")
    monkeypatch.setenv("AUTO_UPDATE", "0")
    monkeypatch.setenv("AUTO_BACKUP", "0")

    mm.main([])

    assert not flag.exists()
