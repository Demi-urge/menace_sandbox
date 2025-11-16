import os
import sys
import types
import subprocess

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
primitives_mod = sys.modules.setdefault(
    "cryptography.hazmat.primitives", types.ModuleType("primitives")
)
asym_mod = sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
)
ed_mod = types.ModuleType("ed25519")
ed_mod.Ed25519PrivateKey = type("P", (), {
    "from_private_bytes": staticmethod(lambda b: object()),
    "public_key": lambda self: type("U", (), {"public_bytes": lambda self,*a,**k: b""})(),
    "public_bytes": lambda self,*a,**k: b"",
    "sign": lambda self,m: b"",
})
ed_mod.Ed25519PublicKey = ed_mod.Ed25519PrivateKey
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519", ed_mod
)
ser_mod = types.ModuleType("serialization")
ser_mod.Encoding = type("E", (), {"Raw": object()})
ser_mod.PublicFormat = type("F", (), {"Raw": object()})
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", ser_mod)
primitives_mod.asymmetric = asym_mod
primitives_mod.serialization = ser_mod
asym_mod.ed25519 = ed_mod
sys.modules["cryptography.hazmat"].primitives = primitives_mod

import menace.environment_restoration_service as ers


def test_run_and_stop(monkeypatch):
    calls = []
    monkeypatch.setattr(ers.EnvironmentBootstrapper, "bootstrap", lambda self: calls.append(True))
    monkeypatch.setattr(ers, "BackgroundScheduler", None)
    recorded = {}

    def fake_add_job(self, func, interval, id):
        recorded["func"] = func
        recorded["interval"] = interval
        recorded["id"] = id

    def fake_shutdown(self):
        recorded["shutdown"] = True

    monkeypatch.setattr(ers._SimpleScheduler, "add_job", fake_add_job)
    monkeypatch.setattr(ers._SimpleScheduler, "shutdown", fake_shutdown)

    svc = ers.EnvironmentRestorationService()
    svc.run_continuous(interval=123)
    assert recorded["id"] == "env_restore"
    assert svc.scheduler is not None

    recorded["func"]()
    assert calls

    svc.stop()
    assert recorded.get("shutdown")
    assert svc.scheduler is None


def test_run_once_retry(monkeypatch):
    calls = []

    def fail_then_ok(self):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("boom")

    monkeypatch.setattr(ers.EnvironmentBootstrapper, "bootstrap", fail_then_ok)
    import menace.retry_utils as ru
    monkeypatch.setattr(ru.time, "sleep", lambda *_: None)

    svc = ers.EnvironmentRestorationService()
    svc._run_once()

    assert len(calls) == 2


def test_run_once_failure_logs(monkeypatch, caplog):
    calls = []

    def always_fail(self):
        calls.append(True)
        raise subprocess.CalledProcessError(1, ["cmd"])

    monkeypatch.setattr(ers.EnvironmentBootstrapper, "bootstrap", always_fail)
    import menace.retry_utils as ru
    monkeypatch.setattr(ru.time, "sleep", lambda *_: None)

    svc = ers.EnvironmentRestorationService()
    caplog.set_level("ERROR")
    svc._run_once()

    assert len(calls) == 3
    assert "environment restoration failed" in caplog.text
