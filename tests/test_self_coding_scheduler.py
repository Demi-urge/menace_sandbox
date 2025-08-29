import os
import logging
import sys
import types
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives", types.ModuleType("primitives")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    types.ModuleType("ed25519"),
)
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault(
    "cryptography.hazmat.primitives.serialization", serialization
)
sys.modules.setdefault("networkx", types.ModuleType("networkx"))
sys.modules.setdefault("pulp", types.ModuleType("pulp"))
sys.modules.setdefault("pandas", types.ModuleType("pandas"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("psutil", types.ModuleType("psutil"))
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sqlalchemy_mod = types.ModuleType("sqlalchemy")
engine_mod = types.ModuleType("sqlalchemy.engine")
class DummyEngine:
    pass

engine_mod.Engine = DummyEngine
sqlalchemy_mod.engine = engine_mod
sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
sys.modules.setdefault("sqlalchemy.engine", engine_mod)
sys.modules.setdefault("prometheus_client", types.ModuleType("prometheus_client"))
scm_stub = types.ModuleType("menace.self_coding_manager")
scm_stub.SelfCodingManager = object
sys.modules.setdefault("menace.self_coding_manager", scm_stub)
ae_stub = types.ModuleType("menace.advanced_error_management")
ae_stub.AutomatedRollbackManager = object
sys.modules.setdefault("menace.advanced_error_management", ae_stub)
sys.modules.setdefault("git", types.ModuleType("git"))

import menace.self_coding_scheduler as sched_mod
from menace.self_coding_scheduler import SelfCodingScheduler
from menace.sandbox_settings import SandboxSettings


class DummyManager:
    def __init__(self):
        self.bot_name = "bot"
        self.engine = types.SimpleNamespace(rollback_patch=lambda pid: None)

    def run_patch(self, *a, **k):
        raise RuntimeError("boom")


def _stop_after_first(sched: SelfCodingScheduler):
    def inner(_: float) -> None:
        sched.running = False
        raise SystemExit

    return inner


def test_patch_failure_logged(monkeypatch):
    mgr = DummyManager()
    data_bot = types.SimpleNamespace(roi=lambda b: 0.0, db=types.SimpleNamespace(fetch=lambda l: []))
    sched = SelfCodingScheduler(mgr, data_bot, interval=0)
    sched.last_roi = 1.0
    monkeypatch.setattr(sched_mod.time, "sleep", _stop_after_first(sched))
    monkeypatch.setattr(sched, "_current_errors", lambda: 0.0)
    sched.running = True
    calls = []
    monkeypatch.setattr(sched, "logger", types.SimpleNamespace(exception=calls.append))
    with pytest.raises(SystemExit):
        sched._loop()
    assert calls and "self-coding loop failed" in calls[0]


def test_settings_provide_defaults():
    mgr = DummyManager()
    data_bot = types.SimpleNamespace(roi=lambda b: 0.0, db=types.SimpleNamespace(fetch=lambda l: []))
    cfg = SandboxSettings(
        self_coding_interval=123,
        self_coding_roi_drop=-0.5,
        self_coding_error_increase=2.5,
    )
    sched = SelfCodingScheduler(mgr, data_bot, settings=cfg)
    assert (sched.interval, sched.roi_drop, sched.error_increase) == (123, -0.5, 2.5)


def test_constructor_overrides_settings():
    mgr = DummyManager()
    data_bot = types.SimpleNamespace(roi=lambda b: 0.0, db=types.SimpleNamespace(fetch=lambda l: []))
    cfg = SandboxSettings(
        self_coding_interval=123,
        self_coding_roi_drop=-0.5,
        self_coding_error_increase=2.5,
    )
    sched = SelfCodingScheduler(
        mgr,
        data_bot,
        interval=5,
        roi_drop=-0.2,
        error_increase=1.1,
        settings=cfg,
    )
    assert (sched.interval, sched.roi_drop, sched.error_increase) == (5, -0.2, 1.1)
