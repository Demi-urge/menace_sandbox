 # flake8: noqa
import os
import sys
import types
import logging

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
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
sys.modules.setdefault(
    "cryptography.hazmat.primitives.serialization", serialization
)
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
nx_mod = types.ModuleType("networkx")
nx_mod.DiGraph = object
sys.modules.setdefault("networkx", nx_mod)
sqlalchemy_mod = types.ModuleType("sqlalchemy")
engine_mod = types.ModuleType("sqlalchemy.engine")
class DummyEngine:
    pass
engine_mod.Engine = DummyEngine
sqlalchemy_mod.engine = engine_mod
sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
sys.modules.setdefault("sqlalchemy.engine", engine_mod)
req_mod = types.ModuleType("requests")
req_mod.Session = lambda: None
sys.modules.setdefault("requests", req_mod)


class DummyContextBuilder:
    def refresh_db_weights(self):
        pass

    def build(self, *a, **k):
        return ""


vs_mod = types.SimpleNamespace(
    ContextBuilder=DummyContextBuilder,
    get_default_context_builder=lambda **kwargs: DummyContextBuilder(),
    CognitionLayer=object,
    EmbeddableDBMixin=object,
    SharedVectorService=object,
)
sys.modules.setdefault("vector_service", vs_mod)

import menace.watchdog as wd
import menace.auto_escalation_manager as aem


def test_notifier_uses_auto_handler(monkeypatch):
    called = {}

    class DummyAuto:
        def handle(self, msg, attachments=None):
            called['msg'] = msg

    n = wd.Notifier(auto_handler=DummyAuto())
    n.escalate("issue")
    assert called['msg'] == "issue"


def test_notifier_default_handler(monkeypatch):
    called = []

    class DummyAuto:
        def handle(self, msg, attachments=None):
            called.append(msg)

    monkeypatch.setattr(wd, "AutoEscalationManager", lambda *a, **k: DummyAuto())
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
    n = wd.Notifier()
    n.auto_handler = wd._default_auto_handler(builder)
    n.escalate("boom")
    assert called == ["boom"]


def test_self_service_override_enables_safe(tmp_path, monkeypatch):
    import menace.self_service_override as so
    import menace.data_bot as db

    class DummyDF(list):
        @property
        def iloc(self):
            class ILoc:
                def __init__(self, rows):
                    self._rows = rows
                def __getitem__(self, idx):
                    return self._rows[idx]
            return ILoc(self)

    class FakeROI:
        def history(self, limit=2):
            return DummyDF([
                {"revenue": 50.0, "api_cost": 0.0},
                {"revenue": 100.0, "api_cost": 0.0},
            ])

    roi = FakeROI()

    monkeypatch.delenv("MENACE_SAFE", raising=False)
    metrics = db.MetricsDB(tmp_path / "m.db")
    metrics.add(db.MetricRecord(bot="b", cpu=1.0, memory=1.0, response_time=0.1, disk_io=1.0, net_io=1.0, errors=10))

    svc = so.SelfServiceOverride(roi, metrics, tracker=so.BaselineTracker())
    monkeypatch.setattr(svc.tracker, "update", lambda *a, **k: (0, 0, 0, 3))
    svc.adjust()
    assert os.environ.get("MENACE_SAFE") == "1"


def test_publish_retry_and_log(monkeypatch, caplog):
    attempts = []

    class DummyBus:
        def publish(self, topic, event):
            attempts.append(True)
            raise RuntimeError("boom")

    from vector_service import get_default_context_builder

    mgr = aem.AutoEscalationManager(
        context_builder=get_default_context_builder(),
        event_bus=DummyBus(),
        publish_attempts=3,
    )
    caplog.set_level(logging.ERROR)
    mgr.handle("x")
    assert len(attempts) == 3
    assert "failed publishing escalation event" in caplog.text


def test_auto_rollback_service(tmp_path, monkeypatch):
    import menace.self_service_override as so
    import menace.data_bot as db
    import subprocess

    class DummyDF(list):
        @property
        def iloc(self):
            class ILoc:
                def __init__(self, rows):
                    self._rows = rows
                def __getitem__(self, idx):
                    return self._rows[idx]
            return ILoc(self)

    class FakeROI:
        def history(self, limit=2):
            return DummyDF([
                {"revenue": 50.0, "api_cost": 0.0},
                {"revenue": 100.0, "api_cost": 0.0},
            ])

    roi = FakeROI()

    monkeypatch.delenv("MENACE_SAFE", raising=False)
    metrics = db.MetricsDB(tmp_path / "m.db")
    metrics.add(db.MetricRecord(bot="b", cpu=1.0, memory=1.0, response_time=0.1, disk_io=1.0, net_io=1.0, errors=10))
    metrics.log_eval("system", "avg_energy_score", 0.2)

    calls = []

    def fake_run(cmd, check=True, stdout=None, stderr=None):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    svc = so.AutoRollbackService(roi, metrics, tracker=so.BaselineTracker())
    monkeypatch.setattr(svc.tracker, "update", lambda *a, **k: (0, 0, 0, 3))
    svc.adjust()
    assert calls and calls[0][0] == "git"
    assert os.environ.get("MENACE_SAFE") == "1"
