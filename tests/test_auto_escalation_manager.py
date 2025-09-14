 # flake8: noqa
import os
import sys
import types
import logging
import pytest

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

# Stub out self-coding interface to avoid heavy dependencies during import.
coding_iface = types.ModuleType("coding_bot_interface")
coding_iface.self_coding_managed = lambda *a, **k: (lambda cls: cls)
sys.modules.setdefault("menace.coding_bot_interface", coding_iface)
sys.modules.setdefault("coding_bot_interface", coding_iface)

# Lightweight watchdog implementation to avoid importing heavy dependencies.
watchdog_mod = types.ModuleType("watchdog")

class Notifier:
    def __init__(self, auto_handler=None):
        self.auto_handler = auto_handler

    def escalate(self, msg, attachments=None):
        if self.auto_handler:
            self.auto_handler.handle(msg, attachments)

watchdog_mod.Notifier = Notifier
watchdog_mod.AutoEscalationManager = None

def _default_auto_handler(builder):
    mgr_cls = watchdog_mod.AutoEscalationManager
    return mgr_cls(context_builder=builder) if mgr_cls else None

watchdog_mod._default_auto_handler = _default_auto_handler

sys.modules.setdefault("menace.watchdog", watchdog_mod)


class DummyContextBuilder:
    def __init__(self, *a, **k):
        pass

    def refresh_db_weights(self):
        pass

    def build(self, *a, **k):
        return ""


vs_mod = types.SimpleNamespace(
    ContextBuilder=DummyContextBuilder,
    CognitionLayer=object,
    EmbeddableDBMixin=object,
    SharedVectorService=object,
)
sys.modules.setdefault("vector_service", vs_mod)
sys.modules.setdefault(
    "vector_service.context_builder",
    types.SimpleNamespace(ContextBuilder=DummyContextBuilder),
)

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

    from vector_service.context_builder import ContextBuilder

    mgr = aem.AutoEscalationManager(
        context_builder=ContextBuilder(
            "bots.db", "code.db", "errors.db", "workflows.db"
        ),
        event_bus=DummyBus(),
        publish_attempts=3,
    )
    caplog.set_level(logging.ERROR)
    mgr.handle("x")
    assert len(attempts) == 3
    assert "failed publishing escalation event" in caplog.text


def test_init_fails_without_self_coding_manager(monkeypatch):
    import menace.auto_escalation_manager as aem
    from vector_service.context_builder import ContextBuilder
    from pathlib import Path

    class BadManager:
        def __init__(self, *a, **k):
            raise RuntimeError("boom")

    class DummyPipeline:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setitem(
        sys.modules,
        "menace.self_coding_manager",
        types.SimpleNamespace(SelfCodingManager=BadManager),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace.model_automation_pipeline",
        types.SimpleNamespace(ModelAutomationPipeline=DummyPipeline),
    )
    monkeypatch.setattr(aem, "SelfCodingEngine", lambda *a, **k: object())
    monkeypatch.setattr(aem, "CodeDB", lambda: object())
    monkeypatch.setattr(aem, "ErrorDB", lambda: object())
    monkeypatch.setattr(aem, "AutomatedDebugger", lambda *a, **k: object())
    monkeypatch.setattr(aem, "resolve_path", lambda name: Path(name))
    monkeypatch.setattr(
        aem, "init_local_knowledge", lambda *a, **k: types.SimpleNamespace(memory=None)
    )

    with pytest.raises(RuntimeError):
        aem.AutoEscalationManager(
            context_builder=ContextBuilder(
                "bots.db", "code.db", "errors.db", "workflows.db"
            )
        )


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
