import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.watchdog as wd  # noqa: E402
import menace.error_bot as eb  # noqa: E402
import menace.resource_allocation_optimizer as rao  # noqa: E402
import menace.data_bot as db  # noqa: E402
from menace.unified_event_bus import UnifiedEventBus  # noqa: E402
from menace.error_logger import TelemetryEvent  # noqa: E402
from vector_service.context_builder import ContextBuilder  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402


def _setup_dbs(tmp_path):
    err = eb.ErrorDB(tmp_path / "e.db")
    roi = rao.ROIDB(tmp_path / "r.db")
    metrics = db.MetricsDB(tmp_path / "m.db")
    return err, roi, metrics


def test_watchdog_triggers(tmp_path, monkeypatch):
    err_db, roi_db, metrics_db = _setup_dbs(tmp_path)
    # add consecutive failures
    for _ in range(4):
        err_db.add_telemetry(TelemetryEvent(stack_trace="boom"))
    # add ROI drop
    roi_db.add(rao.KPIRecord(bot="b", revenue=100.0, api_cost=50.0, cpu_seconds=1.0, success_rate=1.0))  # noqa: E501
    roi_db.add(rao.KPIRecord(bot="b", revenue=70.0, api_cost=50.0, cpu_seconds=1.0, success_rate=1.0))  # noqa: E501
    # add stale metrics (3h old)
    old_ts = (datetime.utcnow() - timedelta(hours=3)).isoformat()
    metrics_db.add(db.MetricRecord(bot="b", cpu=1.0, memory=1.0, response_time=0.1, disk_io=1.0, net_io=1.0, errors=0, ts=old_ts))  # noqa: E501

    notified = {}

    def fake_notify(msg, attachments=None):
        notified["msg"] = msg

    notifier = wd.Notifier()
    notifier.notify = fake_notify
    builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
    watch = wd.Watchdog(
        err_db, roi_db, metrics_db, notifier=notifier, context_builder=builder
    )
    watch.check()
    assert "Failure Dossier" in notified.get("msg", "")


def test_watchdog_no_trigger(tmp_path):
    err_db, roi_db, metrics_db = _setup_dbs(tmp_path)
    err_db.add_telemetry(TelemetryEvent(stack_trace="boom"))
    roi_db.add(rao.KPIRecord(bot="b", revenue=50.0, api_cost=50.0, cpu_seconds=1.0, success_rate=1.0))  # noqa: E501
    roi_db.add(rao.KPIRecord(bot="b", revenue=50.0, api_cost=50.0, cpu_seconds=1.0, success_rate=1.0))  # noqa: E501
    metrics_db.add(db.MetricRecord(bot="b", cpu=1.0, memory=1.0, response_time=0.1, disk_io=1.0, net_io=1.0, errors=0))  # noqa: E501

    notifier = wd.Notifier()
    called = []

    def fake_notify(msg, attachments=None):
        called.append(msg)

    notifier.notify = fake_notify
    builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
    watch = wd.Watchdog(
        err_db, roi_db, metrics_db, notifier=notifier, context_builder=builder
    )
    watch.check()
    assert not called


def test_watchdog_heals_lost_heartbeat(tmp_path, monkeypatch):
    err_db, roi_db, metrics_db = _setup_dbs(tmp_path)
    registry = wd.BotRegistry()
    registry.record_heartbeat("bot1")
    registry.heartbeats["bot1"] -= 120  # stale

    healed = []

    class DummyHealer:
        def heal(self, bot: str) -> None:
            healed.append(bot)

    monkeypatch.setattr(
        wd,
        "SelfHealingOrchestrator",
        lambda g, backend=None: DummyHealer(),
    )
    builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
    watch = wd.Watchdog(
        err_db, roi_db, metrics_db, registry=registry, context_builder=builder
    )
    watch.check()
    assert healed == ["bot1"]


def test_self_healing_triggers_rollback(tmp_path, monkeypatch):
    from menace.knowledge_graph import KnowledgeGraph
    import menace.advanced_error_management as aem

    calls = []

    class DummyRB:
        def auto_rollback(self, pid, nodes, **kw):
            calls.append((pid, tuple(nodes)))

    safed = []

    class DummyDB:
        def set_safe_mode(self, mod):
            safed.append(mod)

    orch = aem.SelfHealingOrchestrator(
        KnowledgeGraph(),
        rollback_mgr=DummyRB(),
        error_db=DummyDB(),
        failure_threshold=2,
    )

    def fail_get(url, timeout=2):
        raise Exception("down")

    monkeypatch.setattr(aem, "requests", type("R", (), {"get": fail_get}))
    monkeypatch.setattr(
        aem.SelfHealingOrchestrator, "heal", lambda self, bot, patch_id=None: None
    )

    orch.probe_and_heal("bot")
    assert not calls
    orch.probe_and_heal("bot")
    assert calls and safed == ["bot"]


def test_watchdog_runs_debugger(tmp_path, monkeypatch):
    err_db, roi_db, metrics_db = _setup_dbs(tmp_path)
    # create metrics with rising errors
    for _ in range(5):
        metrics_db.add(
            db.MetricRecord(
                bot="b",
                cpu=1.0,
                memory=1.0,
                response_time=0.1,
                disk_io=1.0,
                net_io=1.0,
                errors=0,
            )
        )
    for _ in range(5):
        metrics_db.add(
            db.MetricRecord(
                bot="b",
                cpu=1.0,
                memory=1.0,
                response_time=0.1,
                disk_io=1.0,
                net_io=1.0,
                errors=10,
            )
        )

    called = []

    class DummyDebugger:
        def __init__(self, db, eng, **kwargs):
            pass

        def analyse_and_fix(self):
            called.append(True)

    bus = UnifiedEventBus()

    monkeypatch.setattr(wd, "AutomatedDebugger", DummyDebugger)
    builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
    watch = wd.Watchdog(
        err_db,
        roi_db,
        metrics_db,
        thresholds=wd.Thresholds(error_trend=1.0),
        event_bus=bus,
        context_builder=builder,
    )
    watch.check()
    assert called


def test_restart_logging(tmp_path, monkeypatch):
    err_db, roi_db, metrics_db = _setup_dbs(tmp_path)
    registry = wd.BotRegistry()
    registry.record_heartbeat("bot1")
    registry.heartbeats["bot1"] -= 120

    class DummyHealer:
        def heal(self, bot: str) -> None:
            pass

    monkeypatch.setattr(wd, "SelfHealingOrchestrator", lambda g, backend=None: DummyHealer())
    log = tmp_path / "r.log"
    builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
    watch = wd.Watchdog(
        err_db,
        roi_db,
        metrics_db,
        registry=registry,
        restart_log=str(log),
        context_builder=builder,
    )
    watch.check()
    assert "restarted bot1" in log.read_text()


def test_failover_restart(tmp_path, monkeypatch):
    err_db, roi_db, metrics_db = _setup_dbs(tmp_path)
    registry = wd.BotRegistry()
    registry.record_heartbeat("bot1")
    registry.heartbeats["bot1"] -= 120

    class DummyHealer:
        def heal(self, bot: str) -> None:
            raise Exception("fail")

    monkeypatch.setattr(wd, "SelfHealingOrchestrator", lambda g, backend=None: DummyHealer())
    called = {}

    def fake_popen(cmd, stdout=None, stderr=None):
        called["cmd"] = cmd

        class P:
            pass

        return P()

    monkeypatch.setattr(wd.subprocess, "Popen", fake_popen)
    log = tmp_path / "r.log"
    builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
    watch = wd.Watchdog(
        err_db,
        roi_db,
        metrics_db,
        registry=registry,
        failover_hosts=["host1"],
        restart_log=str(log),
        context_builder=builder,
    )
    watch.check()
    assert called.get("cmd") and "host1" in called["cmd"]
    assert "failover bot1 on host1" in log.read_text()
