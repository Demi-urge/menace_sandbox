import types
import json
import hashlib
import menace.error_logger as elog

class StubReplicator:
    instances = []

    def __init__(self, topic="menace.telemetry", hosts=None, *, sentry=None, disk_path=None):
        self.topic = topic
        self.hosts = hosts
        self.events = []
        self.queue = []
        self.fail = False
        StubReplicator.instances.append(self)

    def replicate(self, event):
        # drain queue first
        remaining = []
        for ev in self.queue:
            if self.fail:
                remaining.append(ev)
            else:
                self.events.append(ev)
        self.queue = remaining
        if self.fail:
            self.queue.append(event)
        else:
            self.events.append(event)

    def flush(self):
        self.fail = False
        remaining = []
        for ev in self.queue:
            payload = json.dumps(ev.dict(exclude={"checksum"}), sort_keys=True).encode("utf-8")
            if not ev.checksum or hashlib.sha256(payload).hexdigest() != ev.checksum:
                continue
            if self.fail:
                remaining.append(ev)
            else:
                self.events.append(ev)
        self.queue = remaining


def test_error_logger_telemetry_replication(monkeypatch):
    monkeypatch.setenv("KAFKA_HOSTS", "broker:9092")
    monkeypatch.setenv("PATCH_ID", "42")
    monkeypatch.setenv("DEPLOY_ID", "7")
    monkeypatch.setattr(elog, "TelemetryReplicator", StubReplicator)
    events = []
    db = types.SimpleNamespace(add_telemetry=lambda e: events.append(e))
    monkeypatch.setattr(elog, "get_embedder", lambda: None)
    logger = elog.ErrorLogger(db)
    try:
        raise RuntimeError("boom")
    except Exception as exc:
        logger.log(exc, "t1", "bot1")
    assert events
    assert StubReplicator.instances
    inst = StubReplicator.instances[0]
    assert inst.hosts == "broker:9092"
    assert inst.events and inst.events[0].task_id == "t1"
    assert inst.events[0].patch_id == 42
    assert inst.events[0].deploy_id == 7


def test_replicator_flush_after_failure(monkeypatch):
    monkeypatch.setenv("KAFKA_HOSTS", "broker:9092")
    monkeypatch.setenv("PATCH_ID", "99")
    monkeypatch.setenv("DEPLOY_ID", "13")
    monkeypatch.setattr(elog, "TelemetryReplicator", StubReplicator)
    events = []
    db = types.SimpleNamespace(add_telemetry=lambda e: events.append(e))
    monkeypatch.setattr(elog, "get_embedder", lambda: None)
    logger = elog.ErrorLogger(db)
    inst = StubReplicator.instances[-1]
    inst.fail = True
    try:
        raise RuntimeError("boom2")
    except Exception as exc:
        logger.log(exc, "t2", "bot2")
    assert inst.queue and not inst.events
    inst.fail = False
    logger.replicator.flush()
    assert inst.events and inst.events[0].task_id == "t2"
    assert inst.events[0].patch_id == 99
    assert inst.events[0].deploy_id == 13


def test_event_checksum_and_validation(monkeypatch):
    monkeypatch.setenv("KAFKA_HOSTS", "broker:9092")
    monkeypatch.setattr(elog, "TelemetryReplicator", StubReplicator)
    events = []
    db = types.SimpleNamespace(add_telemetry=lambda e: events.append(e))
    monkeypatch.setattr(elog, "get_embedder", lambda: None)
    logger = elog.ErrorLogger(db)
    try:
        raise RuntimeError("boom3")
    except Exception as exc:
        logger.log(exc, "t3", "bot3")
    inst = StubReplicator.instances[-1]
    ev = inst.events[0]
    data = ev.dict(exclude={"checksum"})
    expected = hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
    assert ev.checksum == expected

    good = ev.copy()
    bad = ev.copy()
    bad.checksum = "deadbeef"
    inst.events = []
    inst.queue = [good, bad]
    logger.replicator.flush()
    assert len(inst.events) == 1
    assert inst.events[0].checksum == good.checksum
