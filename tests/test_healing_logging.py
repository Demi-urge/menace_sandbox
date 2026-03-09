import json
import sqlite3
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

import menace.advanced_error_management as aem
from menace.knowledge_graph import KnowledgeGraph
from menace.rollback_manager import RollbackManager


def test_heal_logs_action(tmp_path):
    priv = Ed25519PrivateKey.generate()
    priv_bytes = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    mgr = RollbackManager(
        str(tmp_path / "rb.db"),
        audit_trail_path=str(tmp_path / "audit.log"),
        audit_privkey=priv_bytes,
    )
    orch = aem.SelfHealingOrchestrator(KnowledgeGraph(), rollback_mgr=mgr)
    orch.heal("bot1", patch_id="p1")

    with sqlite3.connect(mgr.path) as conn:
        rows = conn.execute(
            "SELECT bot, action, patch_id FROM healing_actions"
        ).fetchall()
    assert rows == [("bot1", "heal", "p1")]

    logs = (tmp_path / "audit.log").read_text().splitlines()
    assert logs
    payload = json.loads(logs[0].split(" ", 1)[1])
    assert payload["bot"] == "bot1"
    assert payload["patch_id"] == "p1"
    assert "change_id" in payload
    pub = mgr.audit_trail.public_key_bytes
    assert mgr.audit_trail.verify(pub)


def test_probe_and_heal_uses_runtime_mode_targets(monkeypatch):
    monkeypatch.delenv("MENACE_HEALTHCHECK_URL", raising=False)
    seen = []

    def fake_get(url, timeout=2):
        seen.append(url)
        return type("R", (), {"status_code": 200})()

    monkeypatch.setattr(aem, "requests", type("Req", (), {"get": staticmethod(fake_get)}))
    monkeypatch.setattr(aem.SelfHealingOrchestrator, "heal", lambda self, bot, patch_id=None: None)

    orch_local = aem.SelfHealingOrchestrator(
        KnowledgeGraph(),
        config={"runtime_mode": "local", "health_host": "127.0.0.1"},
    )
    orch_local.probe_and_heal("bot1")

    orch_container = aem.SelfHealingOrchestrator(
        KnowledgeGraph(),
        config={"runtime_mode": "container"},
    )
    orch_container.probe_and_heal("bot1")

    assert "http://127.0.0.1:8000/health" in seen
    assert "http://menace:8000/health" in seen


def test_probe_and_heal_honors_explicit_health_url_override(monkeypatch):
    monkeypatch.setenv("MENACE_HEALTHCHECK_URL", "http://override:8000/health")

    seen = []

    def fake_get(url, timeout=2):
        seen.append(url)
        return type("R", (), {"status_code": 200})()

    monkeypatch.setattr(aem, "requests", type("Req", (), {"get": staticmethod(fake_get)}))
    monkeypatch.setattr(aem.SelfHealingOrchestrator, "heal", lambda self, bot, patch_id=None: None)

    orch = aem.SelfHealingOrchestrator(KnowledgeGraph())
    orch.probe_and_heal("bot1")

    assert seen == ["http://override:8000/health"]


def test_probe_and_heal_dns_failure_is_infra_config_and_skips_restart(monkeypatch, caplog):
    import socket

    def dns_fail(url, timeout=2):
        raise socket.gaierror("name or service not known")

    healed = []

    monkeypatch.setattr(aem, "requests", type("Req", (), {"get": staticmethod(dns_fail)}))
    monkeypatch.setattr(aem.SelfHealingOrchestrator, "heal", lambda self, bot, patch_id=None: healed.append(bot))

    orch = aem.SelfHealingOrchestrator(KnowledgeGraph(), config={"runtime_mode": "container"})
    with caplog.at_level("WARNING"):
        orch.probe_and_heal("bot1")

    assert healed == []
    assert any("classifying as infra/config issue" in rec.getMessage() for rec in caplog.records)


def test_probe_and_heal_rate_limits_target_warning(monkeypatch, caplog):
    class Ticker:
        def __init__(self):
            self.values = [61.0, 62.0, 63.0]

        def __call__(self):
            return self.values.pop(0)

    monkeypatch.setattr(aem.time, "time", Ticker())
    monkeypatch.setattr(aem, "requests", type("Req", (), {"get": staticmethod(lambda url, timeout=2: type("R", (), {"status_code": 200})())}))
    monkeypatch.setattr(aem.SelfHealingOrchestrator, "heal", lambda self, bot, patch_id=None: None)

    orch = aem.SelfHealingOrchestrator(KnowledgeGraph(), config={"runtime_mode": "container"})
    orch._health_target_warning_interval = 60.0

    with caplog.at_level("WARNING"):
        orch.probe_and_heal("bot1")
        orch.probe_and_heal("bot1")

    health_target_logs = [
        rec.getMessage() for rec in caplog.records if "healthcheck target resolved" in rec.getMessage().lower()
    ]
    assert len(health_target_logs) == 1
