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
