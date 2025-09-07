import pytest

import menace.self_coding_engine as sce
import menace.code_database as cd
import menace.menace_memory_manager as mm
from menace.rollback_manager import RollbackManager


def test_apply_patch_denied_logs(tmp_path):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
    priv = Ed25519PrivateKey.generate()
    priv_bytes = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    engine = sce.SelfCodingEngine(
        cd.CodeDB(tmp_path / "c.db"),
        mm.MenaceMemoryManager(tmp_path / "m.db"),
        bot_roles={"reader": "read"},
        audit_trail_path=str(tmp_path / "audit.log"),
        audit_privkey=priv_bytes,
        context_builder=types.SimpleNamespace(
            build_context=lambda *a, **k: {},
            refresh_db_weights=lambda *a, **k: None,
        ),
    )

    path = tmp_path / "file.py"  # path-ignore
    path.write_text("print('x')\n")

    with pytest.raises(PermissionError):
        engine.apply_patch(path, "helper", requesting_bot="reader")

    assert "helper" not in path.read_text()
    logs = (tmp_path / "audit.log").read_text().splitlines()
    assert logs
    pub = engine.audit_trail.public_key_bytes
    assert engine.audit_trail.verify(pub)
    assert "apply_patch" in logs[0].split(" ", 1)[1]


def test_rollback_denied_logs(tmp_path):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
    priv = Ed25519PrivateKey.generate()
    priv_bytes = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    mgr = RollbackManager(
        str(tmp_path / "rb.db"),
        bot_roles={"reader": "read"},
        audit_trail_path=str(tmp_path / "audit.log"),
        audit_privkey=priv_bytes,
    )

    mgr.register_patch("p1", "node")

    with pytest.raises(PermissionError):
        mgr.rollback("p1", requesting_bot="reader")

    patches = mgr.applied_patches()
    assert patches
    logs = (tmp_path / "audit.log").read_text().splitlines()
    assert logs
    pub = mgr.audit_trail.public_key_bytes
    assert mgr.audit_trail.verify(pub)
    assert "rollback" in logs[0].split(" ", 1)[1]
