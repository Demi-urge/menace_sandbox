import menace.audit_trail as at
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization


def test_audit_trail(tmp_path):
    path = tmp_path / "audit.log"
    priv = Ed25519PrivateKey.generate()
    priv_bytes = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    trail = at.AuditTrail(str(path), priv_bytes)
    trail.record("event1")
    trail.record("event2")
    pub = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    assert trail.verify(pub)


def test_audit_trail_no_key(tmp_path, caplog):
    caplog.set_level("WARNING")
    path = tmp_path / "audit.log"
    trail = at.AuditTrail(str(path))
    assert "will not be signed" in caplog.text
    trail.record("event1")
    line = path.read_text().strip()
    assert line.startswith("- ")
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    assert trail.verify(pub)

