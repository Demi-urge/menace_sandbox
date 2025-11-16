import pytest

pytest.importorskip("sqlalchemy")

from menace.databases import MenaceDB
from menace.db_router import DBRouter
from menace.audit_trail import AuditTrail
from menace.bot_database import BotDB
from menace.code_database import CodeDB, CodeRecord


def setup_router(tmp_path):
    mdb = MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
    priv = Ed25519PrivateKey.generate()
    priv_bytes = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    router = DBRouter(
        bot_db=BotDB(tmp_path / 'b.db'),
        code_db=CodeDB(tmp_path / 'c.db'),
        menace_db=mdb,
        bot_roles={'reader': 'read', 'writer': 'write', 'admin': 'admin'},
        audit_trail_path=str(tmp_path / 'audit.log'),
        audit_privkey=priv_bytes
    )
    return router, mdb


def test_permission_checks_and_audit(tmp_path):
    router, mdb = setup_router(tmp_path)

    # reader can read
    router.execute_query('bot', 'SELECT * FROM bots', requesting_bot='reader')

    # reader cannot write
    with pytest.raises(PermissionError):
        router.insert_code(CodeRecord(code='x', summary='s'), requesting_bot='reader')

    # writer can insert
    cid = router.insert_code(CodeRecord(code='x', summary='s'), requesting_bot='writer')
    router.delete_code(cid, requesting_bot='admin')

    with mdb.engine.begin() as conn:
        logs = list(conn.execute(mdb.audit_log.select()))
    assert logs
    actions = {row.action for row in logs}
    assert 'insert_code' in actions
    assert 'delete_code' in actions
    pub = router.audit_trail.public_key_bytes
    assert router.audit_trail.verify(pub)

