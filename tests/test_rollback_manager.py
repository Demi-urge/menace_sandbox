import importlib
import json
from pathlib import Path

import menace.access_control as access_control
import menace.roles as roles
import menace.rollback_manager as rm


def test_register_and_rollback(tmp_path):
    db_path = tmp_path / "rb.db"
    mgr = rm.RollbackManager(str(db_path))
    mgr.register_patch("p1", "nodeA")
    patches = mgr.applied_patches()
    assert len(patches) == 1
    mgr.rollback("p1")
    patches = mgr.applied_patches()
    assert not patches


def test_region_register_and_rollback(tmp_path):
    db_path = tmp_path / "rb.db"
    mgr = rm.RollbackManager(str(db_path))
    mgr.register_region_patch("p2", "nodeB", "file.txt", 10, 20)
    patches = mgr.applied_region_patches()
    assert len(patches) == 1
    assert patches[0].file == "file.txt"
    mgr.rollback_region("file.txt", 10, 20)
    patches = mgr.applied_region_patches()
    assert not patches


def test_bot_roles_from_config_allow_write(tmp_path, monkeypatch):
    global rm, roles, access_control

    custom = tmp_path / "bot_roles.json"
    custom.write_text(json.dumps({"automation:*": "write"}))
    monkeypatch.setenv("BOT_ROLES_FILE", str(custom))

    roles = importlib.reload(roles)
    access_control = importlib.reload(access_control)
    rm = importlib.reload(rm)

    try:
        db_path = tmp_path / "rb.db"
        mgr = rm.RollbackManager(str(db_path))
        mgr.register_patch("p3", "nodeC")
        mgr.rollback("p3", requesting_bot="automation:rollback")
        assert not mgr.applied_patches()
    finally:
        default_config = Path(__file__).resolve().parents[1] / "config" / "bot_roles.json"
        monkeypatch.setenv("BOT_ROLES_FILE", str(default_config))
        roles = importlib.reload(roles)
        access_control = importlib.reload(access_control)
        rm = importlib.reload(rm)
