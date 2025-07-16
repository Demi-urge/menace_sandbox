import os
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
