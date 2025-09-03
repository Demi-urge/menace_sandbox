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
