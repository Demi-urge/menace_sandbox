import importlib
import sqlite3

import db_router


def _reload_module():
    import experiment_history_db as mod

    return importlib.reload(mod)


def test_experiment_history_db_routing(tmp_path):
    local = tmp_path / "local.db"
    shared = tmp_path / "shared.db"
    db_router.init_db_router(
        "exp_hist", local_db_path=str(local), shared_db_path=str(shared)
    )
    mod = _reload_module()
    db = mod.ExperimentHistoryDB()
    db.add(mod.ExperimentLog("A", 1.0, 0.1, 0.2))
    conn = db_router.GLOBAL_ROUTER.get_connection("experiment_history")
    row = conn.execute("SELECT variant FROM experiment_history").fetchone()
    assert row == ("A",)
    shared_rows = db_router.GLOBAL_ROUTER.shared_conn.execute(
        "SELECT name FROM sqlite_master WHERE name='experiment_history'"
    ).fetchall()
    assert shared_rows == []


def test_experiment_history_no_direct_sqlite(tmp_path, monkeypatch):
    local = tmp_path / "local2.db"
    shared = tmp_path / "shared2.db"
    db_router.init_db_router(
        "exp_hist2", local_db_path=str(local), shared_db_path=str(shared)
    )
    mod = _reload_module()

    calls: list[object] = []

    def bad_connect(*a, **k):  # pragma: no cover - should not run
        calls.append(1)
        raise AssertionError("sqlite3.connect should not be called directly")

    monkeypatch.setattr(sqlite3, "connect", bad_connect)

    db = mod.ExperimentHistoryDB()
    db.add(mod.ExperimentLog("B", 2.0, 0.2, 0.3))
    db.add_test(mod.TestLog("B", "C", 1.0, 0.5))

    assert not calls

