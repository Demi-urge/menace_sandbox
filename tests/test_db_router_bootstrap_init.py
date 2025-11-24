import time
from pathlib import Path

import db_router


def test_init_db_router_skips_resolve_in_bootstrap(monkeypatch, tmp_path):
    monkeypatch.setattr(db_router, "GLOBAL_ROUTER", None)
    monkeypatch.setattr(db_router, "get_project_root", lambda: tmp_path)

    def _fail_resolve(*_args, **_kwargs):
        raise AssertionError("resolve_path should not be used during bootstrap")

    monkeypatch.setattr(db_router, "resolve_path", _fail_resolve)

    router = db_router.init_db_router(
        "boot", bootstrap_mode=True, bootstrap_safe=True, init_deadline_s=5.0
    )

    db_file = router.local_conn.execute("PRAGMA database_list").fetchone()[2]
    assert Path(db_file).parent == tmp_path
    router.close()


def test_init_db_router_fallback_when_deadline_exceeded(monkeypatch, tmp_path):
    monkeypatch.setattr(db_router, "GLOBAL_ROUTER", None)
    monkeypatch.setattr(db_router, "get_project_root", lambda: tmp_path)

    def _slow_resolve(name):  # pragma: no cover - exercised in tests via sleep
        time.sleep(0.05)
        return tmp_path / name

    monkeypatch.setattr(db_router, "resolve_path", _slow_resolve)

    start = time.perf_counter()
    router = db_router.init_db_router(
        "slow",
        bootstrap_safe=True,
        init_deadline_s=0.01,
        bootstrap_mode=False,
    )
    duration = time.perf_counter() - start

    assert duration < 0.3
    assert getattr(router, "init_fallback", False) is True

    database_list = router.shared_conn.execute("PRAGMA database_list").fetchall()
    assert any(entry[2] in {"", ":memory:", None} for entry in database_list)
    router.close()
