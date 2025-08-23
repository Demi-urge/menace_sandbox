import importlib
import threading

import db_router


def test_concurrent_get_connection(tmp_path, monkeypatch):
    """Concurrent access increments counts without errors."""
    importlib.reload(db_router)
    audit_log = tmp_path / "audit.log"
    monkeypatch.setattr(db_router, "_audit_log_path", str(audit_log))
    router = db_router.DBRouter("t3", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))

    def worker() -> None:
        for _ in range(20):
            router.get_connection("bots")
            router.get_connection("models")

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    try:
        counts = router.get_access_counts()
        assert counts["shared"]["bots"] == 5 * 20
        assert counts["local"]["models"] == 5 * 20
        lines = audit_log.read_text().strip().splitlines()
        assert len(lines) == 5 * 20 * 2
    finally:
        router.close()
