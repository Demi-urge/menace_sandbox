import threading

import db_router


def test_get_connection_thread_safe(tmp_path):
    router = db_router.DBRouter("thr", str(tmp_path / "local.db"), str(tmp_path / "shared.db"))
    with router.get_connection("bots", operation="write") as conn:
        conn.execute("CREATE TABLE bots(id INTEGER)")
        conn.commit()
    with router.get_connection("models", operation="write") as conn:
        conn.execute("CREATE TABLE models(id INTEGER)")
        conn.commit()

    errors: list[Exception] = []
    barrier = threading.Barrier(10)

    def use_shared() -> None:
        try:
            barrier.wait()
            for _ in range(20):
                conn = router.get_connection("bots")
                conn.execute("SELECT COUNT(*) FROM bots")
        except Exception as exc:  # pragma: no cover - unexpected
            errors.append(exc)

    def use_local() -> None:
        try:
            barrier.wait()
            for _ in range(20):
                conn = router.get_connection("models")
                conn.execute("SELECT COUNT(*) FROM models")
        except Exception as exc:  # pragma: no cover - unexpected
            errors.append(exc)

    threads = [threading.Thread(target=use_shared) for _ in range(5)] + [
        threading.Thread(target=use_local) for _ in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    counts = router.get_access_counts()
    assert counts["shared"]["bots"] >= 5 * 20 + 1
    assert counts["local"]["models"] >= 5 * 20 + 1
    router.close()
