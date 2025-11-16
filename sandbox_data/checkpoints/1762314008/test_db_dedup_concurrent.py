import logging
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

from db_dedup import insert_if_unique


def test_insert_if_unique_concurrent(tmp_path):
    path = tmp_path / "dedup.sqlite"
    base_conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    base_conn.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, content_hash TEXT UNIQUE NOT NULL)"
    )
    base_conn.close()

    logger = logging.getLogger(__name__)
    barrier = threading.Barrier(5)

    def worker() -> int | None:
        conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        try:
            barrier.wait()
            return insert_if_unique(
                "items",
                {"name": "alpha"},
                ["name"],
                "m1",
                conn=conn,
                logger=logger,
            )
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(worker) for _ in range(5)]
        ids = [f.result() for f in futures]

    assert len(set(ids)) == 1
    conn = sqlite3.connect(path, isolation_level=None)
    count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    conn.close()
    assert count == 1
