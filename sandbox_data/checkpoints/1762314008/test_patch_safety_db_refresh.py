import sqlite3
import time

from patch_safety import PatchSafety


def _make_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE failures (cause TEXT, demographics TEXT, profitability REAL, retention REAL, cac REAL, roi REAL)"
    )
    conn.commit()
    conn.close()


def test_failure_db_refresh(tmp_path):
    db = tmp_path / "failures.db"
    _make_db(str(db))
    ps = PatchSafety(storage_path=None, failure_db_path=str(db), refresh_interval=0.2)
    example = {
        "cause": "foo",
        "demographics": "bar",
        "profitability": 1.0,
        "retention": 2.0,
        "cac": 3.0,
        "roi": 4.0,
    }
    ok, _, _ = ps.evaluate({}, example)
    assert ok
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO failures (cause, demographics, profitability, retention, cac, roi) VALUES (?,?,?,?,?,?)",
        (
            example["cause"],
            example["demographics"],
            example["profitability"],
            example["retention"],
            example["cac"],
            example["roi"],
        ),
    )
    conn.commit()
    conn.close()
    ps.load_failures()
    ok, _, _ = ps.evaluate({}, example)
    assert ok
    time.sleep(0.25)
    ps.load_failures()
    ok, score, _ = ps.evaluate({}, example)
    assert not ok
    assert score >= ps.threshold
