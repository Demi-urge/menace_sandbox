import sqlite3
import time
from pathlib import Path

from fcntl_compat import LOCK_EX, LOCK_NB, flock

import audit
import db_router as dr
import research_aggregator_bot as rab


def test_logged_cursor_skips_audit_when_bootstrap_safe(monkeypatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _capture(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("audit logging should be skipped when bootstrap_safe")

    monkeypatch.setattr(dr, "_log_db_access", _capture)

    conn: dr.LoggedConnection = sqlite3.connect(  # type: ignore[assignment]  # noqa: SQL001
        ":memory:", factory=dr.LoggedConnection
    )
    conn.menace_id = "m-bootstrap"
    conn.audit_bootstrap_safe = True

    cur = conn.cursor()
    cur.execute("CREATE TABLE example(id INTEGER)")

    assert calls == []


def test_infodb_bootstrap_ignores_locked_audit_log(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "bootstrap_audit.log"
    state_path = Path(f"{log_path}.state")
    state_path.touch()

    audit._loggers.clear()
    monkeypatch.setattr(audit, "DEFAULT_LOG_PATH", log_path)
    monkeypatch.setattr(audit, "LOCK_TIMEOUT", 5.0)
    monkeypatch.setattr(audit, "BOOTSTRAP_LOCK_TIMEOUT", 5.0)
    monkeypatch.setattr(dr, "_audit_bootstrap_safe_default", False)

    with state_path.open("r+") as sf:
        flock(sf.fileno(), LOCK_EX | LOCK_NB)
        start = time.perf_counter()
        rab.InfoDB(tmp_path / "info.db")
        duration = time.perf_counter() - start

    assert duration < 1
