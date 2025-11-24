import sqlite3
import time
from pathlib import Path

from fcntl_compat import LOCK_EX, LOCK_NB, flock

import audit
import db_router as dr
try:
    import menace_sandbox.research_aggregator_bot as rab
except ImportError:  # pragma: no cover - flat import fallback
    import research_aggregator_bot as rab


def test_logged_cursor_skips_audit_when_bootstrap_safe(monkeypatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _capture(*args, **kwargs):  # pragma: no cover - guarded by bootstrap flag
        calls.append((args, kwargs))
        raise AssertionError("audit logging should be skipped when bootstrap_safe")

    monkeypatch.setattr(dr, "_log_db_access_fn", _capture)
    monkeypatch.setattr(dr, "_ensure_log_db_access", lambda: _capture)

    conn: dr.LoggedConnection = sqlite3.connect(  # type: ignore[assignment]  # noqa: SQL001
        ":memory:", factory=dr.LoggedConnection
    )
    conn.menace_id = "m-bootstrap"
    conn.audit_bootstrap_safe = True

    cur = conn.cursor()
    cur.execute("CREATE TABLE example(id INTEGER)")

    assert calls == []


def test_bootstrap_safe_short_circuits_audit_state(monkeypatch, tmp_path) -> None:
    seen: list[Path] = []

    real_open_state = audit._open_state_file

    def _tracking_open(path: Path, *, bootstrap_safe: bool):
        seen.append(path)
        return real_open_state(path, bootstrap_safe=bootstrap_safe)

    monkeypatch.setattr(dr, "_log_db_access_fn", None)
    monkeypatch.setattr(audit, "_open_state_file", _tracking_open)

    log_path = tmp_path / "audit.log"

    dr._log_db_access("write", "bootstrap", 1, "m-bootstrap", log_path=log_path, bootstrap_safe=True)

    assert seen == []

    dr._log_db_access("write", "normal", 1, "m-normal", log_path=log_path, bootstrap_safe=False)

    assert Path(f"{log_path}.state") in seen


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


def test_bootstrap_buffering_defers_audit_and_flushes(monkeypatch) -> None:
    flushed_logs: list[tuple[tuple[object, ...], dict[str, object]]] = []
    flushed_records: list[dict[str, str]] = []

    def _capture_log(*args, **kwargs):
        flushed_logs.append((args, kwargs))

    def _capture_record(entry: dict[str, str]):
        flushed_records.append(entry)

    monkeypatch.setattr(dr, "_log_db_access_fn", _capture_log)
    monkeypatch.setattr(dr, "_ensure_log_db_access", lambda: _capture_log)
    monkeypatch.setattr(dr, "_record_audit_impl", _capture_record)

    baseline = dr.get_bootstrap_audit_metrics().get("bootstrap_replayed_calls", 0)

    with dr.bootstrap_audit_buffering():
        dr._log_db_access("write", "buffered", 1, "m-buf")
        dr._record_audit({"table_name": "buffered"})
        assert flushed_logs == []
        assert flushed_records == []

    assert len(flushed_logs) == 1
    assert len(flushed_records) == 1
    metrics = dr.get_bootstrap_audit_metrics()
    assert metrics.get("bootstrap_replayed_calls", 0) >= baseline + 1
