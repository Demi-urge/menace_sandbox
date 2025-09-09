import logging
import sqlite3 as sql
import threading
from contextlib import contextmanager
from pathlib import Path
import sys

from dynamic_path_router import resolve_path

import pytest

# Adjust path so SelfTestService can be imported as a package module
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "menace_sandbox"))
sys.path.insert(0, str(repo_root))
from menace_sandbox.self_test_service import SelfTestService  # noqa: E402


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build_context(self, *a, **k):
        return "", "", {}


class DummySandbox:
    """Minimal stand-in for SelfDebuggerSandbox._history_db."""

    def __init__(self):
        self.logger = logging.getLogger("DummySandbox")
        self._history_conn = sql.connect(":memory:")
        self._history_lock = threading.Lock()

    @contextmanager
    def _history_db(self):
        if not self._history_conn:
            yield None
            return
        with self._history_lock:
            try:
                yield self._history_conn
                self._history_conn.commit()
            except sql.DatabaseError:
                try:
                    self._history_conn.rollback()
                except sql.DatabaseError:
                    self.logger.exception("history rollback failed")
                self.logger.exception("history commit failed")
                raise


def test_history_db_commit_failure_logs(caplog):
    sandbox = DummySandbox()
    sandbox._history_conn.close()
    with caplog.at_level(logging.ERROR):
        with pytest.raises(sql.DatabaseError):
            with sandbox._history_db():
                pass
    assert "history commit failed" in caplog.text
    assert "history rollback failed" in caplog.text


def test_state_file_load_failure_logs(tmp_path, caplog):
    bad = resolve_path("state.json", tmp_path)
    bad.write_text("{")
    with caplog.at_level(logging.ERROR):
        svc = SelfTestService(state_path=bad, context_builder=DummyBuilder())
    assert "failed to load state file" in caplog.text
    assert svc._state is None
