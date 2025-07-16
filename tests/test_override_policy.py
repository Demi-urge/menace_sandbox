import os
import threading
import time
import sqlite3
from pathlib import Path

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import menace.override_policy as op


class StubEnhancementDB:
    def __init__(self, path: Path) -> None:
        self.path = path
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_history(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_fingerprint TEXT,
                    prompt TEXT,
                    fix TEXT,
                    success INTEGER,
                    ts TEXT
                )
                """
            )
            conn.commit()

    def add_prompt_history(
        self, fingerprint: str, prompt: str, fix: str = "", success: bool = False
    ) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO prompt_history(error_fingerprint, prompt, fix, success, ts) VALUES (?,?,?,?,?)",
                (fingerprint, prompt, fix, int(success), ""),
            )
            conn.commit()


def test_update_policy_disables_flag(tmp_path):
    enh_db = StubEnhancementDB(tmp_path / "enh.db")
    ov_db = op.OverrideDB(tmp_path / "ov.db")
    mgr = op.OverridePolicyManager(ov_db, enh_db, window_size=20, confidence=0.8)

    ov_db.set("sig", True)
    for _ in range(20):
        enh_db.add_prompt_history("err", "p", fix="sig", success=True)
    mgr.update_policy("sig")
    assert not mgr.require_human("sig")


def test_run_continuous_updates(tmp_path):
    enh_db = StubEnhancementDB(tmp_path / "e.db")
    ov_db = op.OverrideDB(tmp_path / "o.db")
    mgr = op.OverridePolicyManager(ov_db, enh_db, window_size=20, confidence=0.8)

    ov_db.set("sig", True)
    for _ in range(20):
        enh_db.add_prompt_history("err", "p", fix="sig", success=True)
    stop = threading.Event()
    thread = mgr.run_continuous(interval=0.01, stop_event=stop)
    time.sleep(0.15)
    stop.set()
    thread.join(timeout=0.1)
    assert not mgr.require_human("sig")


def test_run_continuous_logs_repeated_failure(tmp_path, monkeypatch, caplog):
    enh_db = StubEnhancementDB(tmp_path / "c.db")
    ov_db = op.OverrideDB(tmp_path / "c2.db")
    mgr = op.OverridePolicyManager(ov_db, enh_db)

    def boom(*a, **k):
        raise RuntimeError("fail")

    monkeypatch.setattr(mgr, "update_all", boom)
    stop = threading.Event()
    caplog.set_level("ERROR")
    thread = mgr.run_continuous(interval=0.01, stop_event=stop)
    time.sleep(0.05)
    stop.set()
    thread.join(timeout=0.1)
    assert "policy updates failing repeatedly" in caplog.text


def test_update_all_logs_failure(tmp_path, monkeypatch, caplog):
    enh_db = StubEnhancementDB(tmp_path / "f.db")
    ov_db = op.OverrideDB(tmp_path / "o.db")
    mgr = op.OverridePolicyManager(ov_db, enh_db)

    ov_db.set("sig", True)

    def boom(*a, **k):
        raise RuntimeError("fail")

    monkeypatch.setattr(mgr, "_fetch_results", boom)
    caplog.set_level("ERROR")
    mgr.update_all()
    assert "failed updating policy" in caplog.text
