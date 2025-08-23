from __future__ import annotations

import json
import sqlite3
import threading
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

from db_router import GLOBAL_ROUTER, init_db_router

logger = logging.getLogger(__name__)

router = GLOBAL_ROUTER or init_db_router("visual_agent_queue")


class VisualAgentQueue:
    """SQLite-backed queue for visual agent tasks."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        try:
            conn = router.get_connection("tasks")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks(
                id TEXT PRIMARY KEY,
                prompt TEXT,
                branch TEXT,
                status TEXT,
                error TEXT,
                ts REAL,
                qorder INTEGER
            )
            """
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS metadata(last_completed REAL)"
            )
            if conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0] == 0:
                conn.execute("INSERT INTO metadata(last_completed) VALUES(0)")

            cols = [row[1] for row in conn.execute("PRAGMA table_info(tasks)")]
            if "qorder" not in cols:
                conn.execute("ALTER TABLE tasks ADD COLUMN qorder INTEGER")
                conn.execute("UPDATE tasks SET qorder = rowid WHERE qorder IS NULL")
            if "status" not in cols:
                conn.execute("ALTER TABLE tasks ADD COLUMN status TEXT")
            if "error" not in cols:
                conn.execute("ALTER TABLE tasks ADD COLUMN error TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status_qorder ON tasks(status, qorder)"
            )
            conn.commit()
        except sqlite3.DatabaseError:
            ts = int(time.time())
            corrupt = self.path.with_name(f"{self.path.name}.corrupt.{ts}")
            try:
                if self.path.exists():
                    self.path.rename(corrupt)
            except Exception:
                pass
            logger.warning("queue database corrupt; moved to %s", corrupt)
            conn = router.get_connection("tasks")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks(
                    id TEXT PRIMARY KEY,
                    prompt TEXT,
                    branch TEXT,
                    status TEXT,
                    error TEXT,
                    ts REAL,
                    qorder INTEGER
                )
                """
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS metadata(last_completed REAL)"
            )
            conn.execute("INSERT INTO metadata(last_completed) VALUES(0)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status_qorder ON tasks(status, qorder)"
            )
            conn.commit()

    # ------------------------------------------------------------------
    def check_integrity(self) -> bool:
        """Verify required tables exist. Rebuild DB if corrupt."""
        try:
            conn = router.get_connection("tasks")
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            if {"tasks", "metadata"} <= tables:
                return False
        except sqlite3.DatabaseError:
            pass

        ts = int(time.time())
        backup = self.path.with_name(f"{self.path.name}.corrupt.{ts}")
        try:
            if self.path.exists():
                self.path.rename(backup)
        except Exception:
            pass
        logger.warning("queue database invalid; moved to %s", backup)
        self._init_db()
        return True

    # ------------------------------------------------------------------
    @staticmethod
    def migrate_from_jsonl(
        db_path: Path,
        jsonl_path: Path,
        state_path: Optional[Path] = None,
    ) -> None:
        """Convert legacy JSONL queue and state to SQLite."""
        if db_path.exists() or not jsonl_path.exists():
            return

        job_status: Dict[str, Dict[str, object]] = {}
        last_completed = 0.0
        if state_path and state_path.exists():
            try:
                data = json.loads(state_path.read_text())
                if isinstance(data, dict) and isinstance(data.get("status"), dict):
                    job_status = dict(data["status"])
                    lc = data.get("last_completed")
                    if isinstance(lc, (int, float)):
                        last_completed = float(lc)
            except Exception:
                pass

        items: List[Dict[str, object]] = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        items.append(obj)
        except Exception:
            items = []

        q = VisualAgentQueue(db_path)
        conn = router.get_connection("tasks")
        now = time.time()
        order = 1
        for itm in items:
            tid = str(itm.get("id"))
            if not tid:
                continue
            info = job_status.get(tid, {})
            conn.execute(
                "INSERT OR IGNORE INTO tasks(id,prompt,branch,status,error,ts,qorder) VALUES(?,?,?,?,?,?,?)",
                (
                    tid,
                    itm.get("prompt", info.get("prompt", "")),
                    itm.get("branch", info.get("branch")),
                    info.get("status", "queued"),
                    info.get("error"),
                    now,
                    order,
                ),
            )
            order += 1
        for tid, info in job_status.items():
            if any(i.get("id") == tid for i in items):
                continue
            conn.execute(
                "INSERT OR IGNORE INTO tasks(id,prompt,branch,status,error,ts,qorder) VALUES(?,?,?,?,?,?,?)",
                (
                    tid,
                    info.get("prompt", ""),
                    info.get("branch"),
                    info.get("status", "queued"),
                    info.get("error"),
                    now,
                    order,
                ),
            )
            order += 1
        conn.execute(
            "UPDATE metadata SET last_completed=?", (last_completed,)
        )
        conn.commit()

        try:
            jsonl_path.rename(jsonl_path.with_suffix(jsonl_path.suffix + ".bak"))
        except Exception:
            pass
        if state_path and state_path.exists():
            try:
                state_path.rename(state_path.with_suffix(state_path.suffix + ".bak"))
            except Exception:
                pass
            hash_path = state_path.with_suffix(state_path.suffix + ".sha256")
            if hash_path.exists():
                try:
                    hash_path.rename(hash_path.with_suffix(hash_path.suffix + ".bak"))
                except Exception:
                    pass

    # ------------------------------------------------------------------
    def append(self, item: Dict[str, object]) -> None:
        with self._lock:
            conn = router.get_connection("tasks")
            qorder = conn.execute("SELECT COALESCE(MAX(qorder),0)+1 FROM tasks").fetchone()[0]
            conn.execute(
                "INSERT OR REPLACE INTO tasks(id,prompt,branch,status,error,ts,qorder) VALUES(?,?,?,?,?,?,?)",
                (
                    item.get("id"),
                    item.get("prompt", ""),
                    item.get("branch"),
                    item.get("status", "queued"),
                    item.get("error"),
                    time.time(),
                    qorder,
                ),
            )
            conn.commit()

    def popleft(self) -> Dict[str, object]:
        with self._lock:
            conn = router.get_connection("tasks")
            row = conn.execute(
                "SELECT id,prompt,branch FROM tasks WHERE status='queued' ORDER BY qorder LIMIT 1"
            ).fetchone()
            if row is None:
                raise IndexError("pop from empty queue")
            conn.execute(
                "UPDATE tasks SET status='running' WHERE id=?",
                (row[0],),
            )
            conn.commit()
            return {"id": row[0], "prompt": row[1], "branch": row[2]}

    def peek(self) -> Optional[Dict[str, object]]:
        with self._lock:
            conn = router.get_connection("tasks")
            row = conn.execute(
                "SELECT id,prompt,branch,status FROM tasks WHERE status='queued' ORDER BY qorder LIMIT 1"
            ).fetchone()
            if row:
                return {"id": row[0], "prompt": row[1], "branch": row[2], "status": row[3]}
            return None

    def load_all(self) -> List[Dict[str, object]]:
        with self._lock:
            conn = router.get_connection("tasks")
            rows = conn.execute(
                "SELECT id,prompt,branch,status FROM tasks WHERE status='queued' ORDER BY qorder"
            ).fetchall()
            return [
                {"id": r[0], "prompt": r[1], "branch": r[2], "status": r[3]} for r in rows
            ]

    def update_status(self, task_id: str, status: str, error: Optional[str] = None) -> None:
        with self._lock:
            conn = router.get_connection("tasks")
            conn.execute(
                "UPDATE tasks SET status=?, error=?, ts=? WHERE id=?",
                (status, error, time.time(), task_id),
            )
            conn.commit()

    def get_status(self) -> Dict[str, Dict[str, object]]:
        with self._lock:
            conn = router.get_connection("tasks")
            rows = conn.execute(
                "SELECT id,prompt,branch,status,error FROM tasks"
            ).fetchall()
            result: Dict[str, Dict[str, object]] = {}
            for r in rows:
                entry: Dict[str, object] = {
                    "status": r[3],
                    "prompt": r[1],
                    "branch": r[2],
                }
                if r[4]:
                    entry["error"] = r[4]
                result[r[0]] = entry
            return result

    def reset_running_tasks(self) -> None:
        """Mark any ``running`` tasks as ``queued``."""
        with self._lock:
            conn = router.get_connection("tasks")
            conn.execute("UPDATE tasks SET status='queued' WHERE status='running'")
            conn.commit()

    def get_last_completed(self) -> float:
        with self._lock:
            conn = router.get_connection("metadata")
            row = conn.execute("SELECT last_completed FROM metadata").fetchone()
            if row and row[0] is not None:
                return float(row[0])
            return 0.0

    def set_last_completed(self, ts: float) -> None:
        with self._lock:
            conn = router.get_connection("metadata")
            conn.execute("UPDATE metadata SET last_completed=?", (ts,))
            conn.commit()

    # ------------------------------------------------------------------
    def clear(self) -> None:
        with self._lock:
            conn = router.get_connection("tasks")
            conn.execute("DELETE FROM tasks")
            conn.commit()

    def __len__(self) -> int:
        with self._lock:
            conn = router.get_connection("tasks")
            row = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status='queued'"
            ).fetchone()
            return int(row[0]) if row else 0

    def __bool__(self) -> bool:
        return len(self) > 0

    def __iter__(self):
        return iter(self.load_all())

