"""Consolidated workflow discovery, seeding, and run-state tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import threading
from typing import Iterable, Mapping, Sequence
from uuid import uuid4

from bot_discovery import _iter_bot_modules
from dynamic_path_router import resolve_path
from self_improvement.workflow_discovery import DEFAULT_EXCLUDED_DIRS
from task_handoff_bot import WorkflowDB, WorkflowRecord

_RUN_DATA_DIR = Path(resolve_path("sandbox_data"))
_DEFAULT_DB_PATH = _RUN_DATA_DIR / "workflow_run_state.db"
_DEFAULT_JSONL_PATH = _RUN_DATA_DIR / "workflow_run_state.jsonl"
_LOCK = threading.Lock()


@dataclass(frozen=True)
class WorkflowRunRecord:
    """Durable record for a workflow execution attempt."""

    run_id: str
    workflow_id: str
    ts: str
    error_classification: str
    patch_attempts: int
    roi_delta: float
    retry_count: int
    metadata: Mapping[str, object] | None = None


class WorkflowRunStore:
    """Persist workflow run records in SQLite with JSONL backup."""

    def __init__(
        self,
        db_path: Path | None = None,
        jsonl_path: Path | None = None,
    ) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._jsonl_path = jsonl_path or _DEFAULT_JSONL_PATH
        self._initialised = False

    def _ensure_schema(self) -> None:
        if self._initialised:
            return
        _RUN_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_runs (
                    run_id TEXT PRIMARY KEY,
                    workflow_id TEXT,
                    ts TEXT,
                    error_classification TEXT,
                    patch_attempts INTEGER,
                    roi_delta REAL,
                    retry_count INTEGER,
                    metadata TEXT
                )
                """
            )
        self._initialised = True

    def record(self, record: WorkflowRunRecord) -> None:
        self._ensure_schema()
        payload = {
            **asdict(record),
            "metadata": json.dumps(record.metadata) if record.metadata else None,
        }
        with _LOCK:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO workflow_runs (
                        run_id, workflow_id, ts, error_classification,
                        patch_attempts, roi_delta, retry_count, metadata
                    ) VALUES (
                        :run_id, :workflow_id, :ts, :error_classification,
                        :patch_attempts, :roi_delta, :retry_count, :metadata
                    )
                    """,
                    payload,
                )
            with self._jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")

    def record_run(
        self,
        *,
        workflow_id: str,
        error_classification: str,
        patch_attempts: int,
        roi_delta: float,
        retry_count: int,
        metadata: Mapping[str, object] | None = None,
    ) -> WorkflowRunRecord:
        record = WorkflowRunRecord(
            run_id=str(uuid4()),
            workflow_id=str(workflow_id),
            ts=datetime.utcnow().isoformat(),
            error_classification=error_classification,
            patch_attempts=int(patch_attempts),
            roi_delta=float(roi_delta),
            retry_count=int(retry_count),
            metadata=metadata,
        )
        self.record(record)
        return record

    def list_runs(self) -> list[WorkflowRunRecord]:
        self._ensure_schema()
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT run_id, workflow_id, ts, error_classification,
                       patch_attempts, roi_delta, retry_count, metadata
                  FROM workflow_runs
                 ORDER BY workflow_id ASC, ts ASC, run_id ASC
                """
            ).fetchall()
        records: list[WorkflowRunRecord] = []
        for row in rows:
            metadata = None
            if row[7]:
                try:
                    metadata = json.loads(row[7])
                except json.JSONDecodeError:
                    metadata = None
            records.append(
                WorkflowRunRecord(
                    run_id=row[0],
                    workflow_id=row[1],
                    ts=row[2],
                    error_classification=row[3],
                    patch_attempts=int(row[4]),
                    roi_delta=float(row[5]),
                    retry_count=int(row[6]),
                    metadata=metadata,
                )
            )
        return records

    def deterministic_retry_order(self) -> list[str]:
        """Return workflow IDs ordered to keep retries deterministic."""
        latest: dict[str, WorkflowRunRecord] = {}
        for record in self.list_runs():
            latest[record.workflow_id] = record
        ordered = sorted(
            latest.values(),
            key=lambda rec: (rec.retry_count, rec.workflow_id, rec.run_id),
        )
        return [rec.workflow_id for rec in ordered]


_STORE = WorkflowRunStore()


def get_run_store() -> WorkflowRunStore:
    return _STORE


def _module_name_from_path(root: Path, path: Path) -> str:
    return ".".join(path.relative_to(root).with_suffix("").parts)


def _is_excluded(path: Path, excluded_dirs: set[str]) -> bool:
    return any(part in excluded_dirs for part in path.parts)


def discover_workflow_modules(root: Path, *, include_bots: bool = True) -> list[str]:
    excluded = set(DEFAULT_EXCLUDED_DIRS)
    modules: list[str] = []
    for path in root.rglob("workflow_*.py"):
        if path.name == "__init__.py" or _is_excluded(path, excluded):
            continue
        modules.append(_module_name_from_path(root, path))
    if include_bots:
        for path in _iter_bot_modules(root):
            modules.append(_module_name_from_path(root, path))
    return sorted(set(modules))


def seed_workflow_db(
    modules: Sequence[str],
    *,
    workflow_db: WorkflowDB,
    source_menace_id: str,
) -> list[int]:
    workflow_ids: list[int] = []
    for module in sorted(modules):
        record = WorkflowRecord(
            workflow=[module],
            task_sequence=[module],
            title=f"Auto-discovered workflow: {module}",
            description="Auto-seeded workflow module for sandbox self-debugging.",
            tags=["auto", "self-debug"],
            status="pending",
        )
        wid = workflow_db.add(record, source_menace_id=source_menace_id)
        if wid:
            workflow_ids.append(wid)
    return workflow_ids


def classify_error(error_text: str | None) -> str:
    if not error_text:
        return "none"
    lowered = error_text.lower()
    if "timeout" in lowered:
        return "timeout"
    if "validation" in lowered or "invalid" in lowered:
        return "validation"
    return "exception"


__all__ = [
    "WorkflowRunRecord",
    "WorkflowRunStore",
    "get_run_store",
    "discover_workflow_modules",
    "seed_workflow_db",
    "classify_error",
]
