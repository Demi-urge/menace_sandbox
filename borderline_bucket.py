import argparse
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

DB_PATH = os.environ.get("BORDERLINE_BUCKET_DB", "borderline_bucket.db")


def _get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def _init_db() -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                workflow_id TEXT PRIMARY KEY,
                raroi REAL,
                confidence REAL,
                status TEXT,
                outcomes TEXT
            )
            """
        )


# Initialise database on import
_init_db()


def add_candidate(workflow_id: str, raroi: float, confidence: float) -> None:
    """Enqueue a candidate workflow.

    If the workflow already exists, its ``raroi`` and ``confidence`` are
    updated while preserving status and outcomes. Newly inserted candidates
    start with ``status`` of ``queued`` and an empty outcomes list.
    """
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT outcomes, status FROM candidates WHERE workflow_id=?",
            (workflow_id,),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO candidates(workflow_id, raroi, confidence, status, outcomes) "
                "VALUES (?, ?, ?, ?, ?)",
                (workflow_id, raroi, confidence, "queued", json.dumps([])),
            )
        else:
            cur.execute(
                "UPDATE candidates SET raroi=?, confidence=? WHERE workflow_id=?",
                (raroi, confidence, workflow_id),
            )
        conn.commit()


def record_outcome(workflow_id: str, passed: bool) -> None:
    """Record a test outcome for ``workflow_id``."""
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT outcomes FROM candidates WHERE workflow_id=?",
            (workflow_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"Unknown workflow_id {workflow_id}")
        outcomes = json.loads(row[0]) if row[0] else []
        outcomes.append(bool(passed))
        cur.execute(
            "UPDATE candidates SET outcomes=? WHERE workflow_id=?",
            (json.dumps(outcomes), workflow_id),
        )
        conn.commit()


def _set_status(workflow_id: str, status: str) -> None:
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE candidates SET status=? WHERE workflow_id=?",
            (status, workflow_id),
        )
        if cur.rowcount == 0:
            raise KeyError(f"Unknown workflow_id {workflow_id}")
        conn.commit()


def promote(workflow_id: str) -> None:
    """Mark ``workflow_id`` as promoted."""
    _set_status(workflow_id, "promoted")


def terminate(workflow_id: str) -> None:
    """Mark ``workflow_id`` as terminated."""
    _set_status(workflow_id, "terminated")


def list_candidates(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return all stored candidates, optionally filtered by ``status``."""
    with _get_conn() as conn:
        cur = conn.cursor()
        if status is None:
            cur.execute(
                "SELECT workflow_id, raroi, confidence, status, outcomes FROM candidates"
            )
        else:
            cur.execute(
                "SELECT workflow_id, raroi, confidence, status, outcomes FROM candidates WHERE status=?",
                (status,),
            )
        rows = cur.fetchall()
    result: List[Dict[str, Any]] = []
    for wid, raroi, confidence, st, outcomes_json in rows:
        outcomes = json.loads(outcomes_json) if outcomes_json else []
        result.append(
            {
                "workflow_id": wid,
                "raroi": raroi,
                "confidence": confidence,
                "status": st,
                "outcomes": outcomes,
            }
        )
    return result


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Manage borderline bucket")
    sub = parser.add_subparsers(dest="command", required=True)

    add_p = sub.add_parser("add", help="enqueue a candidate")
    add_p.add_argument("workflow_id")
    add_p.add_argument("raroi", type=float)
    add_p.add_argument("confidence", type=float)

    outcome_p = sub.add_parser("outcome", help="record test outcome")
    outcome_p.add_argument("workflow_id")
    outcome_p.add_argument("passed", type=_parse_bool)

    promote_p = sub.add_parser("promote", help="promote a workflow")
    promote_p.add_argument("workflow_id")

    terminate_p = sub.add_parser("terminate", help="terminate a workflow")
    terminate_p.add_argument("workflow_id")

    list_p = sub.add_parser("list", help="list workflows")
    list_p.add_argument("--status", choices=["queued", "promoted", "terminated"])

    args = parser.parse_args(argv)

    if args.command == "add":
        add_candidate(args.workflow_id, args.raroi, args.confidence)
    elif args.command == "outcome":
        record_outcome(args.workflow_id, args.passed)
    elif args.command == "promote":
        promote(args.workflow_id)
    elif args.command == "terminate":
        terminate(args.workflow_id)
    elif args.command == "list":
        print(json.dumps(list_candidates(args.status), indent=2))


if __name__ == "__main__":
    main()
