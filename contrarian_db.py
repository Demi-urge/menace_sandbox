"""Contrarian Database for storing experimental workflows."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List


@dataclass
class ContrarianRecord:
    """Single contrarian experiment entry."""

    innovation_name: str = ""
    innovation_type: str = ""
    risk_score: float = 0.0
    reward_score: float = 0.0
    activation_trigger: str = ""
    resource_allocation: Dict[str, float] = field(default_factory=dict)
    timestamp_created: str = datetime.utcnow().isoformat()
    timestamp_last_evaluated: str = ""
    status: str = "active"
    contrarian_id: Optional[int] = None


class ContrarianDB:
    """SQLite-backed storage for contrarian experiment history."""

    def __init__(self, path: Path | str = "contrarian.db") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS contrarian_experiments(
                contrarian_id INTEGER PRIMARY KEY AUTOINCREMENT,
                innovation_name TEXT,
                innovation_type TEXT,
                risk_score REAL,
                reward_score REAL,
                activation_trigger TEXT,
                resource_allocation TEXT,
                timestamp_created TEXT,
                timestamp_last_evaluated TEXT,
                status TEXT
            )
            """
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS contrarian_models(contrarian_id INTEGER, model_id INTEGER)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS contrarian_workflows(contrarian_id INTEGER, workflow_id INTEGER)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS contrarian_enhancements(contrarian_id INTEGER, enhancement_id INTEGER)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS contrarian_errors(contrarian_id INTEGER, error_id INTEGER)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS contrarian_discrepancies(contrarian_id INTEGER, discrepancy_id INTEGER)"
        )
        self.conn.commit()

    def get(self, contrarian_id: int) -> Optional[ContrarianRecord]:
        """Retrieve a single contrarian record by id."""
        self.conn.row_factory = sqlite3.Row
        row = self.conn.execute(
            "SELECT * FROM contrarian_experiments WHERE contrarian_id=?",
            (contrarian_id,),
        ).fetchone()
        if not row:
            return None
        return ContrarianRecord(
            innovation_name=row["innovation_name"],
            innovation_type=row["innovation_type"],
            risk_score=row["risk_score"],
            reward_score=row["reward_score"],
            activation_trigger=row["activation_trigger"],
            resource_allocation=json.loads(row["resource_allocation"] or "{}"),
            timestamp_created=row["timestamp_created"],
            timestamp_last_evaluated=row["timestamp_last_evaluated"],
            status=row["status"],
            contrarian_id=row["contrarian_id"],
        )

    def update_timestamp(self, contrarian_id: int, ts: str | None = None) -> None:
        """Update the last evaluated timestamp for a contrarian strategy."""
        ts = ts or datetime.utcnow().isoformat()
        self.conn.execute(
            "UPDATE contrarian_experiments SET timestamp_last_evaluated=? WHERE contrarian_id=?",
            (ts, contrarian_id),
        )
        self.conn.commit()

    def add(self, rec: ContrarianRecord) -> int:
        alloc = json.dumps(rec.resource_allocation)
        cur = self.conn.execute(
            """
            INSERT INTO contrarian_experiments(
                innovation_name,
                innovation_type,
                risk_score,
                reward_score,
                activation_trigger,
                resource_allocation,
                timestamp_created,
                timestamp_last_evaluated,
                status
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                rec.innovation_name,
                rec.innovation_type,
                rec.risk_score,
                rec.reward_score,
                rec.activation_trigger,
                alloc,
                rec.timestamp_created,
                rec.timestamp_last_evaluated,
                rec.status,
            ),
        )
        contrarian_id = int(cur.lastrowid)
        self.conn.commit()
        return contrarian_id

    def link_model(self, contrarian_id: int, model_id: int) -> None:
        self.conn.execute(
            "INSERT INTO contrarian_models (contrarian_id, model_id) VALUES (?, ?)",
            (contrarian_id, model_id),
        )
        self.conn.commit()

    def link_workflow(self, contrarian_id: int, workflow_id: int) -> None:
        self.conn.execute(
            "INSERT INTO contrarian_workflows (contrarian_id, workflow_id) VALUES (?, ?)",
            (contrarian_id, workflow_id),
        )
        self.conn.commit()

    def link_enhancement(self, contrarian_id: int, enhancement_id: int) -> None:
        self.conn.execute(
            "INSERT INTO contrarian_enhancements (contrarian_id, enhancement_id) VALUES (?, ?)",
            (contrarian_id, enhancement_id),
        )
        self.conn.commit()

    def link_error(self, contrarian_id: int, error_id: int) -> None:
        self.conn.execute(
            "INSERT INTO contrarian_errors (contrarian_id, error_id) VALUES (?, ?)",
            (contrarian_id, error_id),
        )
        self.conn.commit()

    def link_discrepancy(self, contrarian_id: int, discrepancy_id: int) -> None:
        self.conn.execute(
            "INSERT INTO contrarian_discrepancies (contrarian_id, discrepancy_id) VALUES (?, ?)",
            (contrarian_id, discrepancy_id),
        )
        self.conn.commit()

    def update_status(self, contrarian_id: int, status: str) -> None:
        """Update the status of a contrarian experiment."""
        self.conn.execute(
            "UPDATE contrarian_experiments SET status=? WHERE contrarian_id=?",
            (status, contrarian_id),
        )
        self.conn.commit()

    def models_for(self, contrarian_id: int) -> List[int]:
        rows = self.conn.execute(
            "SELECT model_id FROM contrarian_models WHERE contrarian_id=?",
            (contrarian_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def workflows_for(self, contrarian_id: int) -> List[int]:
        rows = self.conn.execute(
            "SELECT workflow_id FROM contrarian_workflows WHERE contrarian_id=?",
            (contrarian_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def enhancements_for(self, contrarian_id: int) -> List[int]:
        rows = self.conn.execute(
            "SELECT enhancement_id FROM contrarian_enhancements WHERE contrarian_id=?",
            (contrarian_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def errors_for(self, contrarian_id: int) -> List[int]:
        rows = self.conn.execute(
            "SELECT error_id FROM contrarian_errors WHERE contrarian_id=?",
            (contrarian_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def discrepancies_for(self, contrarian_id: int) -> List[int]:
        rows = self.conn.execute(
            "SELECT discrepancy_id FROM contrarian_discrepancies WHERE contrarian_id=?",
            (contrarian_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def fetch(self, limit: int = 50) -> List[ContrarianRecord]:
        self.conn.row_factory = sqlite3.Row
        rows = self.conn.execute(
            "SELECT * FROM contrarian_experiments ORDER BY contrarian_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results: List[ContrarianRecord] = []
        for row in rows:
            results.append(
                ContrarianRecord(
                    innovation_name=row["innovation_name"],
                    innovation_type=row["innovation_type"],
                    risk_score=row["risk_score"],
                    reward_score=row["reward_score"],
                    activation_trigger=row["activation_trigger"],
                    resource_allocation=json.loads(row["resource_allocation"] or "{}"),
                    timestamp_created=row["timestamp_created"],
                    timestamp_last_evaluated=row["timestamp_last_evaluated"],
                    status=row["status"],
                    contrarian_id=row["contrarian_id"],
                )
            )
        return results


__all__ = [
    "ContrarianRecord",
    "ContrarianDB",
]
