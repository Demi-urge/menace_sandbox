"""Contrarian Database for storing experimental workflows."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from db_router import DBRouter, GLOBAL_ROUTER, LOCAL_TABLES
from vector_service import EmbeddableDBMixin

logger = logging.getLogger(__name__)


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


class ContrarianDB(EmbeddableDBMixin):
    """SQLite-backed storage for contrarian experiment history with embeddings."""

    def __init__(
        self,
        path: Path | str = "contrarian.db",
        *,
        router: DBRouter | None = None,
        vector_index_path: str | Path | None = None,
        embedding_version: int = 1,
        vector_backend: str = "annoy",
    ) -> None:
        LOCAL_TABLES.update(
            {
                "contrarian_experiments",
                "contrarian_models",
                "contrarian_workflows",
                "contrarian_enhancements",
                "contrarian_errors",
                "contrarian_discrepancies",
            }
        )

        self.router = router or GLOBAL_ROUTER or DBRouter("contrarian", str(path), str(path))
        self.conn = self.router.get_connection("contrarian_experiments")
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
            "CREATE TABLE IF NOT EXISTS contrarian_models(contrarian_id INTEGER, model_id INTEGER)"  # noqa: E501
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS contrarian_workflows(contrarian_id INTEGER, workflow_id INTEGER)"  # noqa: E501
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS contrarian_enhancements(contrarian_id INTEGER, enhancement_id INTEGER)"  # noqa: E501
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS contrarian_errors(contrarian_id INTEGER, error_id INTEGER)"  # noqa: E501
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS contrarian_discrepancies(contrarian_id INTEGER, discrepancy_id INTEGER)"  # noqa: E501
        )
        self.conn.commit()

        index_path = (
            Path(vector_index_path)
            if vector_index_path is not None
            else Path(path).with_suffix(".index")
        )
        meta_path = Path(index_path).with_suffix(".json")
        EmbeddableDBMixin.__init__(
            self,
            index_path=index_path,
            metadata_path=meta_path,
            embedding_version=embedding_version,
            backend=vector_backend,
        )

    # ------------------------------------------------------------------
    def _embed_text(self, rec: ContrarianRecord | Dict[str, Any]) -> str:
        if isinstance(rec, ContrarianRecord):
            data = {
                "innovation_name": rec.innovation_name,
                "innovation_type": rec.innovation_type,
                "activation_trigger": rec.activation_trigger,
                "status": rec.status,
            }
        else:
            data = {
                "innovation_name": rec.get("innovation_name", ""),
                "innovation_type": rec.get("innovation_type", ""),
                "activation_trigger": rec.get("activation_trigger", ""),
                "status": rec.get("status", ""),
            }
        parts = [f"{k}={v}" for k, v in data.items() if v]
        return " ".join(parts)

    def license_text(self, rec: Any) -> str | None:
        if isinstance(rec, (ContrarianRecord, dict)):
            return self._embed_text(rec)
        return None

    def vector(self, rec: Any) -> List[float] | None:
        if isinstance(rec, (ContrarianRecord, dict)):
            text = self._embed_text(rec)
            return self.encode_text(text)
        return None

    def _embed_record_on_write(self, rec_id: int, rec: ContrarianRecord) -> None:
        try:
            self.add_embedding(rec_id, rec, "contrarian", source_id=str(rec_id))
        except Exception:  # pragma: no cover - best effort
            logger.exception("embedding hook failed for %s", rec_id)

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
        self._embed_record_on_write(contrarian_id, rec)
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

    # ------------------------------------------------------------------
    def iter_records(self) -> Iterator[tuple[int, Dict[str, Any], str]]:
        cur = self.conn.execute(
            "SELECT contrarian_id, innovation_name, innovation_type, activation_trigger, status FROM contrarian_experiments"  # noqa: E501
        )
        for row in cur.fetchall():
            data = {
                "innovation_name": row[1],
                "innovation_type": row[2],
                "activation_trigger": row[3],
                "status": row[4],
            }
            yield row[0], data, "contrarian"

    def to_vector_dict(self, rec: ContrarianRecord | Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(rec, ContrarianRecord):
            return {
                "innovation_name": rec.innovation_name,
                "innovation_type": rec.innovation_type,
                "activation_trigger": rec.activation_trigger,
                "status": rec.status,
            }
        return {
            "innovation_name": rec.get("innovation_name", ""),
            "innovation_type": rec.get("innovation_type", ""),
            "activation_trigger": rec.get("activation_trigger", ""),
            "status": rec.get("status", ""),
        }

    def backfill_embeddings(self, batch_size: int = 100) -> None:
        EmbeddableDBMixin.backfill_embeddings(self)


__all__ = [
    "ContrarianRecord",
    "ContrarianDB",
]
