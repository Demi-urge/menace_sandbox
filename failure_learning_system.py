"""Failure Learning System storing model failures."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from . import bot_planning_bot as bpb
from . import capital_management_bot as cmb
from vector_service import EmbeddableDBMixin


@dataclass
class FailureRecord:
    """Detailed failure event data."""

    model_id: str
    cause: str
    features: str
    demographics: str
    profitability: float
    retention: float
    cac: float
    roi: float
    ts: str = datetime.utcnow().isoformat()


class DiscrepancyDB(EmbeddableDBMixin):
    """SQLite store for failures and discrepancy detections with embeddings."""

    def __init__(
        self,
        path: Path | str = "failures.db",
        *,
        vector_index_path: Path | str | None = None,
        embedding_version: int = 1,
        vector_backend: str = "annoy",
    ) -> None:
        # allow the connection to be shared across threads because the
        # failure learning system may be accessed by background workers
        # running in different threads
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS failures(
                model_id TEXT,
                cause TEXT,
                features TEXT,
                demographics TEXT,
                profitability REAL,
                retention REAL,
                cac REAL,
                roi REAL,
                ts TEXT
            )
            """,
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections(
                rule TEXT,
                message TEXT,
                severity REAL,
                workflow TEXT,
                ts TEXT
            )
            """,
        )
        self.conn.commit()

        if vector_index_path is None:
            vector_index_path = Path(path).with_suffix(".index")
        meta_path = Path(vector_index_path).with_suffix(".json")
        EmbeddableDBMixin.__init__(
            self,
            index_path=vector_index_path,
            metadata_path=meta_path,
            embedding_version=embedding_version,
            backend=vector_backend,
        )

    # ------------------------------------------------------------------
    def _embed(self, text: str) -> List[float]:
        """Encode ``text`` into a vector (overridable for tests)."""
        return self.encode_text(text)

    def _embed_record(self, rec_id: int, rec: Dict[str, Any], kind: str) -> None:
        """Best-effort embedding hook for inserts."""
        try:
            self.add_embedding(rec_id, rec, kind, source_id=str(rec_id))
        except Exception:  # pragma: no cover - best effort
            logging.exception("embedding hook failed for %s", rec_id)

    def log(self, rec: FailureRecord) -> None:
        cur = self.conn.execute(
            "INSERT INTO failures VALUES(?,?,?,?,?,?,?,?,?)",
            (
                rec.model_id,
                rec.cause,
                rec.features,
                rec.demographics,
                rec.profitability,
                rec.retention,
                rec.cac,
                rec.roi,
                rec.ts,
            ),
        )
        self.conn.commit()
        rec_id = int(cur.lastrowid)
        self._embed_record(rec_id, rec.__dict__, "failure")

    def fetch(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM failures", self.conn)

    def log_detection(
        self, rule: str, severity: float, message: str, workflow: str | None = None
    ) -> None:
        cur = self.conn.execute(
            "INSERT INTO detections(rule, message, severity, workflow, ts) VALUES(?,?,?,?,?)",
            (
                rule,
                message,
                severity,
                workflow or "",
                datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()
        rec = {
            "rule": rule,
            "message": message,
            "severity": severity,
            "workflow": workflow or "",
        }
        self._embed_record(int(cur.lastrowid), rec, "detection")

    def fetch_detections(self, min_severity: float = 0.0) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT rule, message, severity, workflow, ts FROM detections WHERE severity >= ?",
            self.conn,
            params=(min_severity,),
        )

    # ------------------------------------------------------------------
    def vector(self, rec: Any) -> List[float] | None:
        """Return an embedding for ``rec``."""
        if isinstance(rec, int):
            meta = self._metadata.get(str(rec))
            if meta and "vector" in meta:
                return list(meta["vector"])
            return None
        if isinstance(rec, str):
            text = rec
        elif isinstance(rec, dict):
            parts = [
                rec.get("cause") or rec.get("message") or "",
                rec.get("features") or "",
                rec.get("demographics") or "",
            ]
            text = " ".join(p for p in parts if p)
        else:
            msg = getattr(rec, "cause", "") or getattr(rec, "message", "")
            feats = getattr(rec, "features", "")
            demo = getattr(rec, "demographics", "")
            text = " ".join(p for p in [msg, feats, demo] if p)
        return self._embed(text) if text else None

    def iter_records(self) -> Iterator[Tuple[int, Dict[str, Any], str]]:
        """Yield failure and detection records for embedding backfill."""
        cur = self.conn.execute(
            "SELECT rowid, model_id, cause, features, demographics FROM failures"
        )
        for row in cur.fetchall():
            rec = {
                "model_id": row[1],
                "cause": row[2],
                "features": row[3],
                "demographics": row[4],
            }
            yield int(row[0]), rec, "failure"
        cur = self.conn.execute(
            "SELECT rowid, rule, message, severity, workflow FROM detections"
        )
        for row in cur.fetchall():
            rec = {
                "rule": row[1],
                "message": row[2],
                "severity": row[3],
                "workflow": row[4],
            }
            yield int(row[0]), rec, "detection"

    def backfill_embeddings(self, batch_size: int = 100) -> None:
        """Generate embeddings for records lacking them."""
        EmbeddableDBMixin.backfill_embeddings(self)

    def search_by_vector(
        self, vector: Sequence[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        matches = EmbeddableDBMixin.search_by_vector(self, vector, top_k)
        results: List[Dict[str, Any]] = []
        for rec_id, dist in matches:
            rid = int(rec_id)
            meta = self._metadata.get(str(rec_id), {})
            kind = meta.get("kind")
            if kind == "failure":
                row = self.conn.execute(
                    "SELECT model_id, cause, features, demographics, profitability, retention, cac, roi, ts FROM failures WHERE rowid=?",
                    (rid,),
                ).fetchone()
                if row:
                    results.append(
                        {
                            "id": rid,
                            "model_id": row[0],
                            "cause": row[1],
                            "features": row[2],
                            "demographics": row[3],
                            "profitability": row[4],
                            "retention": row[5],
                            "cac": row[6],
                            "roi": row[7],
                            "ts": row[8],
                            "_distance": dist,
                        }
                    )
            elif kind == "detection":
                row = self.conn.execute(
                    "SELECT rule, message, severity, workflow, ts FROM detections WHERE rowid=?",
                    (rid,),
                ).fetchone()
                if row:
                    results.append(
                        {
                            "id": rid,
                            "rule": row[0],
                            "message": row[1],
                            "severity": row[2],
                            "workflow": row[3],
                            "ts": row[4],
                            "_distance": dist,
                        }
                    )
        return results


class FailureLearningSystem:
    """Record failures and derive planning and funding insights."""

    def __init__(self, db: DiscrepancyDB | None = None) -> None:
        self.db = db or DiscrepancyDB()

    def record_failure(self, rec: FailureRecord) -> None:
        self.db.log(rec)

    def risk_features(self) -> Dict[str, int]:
        df = self.db.fetch()
        counts: Dict[str, int] = {}
        if "features" not in df:
            return counts
        for feats in df["features"]:
            for f in feats.split(","):
                f = f.strip()
                if not f:
                    continue
                counts[f] = counts.get(f, 0) + 1
        return counts

    def failure_score(self, model_id: str) -> float:
        df = self.db.fetch()
        total = len(df)
        if total == 0:
            return 0.0
        return float(len(df[df["model_id"] == model_id])) / float(total)

    def advise_planner(self, planner: bpb.BotPlanningBot) -> List[str]:
        counts = self.risk_features()
        return [feat for feat, c in counts.items() if c > 1]

    def advise_capital(self, manager: cmb.CapitalManagementBot, model_id: str) -> float:
        return self.failure_score(model_id)


__all__ = ["FailureRecord", "DiscrepancyDB", "FailureLearningSystem"]
