"""Failure Learning System storing model failures."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

from vector_service import EmbeddableDBMixin

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from . import bot_planning_bot as bpb
from . import capital_management_bot as cmb


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
    """SQLite store for failures and discrepancy detections."""

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
        rid = int(cur.lastrowid)
        try:
            self.add_embedding(rid, asdict(rec), "failure", source_id=str(rid))
        except Exception:  # pragma: no cover - best effort
            pass

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
        rid = f"d{int(cur.lastrowid)}"
        data = {
            "rule": rule,
            "message": message,
            "severity": severity,
            "workflow": workflow or "",
        }
        try:
            self.add_embedding(rid, data, "detection", source_id=str(rid))
        except Exception:  # pragma: no cover - best effort
            pass

    def fetch_detections(self, min_severity: float = 0.0) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT rule, message, severity, workflow, ts FROM detections WHERE severity >= ?",
            self.conn,
            params=(min_severity,),
        )

    # ------------------------------------------------------------------
    def _embed_text(self, data: dict[str, Any]) -> str:
        return " ".join(str(v) for v in data.values() if v not in (None, ""))

    def license_text(self, rec: Any) -> str | None:
        if isinstance(rec, FailureRecord):
            return self._embed_text(asdict(rec))
        if isinstance(rec, dict):
            return self._embed_text(rec)
        return None

    def backfill_embeddings(self, batch_size: int = 100) -> None:
        """Delegate to :class:`EmbeddableDBMixin` for compatibility."""
        EmbeddableDBMixin.backfill_embeddings(self)

    def iter_records(self) -> Iterator[tuple[str, dict[str, Any], str]]:
        """Yield failure and detection rows for embedding backfill."""
        cur = self.conn.execute("SELECT rowid, * FROM failures")
        for row in cur.fetchall():
            data = dict(row)
            rid = str(data.pop("rowid"))
            yield rid, data, "failure"
        cur = self.conn.execute("SELECT rowid, * FROM detections")
        for row in cur.fetchall():
            data = dict(row)
            rid = "d" + str(data.pop("rowid"))
            yield rid, data, "detection"

    def vector(self, rec: Any) -> List[float] | None:
        if isinstance(rec, (int, str)):
            rid = str(rec)
            meta = getattr(self, "_metadata", {}).get(rid)
            if meta and "vector" in meta:
                return meta["vector"]
            if rid.startswith("d"):
                row = self.conn.execute(
                    "SELECT rule, message, severity, workflow FROM detections WHERE rowid=?",
                    (int(rid[1:]),),
                ).fetchone()
            else:
                try:
                    row = self.conn.execute(
                        "SELECT model_id, cause, features, demographics, profitability, retention, cac, roi FROM failures WHERE rowid=?",
                        (int(rid),),
                    ).fetchone()
                except ValueError:
                    return None
            if not row:
                return None
            data = dict(row)
            text = self._embed_text(data)
            return self.encode_text(text) if text else None
        if isinstance(rec, FailureRecord):
            data = asdict(rec)
        elif isinstance(rec, dict):
            data = rec
        else:
            return None
        text = self._embed_text(data)
        return self.encode_text(text) if text else None

    def search_by_vector(
        self, vector: Sequence[float], top_k: int = 5
    ) -> List[dict[str, Any]]:
        matches = EmbeddableDBMixin.search_by_vector(self, vector, top_k)
        results: List[dict[str, Any]] = []
        for rid, dist in matches:
            if str(rid).startswith("d"):
                row = self.conn.execute(
                    "SELECT rule, message, severity, workflow FROM detections WHERE rowid=?",
                    (int(str(rid)[1:]),),
                ).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT model_id, cause, features, demographics, profitability, retention, cac, roi FROM failures WHERE rowid=?",
                    (int(rid),),
                ).fetchone()
            if row:
                data = dict(row)
                data["_distance"] = dist
                results.append(data)
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
