"""Failure Learning System storing model failures."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Sequence
import importlib

from vector_service import EmbeddableDBMixin
from db_router import DBRouter, GLOBAL_ROUTER, init_db_router
from scope_utils import Scope, build_scope_clause

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - imported for type annotations only
    from .bot_planning_bot import BotPlanningBot
    from .capital_management_bot import CapitalManagementBot


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
        router: DBRouter | None = None,
        vector_index_path: Path | str | None = None,
        embedding_version: int = 1,
        vector_backend: str = "annoy",
    ) -> None:
        p = Path(path).resolve()
        self.router = router or GLOBAL_ROUTER or init_db_router(
            "failures_db", str(p), str(p)
        )
        self.conn = self.router.get_connection("failures")
        self.dconn = self.router.get_connection("detections")
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
        self.dconn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections(
                rule TEXT,
                message TEXT,
                severity REAL,
                workflow TEXT,
                source_menace_id TEXT NOT NULL,
                ts TEXT
            )
            """,
        )
        cols = [r[1] for r in self.dconn.execute("PRAGMA table_info(detections)").fetchall()]
        if "source_menace_id" not in cols:
            self.dconn.execute(
                "ALTER TABLE detections ADD COLUMN source_menace_id TEXT NOT NULL DEFAULT ''"
            )
            self.dconn.execute(
                "UPDATE detections SET source_menace_id='' WHERE source_menace_id IS NULL"
            )
        self.dconn.execute(
            "CREATE INDEX IF NOT EXISTS idx_detections_source_menace_id ON detections(source_menace_id)"
        )
        self.conn.commit()
        self.dconn.commit()
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
        self,
        rule: str,
        severity: float,
        message: str,
        workflow: str | None = None,
        *,
        source_menace_id: str | None = None,
    ) -> None:
        menace_id = source_menace_id or getattr(self.router, "menace_id", "")
        cur = self.dconn.execute(
            "INSERT INTO detections(rule, message, severity, workflow, source_menace_id, ts) VALUES(?,?,?,?,?,?)",
            (
                rule,
                message,
                severity,
                workflow or "",
                menace_id,
                datetime.utcnow().isoformat(),
            ),
        )
        self.dconn.commit()
        rid = f"d{int(cur.lastrowid)}"
        data = {
            "rule": rule,
            "message": message,
            "severity": severity,
            "workflow": workflow or "",
            "source_menace_id": menace_id,
        }
        try:
            self.add_embedding(rid, data, "detection", source_id=str(rid))
        except Exception:  # pragma: no cover - best effort
            pass

    def fetch_detections(
        self,
        min_severity: float = 0.0,
        *,
        scope: Scope | str = Scope.LOCAL,
        source_menace_id: str | None = None,
    ) -> pd.DataFrame:
        menace_id = source_menace_id or getattr(self.router, "menace_id", "")
        clause, scope_params = build_scope_clause("detections", scope, menace_id)
        sql = (
            "SELECT rule, message, severity, workflow, ts FROM detections WHERE severity >= ?"
        )
        params = [min_severity]
        if clause:
            sql += f" AND {clause}"
            params.extend(scope_params)
        return pd.read_sql(sql, self.dconn, params=params)

    # ------------------------------------------------------------------
    def _embed_text(self, data: dict[str, Any]) -> str:
        return " ".join(str(v) for v in data.values() if v not in (None, ""))

    def license_text(self, rec: Any) -> str | None:
        if isinstance(rec, FailureRecord):
            return self._embed_text(asdict(rec))
        if isinstance(rec, dict):
            return self._embed_text(rec)
        return None

    def log_license_violation(self, path: str, license_name: str, hash_: str) -> None:
        try:  # pragma: no cover - best effort
            CodeDB = importlib.import_module("code_database").CodeDB
            CodeDB().log_license_violation(path, license_name, hash_)
        except Exception:
            pass

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
        cur = self.dconn.execute("SELECT rowid, * FROM detections")
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
                row = self.dconn.execute(
                    "SELECT rule, message, severity, workflow FROM detections WHERE rowid=?",
                    (int(rid[1:]),),
                ).fetchone()
            else:
                try:
                    row = self.conn.execute(
                        (
                            "SELECT model_id, cause, features, demographics, profitability, "
                            "retention, cac, roi FROM failures WHERE rowid=?"
                        ),
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
                row = self.dconn.execute(
                    "SELECT rule, message, severity, workflow FROM detections WHERE rowid=?",
                    (int(str(rid)[1:]),),
                ).fetchone()
            else:
                row = self.conn.execute(
                    (
                        "SELECT model_id, cause, features, demographics, profitability, "
                        "retention, cac, roi FROM failures WHERE rowid=?"
                    ),
                    (int(rid),),
                ).fetchone()
            if row:
                data = dict(row)
                data["_distance"] = dist
                results.append(data)
        return results


class FailureDB(DiscrepancyDB):
    """Expose failure embeddings for the vector service."""

    DB_FILE = "failures.db"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # provide mapping access for ``dict(row)`` calls
        self.conn.row_factory = sqlite3.Row

    def iter_records(self) -> Iterator[tuple[str, dict[str, Any], str]]:
        """Yield only failure rows for embedding backfill."""
        cur = self.conn.execute("SELECT rowid, * FROM failures")
        for row in cur.fetchall():
            data = dict(row)
            rid = str(data.pop("rowid"))
            yield rid, data, "failure"


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

    def advise_planner(self, planner: "BotPlanningBot") -> List[str]:
        counts = self.risk_features()
        return [feat for feat, c in counts.items() if c > 1]

    def advise_capital(self, manager: "CapitalManagementBot", model_id: str) -> float:
        return self.failure_score(model_id)


__all__ = ["FailureRecord", "DiscrepancyDB", "FailureDB", "FailureLearningSystem"]
