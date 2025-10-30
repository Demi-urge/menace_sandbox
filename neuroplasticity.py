"""Log and reinforce action pathways for Menace."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple
from contextlib import closing
import logging

logger = logging.getLogger(__name__)

from db_router import GLOBAL_ROUTER, init_db_router
from .unified_event_bus import UnifiedEventBus


class Outcome(StrEnum):
    """Possible execution results for a pathway."""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILURE = "FAILURE"


@dataclass
class PathwayRecord:
    """Single execution trace for a Menace task."""

    actions: str
    inputs: str
    outputs: str
    exec_time: float
    resources: str
    outcome: Outcome
    roi: float
    ts: str = datetime.utcnow().isoformat()


class PathwayDB:
    """SQLite-backed store tracking pathway statistics."""

    def __init__(
        self,
        path: Path | str = "pathways.db",
        half_life_days: int = 30,
        *,
        event_bus: "UnifiedEventBus" | None = None,
    ) -> None:
        router = GLOBAL_ROUTER or init_db_router("neuroplasticity", str(path), str(path))
        self.conn = router.get_connection("metadata")
        self.half_life = half_life_days
        self.event_bus = event_bus
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pathways(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                actions TEXT,
                inputs TEXT,
                outputs TEXT,
                exec_time REAL,
                resources TEXT,
                outcome TEXT,
                roi REAL,
                ts TEXT
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pathways_ts ON pathways(ts)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pathways_outcome ON pathways(outcome)"
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata(
                pathway_id INTEGER PRIMARY KEY,
                frequency INTEGER,
                avg_exec_time REAL,
                success_rate REAL,
                avg_roi REAL,
                last_activation TEXT,
                myelination_score REAL,
                FOREIGN KEY(pathway_id) REFERENCES pathways(id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS links(
                from_id INTEGER,
                to_id INTEGER,
                weight REAL,
                PRIMARY KEY(from_id, to_id),
                FOREIGN KEY(from_id) REFERENCES pathways(id),
                FOREIGN KEY(to_id) REFERENCES pathways(id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ngrams(
                n INTEGER,
                seq TEXT,
                weight REAL,
                PRIMARY KEY(n, seq)
            )
            """
        )
        self.conn.commit()

    def _fetchone(self, query: str, params: Sequence[Any] | None = None):
        with closing(self.conn.cursor()) as cur:
            cur.execute(query, params or ())
            return cur.fetchone()

    def _fetchall(self, query: str, params: Sequence[Any] | None = None):
        with closing(self.conn.cursor()) as cur:
            cur.execute(query, params or ())
            return cur.fetchall()

    def log(self, rec: PathwayRecord) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO pathways(actions, inputs, outputs, exec_time, resources, outcome, roi, ts)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                rec.actions,
                rec.inputs,
                rec.outputs,
                rec.exec_time,
                rec.resources,
                rec.outcome.value if isinstance(rec.outcome, Outcome) else str(rec.outcome),
                rec.roi,
                rec.ts,
            ),
        )
        pid = int(cur.lastrowid)
        self._update_meta(pid, rec)
        self.conn.commit()
        if self.event_bus:
            try:
                self.event_bus.publish("pathway:new", rec.__dict__)
            except Exception as exc:
                logger.warning("event bus publish failed: %s", exc)
        return pid

    def _update_meta(self, pid: int, rec: PathwayRecord) -> None:
        row = self._fetchone(
            "SELECT frequency, avg_exec_time, success_rate, avg_roi, last_activation FROM metadata WHERE pathway_id=?",
            (pid,),
        )
        if isinstance(rec.outcome, Outcome):
            oc = rec.outcome
        else:
            try:
                oc = Outcome(str(rec.outcome).upper())
            except Exception:
                oc = Outcome.FAILURE
        if oc is Outcome.SUCCESS:
            success = 1.0
        elif oc is Outcome.PARTIAL_SUCCESS:
            success = 0.5
        else:
            success = 0.0
        if row:
            freq, avg_time, suc_rate, avg_roi, last_ts = row
            freq += 1
            avg_time = (avg_time * (freq - 1) + rec.exec_time) / freq
            suc_rate = (suc_rate * (freq - 1) + success) / freq
            avg_roi = (avg_roi * (freq - 1) + rec.roi) / freq
        else:
            freq = 1
            avg_time = rec.exec_time
            suc_rate = success
            avg_roi = rec.roi
            last_ts = rec.ts

        try:
            last_dt = datetime.fromisoformat(last_ts)
        except Exception:
            last_dt = datetime.utcnow()
        now_dt = datetime.fromisoformat(rec.ts)
        elapsed = (now_dt - last_dt).total_seconds() / 86400.0
        decay = 0.5 ** (elapsed / self.half_life)
        score = freq * suc_rate * avg_roi * decay

        self.conn.execute(
            "REPLACE INTO metadata(pathway_id, frequency, avg_exec_time, success_rate, avg_roi, last_activation, myelination_score) VALUES(?,?,?,?,?,?,?)",
            (pid, freq, avg_time, suc_rate, avg_roi, rec.ts, score),
        )

    def reinforce_link(self, from_id: int, to_id: int, weight: float = 1.0) -> None:
        row = self._fetchone(
            "SELECT weight FROM links WHERE from_id=? AND to_id=?",
            (from_id, to_id),
        )
        new_w = (row[0] if row else 0.0) + weight
        self.conn.execute(
            "REPLACE INTO links(from_id, to_id, weight) VALUES(?,?,?)",
            (from_id, to_id, new_w),
        )
        self.conn.commit()

    def record_sequence(self, ids: Iterable[int]) -> None:
        """Reinforce consecutive pathway links based on execution order."""
        prev: int | None = None
        ids_list = list(ids)
        for pid in ids_list:
            if prev is not None:
                self.reinforce_link(prev, pid)
            prev = pid
        if len(ids_list) >= 3:
            self._record_ngrams(ids_list, 3)
        self.conn.commit()

    def _record_ngrams(self, ids: List[int], n: int) -> None:
        for i in range(len(ids) - n + 1):
            seq = "-".join(str(s) for s in ids[i : i + n])
            row = self._fetchone(
                "SELECT weight FROM ngrams WHERE n=? AND seq=?",
                (n, seq),
            )
            new_w = (row[0] if row else 0.0) + 1.0
            self.conn.execute(
                "REPLACE INTO ngrams(n, seq, weight) VALUES(?,?,?)",
                (n, seq, new_w),
            )

    def next_pathway(self, pid: int) -> int | None:
        """Return the most strongly linked pathway after *pid*."""
        row = self._fetchone(
            "SELECT to_id FROM links WHERE from_id=? ORDER BY weight DESC LIMIT 1",
            (pid,),
        )
        return int(row[0]) if row else None

    def is_highly_myelinated(self, pid: int, threshold: float = 1.0) -> bool:
        """Return True if the pathway's score exceeds *threshold*."""
        row = self._fetchone(
            "SELECT myelination_score FROM metadata WHERE pathway_id=?",
            (pid,),
        )
        return bool(row and row[0] >= threshold)

    def similar_actions(self, actions: str, limit: int = 5) -> List[Tuple[int, float]]:
        """Return pathways with similar action traces."""
        rows = self._fetchall(
            "SELECT id FROM pathways WHERE actions LIKE ?",
            (f"%{actions}%",),
        )
        ids = [r[0] for r in rows]
        if not ids:
            return []
        qmarks = ",".join("?" for _ in ids)
        return self._fetchall(
            f"SELECT pathway_id, myelination_score FROM metadata WHERE pathway_id IN ({qmarks}) ORDER BY myelination_score DESC LIMIT ?",
            (*ids, limit),
        )

    def top_pathways(self, limit: int = 5) -> List[Tuple[int, float]]:
        return self._fetchall(
            "SELECT pathway_id, myelination_score FROM metadata ORDER BY myelination_score DESC LIMIT ?",
            (limit,),
        )

    def top_sequences(self, n: int = 3, limit: int = 5) -> List[Tuple[str, float]]:
        return self._fetchall(
            "SELECT seq, weight FROM ngrams WHERE n=? ORDER BY weight DESC LIMIT ?",
            (n, limit),
        )

    def highest_myelination_score(self) -> float:
        """Return the highest myelination score currently recorded."""
        row = self._fetchone("SELECT MAX(myelination_score) FROM metadata")
        return float(row[0] or 0.0)

    def merge_macro_pathways(self, weight_threshold: float = 3.0) -> None:
        """Create combined pathways for strongly linked pairs."""
        pairs = self._fetchall(
            "SELECT from_id, to_id FROM links WHERE weight>=?",
            (weight_threshold,),
        )
        for from_id, to_id in pairs:
            row1 = self._fetchone(
                "SELECT actions FROM pathways WHERE id=?",
                (from_id,),
            )
            row2 = self._fetchone(
                "SELECT actions FROM pathways WHERE id=?",
                (to_id,),
            )
            if not row1 or not row2:
                continue
            actions = f"{row1[0]}->{row2[0]}"
            metas = self._fetchall(
                "SELECT frequency, avg_exec_time, success_rate, avg_roi FROM metadata WHERE pathway_id IN (?,?)",
                (from_id, to_id),
            )
            if not metas:
                continue
            freq = sum(m[0] for m in metas)
            if freq <= 0:
                continue
            avg_time = sum(m[1] * m[0] for m in metas) / freq
            suc = sum(m[2] * m[0] for m in metas) / freq
            avg_roi = sum(m[3] * m[0] for m in metas) / freq
            rec = PathwayRecord(
                actions=actions,
                inputs="",
                outputs="",
                exec_time=avg_time,
                resources="",
                outcome=Outcome.SUCCESS if suc >= 0.5 else Outcome.FAILURE,
                roi=avg_roi,
            )
            self.log(rec)
        self.conn.commit()


__all__ = ["Outcome", "PathwayRecord", "PathwayDB"]
