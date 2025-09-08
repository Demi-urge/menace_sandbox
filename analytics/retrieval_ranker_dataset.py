from __future__ import annotations

"""Build training data for the retrieval ranker.

This utility inspects ``vector_metrics.db`` and companion databases to assemble
feature rows per ``(session_id, vector_id)`` pair.  For each retrieval the
dataset captures the database type, similarity score, record age, execution
frequency, ROI delta derived from ``roi.db`` and prior hit counts.  The result
can be written to CSV or returned as a NumPy array.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any, Iterable
import argparse

import pandas as pd

from ..db_router import GLOBAL_ROUTER, init_db_router
from ..vector_metrics_db import VectorMetricsDB  # type: ignore
try:  # pragma: no cover - optional dependency
    from ..vector_service import Retriever  # type: ignore
    from ..vector_service.context_builder import ContextBuilder  # type: ignore
except BaseException:  # pragma: no cover - fallback for lightweight environments
    class Retriever:  # type: ignore
        def error_frequency(self, *_args: Any, **_kw: Any) -> float:
            return 0.0

        def workflow_usage(self, *_args: Any, **_kw: Any) -> float:
            return 0.0

        def bot_deploy_freq(self, *_args: Any, **_kw: Any) -> float:
            return 0.0

    class ContextBuilder:  # type: ignore
        def __init__(self) -> None:
            self.retriever = Retriever()


router = GLOBAL_ROUTER or init_db_router("retrieval_ranker_dataset")


# ---------------------------------------------------------------------------
@dataclass
class FeatureRow:
    session_id: str
    vector_id: str
    db_type: str
    similarity: float
    age: float
    exec_freq: float
    roi_delta: float
    prior_hits: int
    alignment_severity: float
    win: float
    regret: float
    hit: int


# ---------------------------------------------------------------------------
def _record_age(db_name: str, vec_id: str, *, now: datetime) -> float:
    """Return age in seconds for a record in ``db_name``."""

    mapping = {
        "error": ("errors", "ts"),
        "workflow": ("workflows", "timestamp"),
        "bot": ("bots", "creation_date"),
    }
    table, col = mapping.get(db_name, (None, None))
    if not table:
        return 0.0
    try:
        conn = router.get_connection(table)
        cur = conn.execute(f"SELECT {col} FROM {table} WHERE id=?", (vec_id,))
        row = cur.fetchone()
        if row and row[0]:
            ts = datetime.fromisoformat(str(row[0]))
            return (now - ts).total_seconds()
    except Exception:
        return 0.0
    return 0.0


# ---------------------------------------------------------------------------
def _exec_metric(retriever: Retriever, db_name: str, vec_id: str) -> float:
    """Return execution frequency/usage metric for ``vec_id``."""

    try:
        vid = int(vec_id)
    except Exception:
        vid = 0
    if db_name == "error":
        return retriever.error_frequency(vid)
    if db_name == "workflow":
        return retriever.workflow_usage(vid)
    if db_name == "bot":
        return retriever.bot_deploy_freq(vid)
    return 0.0


# ---------------------------------------------------------------------------
def _roi_delta(conn: sqlite3.Connection, action_id: Any) -> float:
    """Return ROI delta for ``action_id`` from ``roi.db``.

    The function first attempts to read from a ``roi_events`` table which
    stores ``roi_before``/``roi_after`` pairs.  If that table is absent it falls
    back to the ``action_roi`` table and computes the delta between the two most
    recent ROI entries for the action.
    """

    if not action_id:
        return 0.0
    try:
        aid = str(action_id)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='roi_events'"
        )
        if cur.fetchone():
            cur = conn.execute(
                "SELECT roi_after - roi_before FROM roi_events WHERE action=? ORDER BY ts DESC LIMIT 1",
                (aid,),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                return float(row[0])
        cur = conn.execute(
            "SELECT revenue - api_cost FROM action_roi WHERE action=? ORDER BY ts DESC LIMIT 2",
            (aid,),
        )
        vals = [float(r[0] or 0.0) for r in cur.fetchall()]
        if len(vals) >= 2:
            return float(vals[0] - vals[1])
        if vals:
            return float(vals[0])
    except Exception:
        return 0.0
    return 0.0


# ---------------------------------------------------------------------------
def build_dataset(
    *,
    vec_db_path: str | Path = "vector_metrics.db",
    roi_path: str | Path = "roi.db",
    output_csv: str | Path | None = None,
    as_numpy: bool = False,
) -> pd.DataFrame | Any:
    """Construct the retrieval ranking dataset.

    Parameters
    ----------
    vec_db_path:
        Location of ``vector_metrics.db``.
    roi_path:
        Path to ROI history database used to derive ``roi_delta``.
    output_csv:
        Optional path to write the dataset as CSV.
    as_numpy:
        Return a NumPy array instead of a :class:`pandas.DataFrame`.
    """

    vmdb = VectorMetricsDB(vec_db_path)
    builder = ContextBuilder()
    retriever = builder.retriever
    roi_conn = GLOBAL_ROUTER.get_connection("roi_events")

    now = datetime.utcnow()
    cur = vmdb.conn.execute(
        """
        SELECT vm.session_id,
               vm.vector_id,
               vm.db,
               vm.contribution,
               vm.ts,
               vm.patch_id,
               vm.hit,
               vm.win,
               vm.regret,
               pa.alignment_severity
          FROM vector_metrics AS vm
          LEFT JOIN patch_ancestry AS pa
            ON vm.patch_id = pa.patch_id AND vm.vector_id = pa.vector_id
         WHERE vm.event_type='retrieval'
         ORDER BY vm.ts
        """
    )

    rows: list[FeatureRow] = []
    hit_counts: dict[str, int] = {}
    for (
        session_id,
        vec_id,
        db,
        contrib,
        ts,
        patch_id,
        hit,
        win,
        regret,
        align,
    ) in cur.fetchall():
        if not vec_id or db == "heuristic":
            continue
        prior = hit_counts.get(str(vec_id), 0)
        if hit:
            hit_counts[str(vec_id)] = prior + 1
        age = _record_age(str(db), str(vec_id), now=now)
        freq = _exec_metric(retriever, str(db), str(vec_id))
        roi = _roi_delta(roi_conn, patch_id)
        rows.append(
            FeatureRow(
                session_id=str(session_id),
                vector_id=str(vec_id),
                db_type=str(db),
                similarity=float(contrib or 0.0),
                age=float(age),
                exec_freq=float(freq),
                roi_delta=float(roi),
                prior_hits=int(prior),
                alignment_severity=float(align or 0.0),
                win=float(win or 0.0),
                regret=float(regret or 0.0),
                hit=int(hit or 0),
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
    if as_numpy:
        result: Any = df.to_numpy()
    else:
        result = df
    try:
        roi_conn.close()
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build retrieval ranker dataset")
    p.add_argument("--vec-db", default="vector_metrics.db")
    p.add_argument("--roi-db", default="roi.db")
    p.add_argument("--output-csv", default=None)
    p.add_argument("--as-numpy", action="store_true", help="Return numpy array")
    args = p.parse_args(list(argv) if argv is not None else None)
    data = build_dataset(
        vec_db_path=args.vec_db,
        roi_path=args.roi_db,
        output_csv=args.output_csv,
        as_numpy=args.as_numpy,
    )
    if args.as_numpy:
        print(getattr(data, "shape", None))
    else:
        print(data.head())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
