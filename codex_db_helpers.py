from __future__ import annotations

"""Utilities for fetching training samples from various menace databases.

This module standardises retrieval of textual training data from multiple
internal SQLite databases. Each fetch function returns a list of
:class:`TrainingSample` objects that contain the raw text, metadata, source and
optional embedding vectors.
"""

from dataclasses import dataclass
import json
import logging
import sqlite3
from typing import Any, Callable, Dict, List, Optional, Sequence, Sequence as Seq

from chatgpt_enhancement_bot import EnhancementDB
from workflow_summary_db import WorkflowSummaryDB
from discrepancy_db import DiscrepancyDB
from evolution_history_db import EvolutionHistoryDB
from scope_utils import Scope, apply_scope_to_query

logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """Container representing a single training record."""

    text: str
    metadata: Dict[str, Any]
    source: str
    score: Optional[float]
    ts: Optional[str]
    vector: Optional[List[float]] = None


def _run_query(conn: sqlite3.Connection, sql: str, params: Sequence[Any]) -> List[sqlite3.Row]:
    """Execute ``sql`` and return fetched rows while handling DB errors."""

    try:
        cur = conn.execute(sql, params)
        return cur.fetchall()
    except sqlite3.OperationalError as exc:  # pragma: no cover - simple wrapper
        msg = str(exc).lower()
        if "no such table" in msg:
            raise RuntimeError(f"required table missing: {exc}") from exc
        if "no such column" in msg:
            raise RuntimeError(f"required column missing: {exc}") from exc
        raise


# ---------------------------------------------------------------------------
# Enhancement samples
# ---------------------------------------------------------------------------

def fetch_enhancement_samples(
    limit: int = 100,
    sort_by: str = "timestamp",
    with_vectors: bool = False,
) -> List[TrainingSample]:
    """Return enhancement summaries as :class:`TrainingSample` objects."""

    db = EnhancementDB()
    key = sort_by.lower()
    if key in {"ts", "timestamp"}:
        order_col = "timestamp"
    elif key == "outcome_score":
        order_col = "score"
    elif key == "confidence":
        raise ValueError("EnhancementDB does not support sorting by confidence")
    else:  # pragma: no cover - defensive
        raise ValueError(f"unsupported sort field: {sort_by}")

    base = "SELECT id, summary, score, timestamp as ts FROM enhancements"
    sql, params = apply_scope_to_query(base, scope=Scope.ALL, menace_id="")
    sql += f" ORDER BY {order_col} DESC LIMIT ?"
    params.append(limit)
    rows = _run_query(db.conn, sql, params)

    samples: List[TrainingSample] = []
    for row in rows:
        vector = db.vector(row["id"]) if with_vectors and hasattr(db, "vector") else None
        meta = {"id": row["id"]}
        samples.append(
            TrainingSample(
                text=row["summary"],
                metadata=meta,
                source="enhancement",
                score=row["score"],
                ts=row["ts"],
                vector=vector,
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Workflow summary samples
# ---------------------------------------------------------------------------

def fetch_workflow_summary_samples(
    limit: int = 100,
    sort_by: str = "timestamp",
    with_vectors: bool = False,
) -> List[TrainingSample]:
    """Return workflow summaries as :class:`TrainingSample` objects.

    The underlying database does not currently expose scores or timestamps, so
    records are ordered by ``workflow_id`` when ``sort_by`` requests a
    timestamp-based ordering. Sorting by ``confidence`` or ``outcome_score`` is
    unsupported and will raise :class:`ValueError`.
    """

    db = WorkflowSummaryDB()
    key = sort_by.lower()
    if key in {"ts", "timestamp"}:
        order_col = "workflow_id"
    elif key in {"confidence", "outcome_score"}:
        raise ValueError("Workflow summaries do not support the requested sort order")
    else:  # pragma: no cover - defensive
        raise ValueError(f"unsupported sort field: {sort_by}")

    base = "SELECT workflow_id as id, summary FROM workflow_summaries"
    sql, params = apply_scope_to_query(base, scope=Scope.ALL, menace_id="")
    sql += f" ORDER BY {order_col} DESC LIMIT ?"
    params.append(limit)
    rows = _run_query(db.conn, sql, params)

    if with_vectors:
        logger.warning("WorkflowSummaryDB does not provide vector embeddings")

    samples: List[TrainingSample] = []
    for row in rows:
        samples.append(
            TrainingSample(
                text=row["summary"],
                metadata={"workflow_id": row["id"]},
                source="workflow_summary",
                score=None,
                ts=None,
                vector=None,
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Discrepancy samples
# ---------------------------------------------------------------------------

def fetch_discrepancy_samples(
    limit: int = 100,
    sort_by: str = "timestamp",
    with_vectors: bool = False,
) -> List[TrainingSample]:
    """Return discrepancy records as :class:`TrainingSample` objects."""

    db = DiscrepancyDB()
    key = sort_by.lower()
    if key in {"ts", "timestamp"}:
        order_expr = "ts"
    elif key == "outcome_score":
        order_expr = "json_extract(metadata,'$.outcome_score')"
    elif key == "confidence":
        order_expr = "json_extract(metadata,'$.confidence')"
    else:  # pragma: no cover - defensive
        raise ValueError(f"unsupported sort field: {sort_by}")

    base = "SELECT id, message, metadata, ts FROM discrepancies"
    sql, params = apply_scope_to_query(base, scope=Scope.ALL, menace_id="")
    sql += f" ORDER BY {order_expr} DESC LIMIT ?"
    params.append(limit)
    rows = _run_query(db.conn, sql, params)

    samples: List[TrainingSample] = []
    for row in rows:
        meta: Dict[str, Any] = {}
        if row["metadata"]:
            try:
                meta = json.loads(row["metadata"])
            except json.JSONDecodeError:
                meta = {"raw": row["metadata"]}
        score = meta.get("outcome_score") or meta.get("confidence")
        vector = db.vector(row["id"]) if with_vectors and hasattr(db, "vector") else None
        samples.append(
            TrainingSample(
                text=row["message"],
                metadata=meta,
                source="discrepancy",
                score=score,
                ts=row["ts"],
                vector=vector,
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Evolution history samples
# ---------------------------------------------------------------------------

def fetch_evolution_samples(
    limit: int = 100,
    sort_by: str = "timestamp",
    with_vectors: bool = False,
) -> List[TrainingSample]:
    """Return evolution history entries as :class:`TrainingSample` objects."""

    db = EvolutionHistoryDB()
    key = sort_by.lower()
    if key in {"ts", "timestamp"}:
        order_expr = "ts"
    elif key == "outcome_score":
        order_expr = "COALESCE(roi, performance)"
    elif key == "confidence":
        raise ValueError("Evolution history does not support sorting by confidence")
    else:  # pragma: no cover - defensive
        raise ValueError(f"unsupported sort field: {sort_by}")

    base = (
        "SELECT rowid as id, action, roi, performance, ts, before_metric, after_metric, "
        "predicted_roi, efficiency, bottleneck, patch_id, workflow_id, trending_topic, "
        "reason, \"trigger\", parent_event_id, predicted_class, actual_class "
        "FROM evolution_history"
    )
    sql, params = apply_scope_to_query(base, scope=Scope.ALL, menace_id="")
    sql += f" ORDER BY {order_expr} DESC LIMIT ?"
    params.append(limit)
    rows = _run_query(db.conn, sql, params)

    samples: List[TrainingSample] = []
    for row in rows:
        meta = dict(row)
        text = meta.pop("action", "")
        score = meta.pop("roi", None)
        perf = meta.pop("performance", None)
        if score is None:
            score = perf
        ts = meta.pop("ts", None)
        meta.pop("id", None)
        vector = None
        if with_vectors and hasattr(db, "vector"):
            try:
                vector = db.vector(row["id"])
            except Exception:  # pragma: no cover - dependent on external backend
                logger.warning("vector lookup failed for evolution_history id %s", row["id"])
        samples.append(
            TrainingSample(
                text=text,
                metadata=meta,
                source="evolution_history",
                score=score,
                ts=ts,
                vector=vector,
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_FETCHERS: Dict[str, Callable[[int, str, bool], List[TrainingSample]]] = {
    "enhancement": fetch_enhancement_samples,
    "workflow_summary": fetch_workflow_summary_samples,
    "discrepancy": fetch_discrepancy_samples,
    "evolution": fetch_evolution_samples,
}


def aggregate_samples(
    sources: Seq[str],
    limit_per_source: int = 100,
    sort_by: str = "timestamp",
    with_vectors: bool = False,
) -> List[TrainingSample]:
    """Collect and combine samples from multiple ``sources``.

    Parameters
    ----------
    sources:
        Iterable of source names. Valid values are ``"enhancement"``,
        ``"workflow_summary"``, ``"discrepancy"`` and ``"evolution"``.
    limit_per_source:
        Maximum number of samples to pull from each source before merging.
    sort_by:
        Field used for both per-source and global sorting. Supported values are
        ``"confidence"``, ``"outcome_score"`` and ``"ts"``/``"timestamp"``.
    with_vectors:
        If ``True`` attempt to attach embedding vectors to each sample.
    """

    logger.debug(
        "Aggregating samples from %s (limit=%s, sort_by=%s, vectors=%s)",
        sources,
        limit_per_source,
        sort_by,
        with_vectors,
    )

    all_samples: List[TrainingSample] = []
    for src in sources:
        fetcher = _FETCHERS.get(src)
        if not fetcher:
            raise ValueError(f"unknown source: {src}")
        samples = fetcher(limit=limit_per_source, sort_by=sort_by, with_vectors=with_vectors)
        all_samples.extend(samples)

    key_map: Dict[str, Callable[[TrainingSample], Any]] = {
        "confidence": lambda s: s.metadata.get("confidence", 0.0),
        "outcome_score": lambda s: s.score or 0.0,
        "ts": lambda s: s.ts or "",
        "timestamp": lambda s: s.ts or "",
    }
    key = key_map.get(sort_by.lower())
    if key is None:
        raise ValueError(f"unsupported sort field: {sort_by}")
    all_samples.sort(key=key, reverse=True)
    combined_limit = limit_per_source * len(sources)
    return all_samples[:combined_limit]
