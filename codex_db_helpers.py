"""Helpers for extracting training samples from Menace databases.

This module exposes lightweight wrappers around several SQLite databases so
training data can be gathered in a consistent format.  Each helper accepts a
``scope`` parameter controlling which Menace instances participate in the
query, defaulting to :class:`~scope_utils.Scope.ALL`.  Optional embeddings are
attached when the underlying database provides a ``vector`` method.

Available fetchers:

``fetch_enhancements`` – rows from :class:`chatgpt_enhancement_bot.EnhancementDB`.
``fetch_summaries``    – records from :class:`workflow_summary_db.WorkflowSummaryDB`.
``fetch_discrepancies`` – entries from :class:`discrepancy_db.DiscrepancyDB`.
``fetch_workflows``    – stored workflows from :class:`task_handoff_bot.WorkflowDB`.

``aggregate_samples`` merges results from all fetchers and returns a single
sorted list.  ``aggregate_examples`` is provided as a backwards compatible
alias.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from datetime import datetime
from typing import Iterable, List, Optional

from .chatgpt_enhancement_bot import EnhancementDB
from .workflow_summary_db import WorkflowSummaryDB
from .discrepancy_db import DiscrepancyDB
from .task_handoff_bot import WorkflowDB as TaskWorkflowDB
from .scope_utils import Scope, build_scope_clause
from dynamic_path_router import resolve_path


logger = logging.getLogger(__name__)

TRAINING_ANOMALY_FILE = resolve_path("training_data/stripe_anomalies.jsonl")


@dataclass
class TrainingSample:
    """Container representing a training example."""

    source: str
    content: str
    confidence: Optional[float] = None
    outcome_score: Optional[float] = None
    timestamp: Optional[str] = None
    embedding: Optional[List[float]] = None


def _fetch_rows(conn, sql: str, params: Iterable[object]) -> List:
    cur = conn.execute(sql, list(params))
    return cur.fetchall()


def _resolve_order(sort_by: str, columns: dict[str, str], default: str) -> str:
    """Map ``sort_by`` to a column name with graceful fallback."""

    return columns.get(sort_by) or columns.get("timestamp") or default


def fetch_enhancements(
    *,
    sort_by: str = "timestamp",
    limit: int = 100,
    include_embeddings: bool = False,
    scope: Scope = Scope.ALL,
) -> List[TrainingSample]:
    """Return enhancement summaries from :class:`EnhancementDB`.

    Args:
        sort_by: Column used for ordering the results.
        limit: Maximum number of rows to return.
        include_embeddings: When ``True``, attach vector embeddings where
            available.
        scope: Menace scope for the query. Defaults to :attr:`Scope.ALL`.
    """

    db = EnhancementDB()
    menace_id = getattr(getattr(db, "router", None), "menace_id", "")
    clause, params = build_scope_clause("enhancements", scope, menace_id)
    columns = {
        "confidence": "confidence",
        "outcome_score": "outcome_score",
        "timestamp": "timestamp",
    }
    order_col = _resolve_order(sort_by, columns, "timestamp")
    sql = (
        "SELECT id, summary, confidence, outcome_score, timestamp FROM enhancements"
    )
    if clause:
        sql += f" WHERE {clause}"
    sql += f" ORDER BY {order_col} DESC LIMIT ?"
    params.append(limit)

    rows = _fetch_rows(db.conn, sql, params)
    samples: List[TrainingSample] = []
    for row in rows:
        embedding: Optional[List[float]] = None
        if include_embeddings and hasattr(db, "vector"):
            try:
                embedding = db.vector(row["id"])
            except Exception:  # pragma: no cover - best effort
                embedding = None
        samples.append(
            TrainingSample(
                source="enhancement",
                content=row["summary"],
                confidence=row["confidence"],
                outcome_score=row["outcome_score"],
                timestamp=row["timestamp"],
                embedding=embedding,
            )
        )
    return samples


def fetch_summaries(
    *,
    sort_by: str = "timestamp",
    limit: int = 100,
    include_embeddings: bool = False,
    scope: Scope = Scope.ALL,
) -> List[TrainingSample]:
    """Return workflow summaries from :class:`WorkflowSummaryDB`.

    Args:
        sort_by: Column used for ordering the results.
        limit: Maximum number of rows to return.
        include_embeddings: When ``True``, attach vector embeddings where
            available.
        scope: Menace scope for the query. Defaults to :attr:`Scope.ALL`.
    """

    db = WorkflowSummaryDB()
    menace_id = getattr(getattr(db, "router", None), "menace_id", "")
    clause, params = build_scope_clause("workflow_summaries", scope, menace_id)
    order_col = _resolve_order(sort_by, {"timestamp": "timestamp"}, "workflow_id")
    sql = "SELECT workflow_id, summary, timestamp FROM workflow_summaries"
    if clause:
        sql += f" WHERE {clause}"
    sql += f" ORDER BY {order_col} DESC LIMIT ?"
    params.append(limit)

    rows = _fetch_rows(db.conn, sql, params)
    samples: List[TrainingSample] = []
    for row in rows:
        embedding: Optional[List[float]] = None
        if include_embeddings and hasattr(db, "vector"):
            try:
                embedding = db.vector(row["workflow_id"])
            except Exception:  # pragma: no cover - best effort
                embedding = None
        samples.append(
            TrainingSample(
                source="workflow_summary",
                content=row["summary"],
                timestamp=row["timestamp"],
                embedding=embedding,
            )
        )
    return samples


def fetch_discrepancies(
    *,
    sort_by: str = "timestamp",
    limit: int = 100,
    include_embeddings: bool = False,
    scope: Scope = Scope.ALL,
) -> List[TrainingSample]:
    """Return discrepancy messages from :class:`DiscrepancyDB`.

    Args:
        sort_by: Column used for ordering the results.
        limit: Maximum number of rows to return.
        include_embeddings: When ``True``, attach vector embeddings where
            available.
        scope: Menace scope for the query. Defaults to :attr:`Scope.ALL`.
    """

    db = DiscrepancyDB()
    menace_id = getattr(getattr(db, "router", None), "menace_id", "")
    clause, params = build_scope_clause("discrepancies", scope, menace_id)
    columns = {
        "confidence": "confidence",
        "outcome_score": "outcome_score",
        "timestamp": "ts",
    }
    order_col = _resolve_order(sort_by, columns, "ts")
    sql = (
        "SELECT id, message, confidence, outcome_score, ts FROM discrepancies"
    )
    if clause:
        sql += f" WHERE {clause}"
    sql += f" ORDER BY {order_col} DESC LIMIT ?"
    params.append(limit)

    rows = _fetch_rows(db.conn, sql, params)
    samples: List[TrainingSample] = []
    for row in rows:
        embedding: Optional[List[float]] = None
        if include_embeddings and hasattr(db, "vector"):
            try:
                embedding = db.vector(row["id"])
            except Exception:  # pragma: no cover - best effort
                embedding = None
        samples.append(
            TrainingSample(
                source="discrepancy",
                content=row["message"],
                confidence=row["confidence"],
                outcome_score=row["outcome_score"],
                timestamp=row["ts"],
                embedding=embedding,
            )
        )
    return samples


def fetch_workflows(
    *,
    sort_by: str = "timestamp",
    limit: int = 100,
    include_embeddings: bool = False,
    scope: Scope = Scope.ALL,
) -> List[TrainingSample]:
    """Return stored workflows from :class:`task_handoff_bot.WorkflowDB`.

    Args:
        sort_by: Column used for ordering the results.
        limit: Maximum number of rows to return.
        include_embeddings: When ``True``, attach vector embeddings where
            available.
        scope: Menace scope for the query. Defaults to :attr:`Scope.ALL`.
    """

    db = TaskWorkflowDB()
    menace_id = getattr(getattr(db, "router", None), "menace_id", "")
    clause, params = build_scope_clause("workflows", scope, menace_id)
    columns = {"timestamp": "timestamp"}
    order_col = _resolve_order(sort_by, columns, "id")
    sql = "SELECT id, workflow, timestamp FROM workflows"
    if clause:
        sql += f" WHERE {clause}"
    sql += f" ORDER BY {order_col} DESC LIMIT ?"
    params.append(limit)

    rows = _fetch_rows(db.conn, sql, params)
    samples: List[TrainingSample] = []
    for row in rows:
        embedding: Optional[List[float]] = None
        if include_embeddings and hasattr(db, "vector"):
            try:
                embedding = db.vector(row["id"])
            except Exception:  # pragma: no cover - best effort
                embedding = None
        samples.append(
            TrainingSample(
                source="workflow",
                content=row["workflow"],
                timestamp=row["timestamp"],
                embedding=embedding,
            )
        )
    return samples


def fetch_watchdog_anomalies(
    *,
    sort_by: str = "timestamp",
    limit: int = 100,
    include_embeddings: bool = False,
    scope: Scope = Scope.ALL,
) -> List[TrainingSample]:
    """Return anomaly samples exported by :mod:`stripe_watchdog`.

    The ``scope`` parameter is accepted for API compatibility but currently
    ignored as the export file is not scoped.
    """

    if not TRAINING_ANOMALY_FILE.exists():
        return []

    rows: List[TrainingSample] = []
    try:
        with TRAINING_ANOMALY_FILE.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:  # pragma: no cover - best effort
                    continue
                ts = data.get("timestamp")
                if isinstance(ts, (int, float)):
                    ts_str = datetime.utcfromtimestamp(ts).isoformat()
                else:
                    ts_str = str(ts) if ts is not None else None
                rows.append(
                    TrainingSample(
                        source=str(data.get("source", "stripe_watchdog")),
                        content=data.get("content", ""),
                        timestamp=ts_str,
                    )
                )
    except Exception:  # pragma: no cover - best effort
        logger.warning("failed to load stripe anomaly training data")

    key_func = {
        "timestamp": lambda s: s.timestamp or "",
    }.get(sort_by, lambda s: s.timestamp or "")
    return sorted(rows, key=key_func, reverse=True)[:limit]


def aggregate_samples(
    sort_by: str = "timestamp",
    limit: int = 100,
    include_embeddings: bool = False,
    scope: Scope = Scope.ALL,
) -> List[TrainingSample]:
    """Return combined samples from all data sources.

    Args:
        sort_by: Column used for ordering the combined results.
        limit: Maximum number of rows to return.
        include_embeddings: When ``True``, attach vector embeddings where
            available.
        scope: Menace scope forwarded to each fetcher. Defaults to
            :attr:`Scope.ALL`.
    """

    fetchers = [
        ("fetch_enhancements", fetch_enhancements),
        ("fetch_summaries", fetch_summaries),
        ("fetch_discrepancies", fetch_discrepancies),
        ("fetch_workflows", fetch_workflows),
        ("fetch_watchdog_anomalies", fetch_watchdog_anomalies),
    ]
    samples: List[TrainingSample] = []
    for name, fetch in fetchers:
        try:
            samples.extend(
                fetch(
                    sort_by=sort_by,
                    limit=limit,
                    include_embeddings=include_embeddings,
                    scope=scope,
                )
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("%s failed: %s", name, exc)
            continue

    key_map = {
        "confidence": lambda s: (
            s.confidence if s.confidence is not None else float("-inf")
        ),
        "outcome_score": lambda s: (
            s.outcome_score if s.outcome_score is not None else float("-inf")
        ),
        "timestamp": lambda s: s.timestamp or "",
    }
    key_func = key_map.get(sort_by, key_map["timestamp"])
    return sorted(samples, key=key_func, reverse=True)[:limit]


# Backwards compatible aliases -------------------------------------------------
fetch_workflow_summaries = fetch_summaries
fetch_workflow_history = fetch_workflows
aggregate_examples = aggregate_samples

__all__ = [
    "TrainingSample",
    "fetch_enhancements",
    "fetch_summaries",
    "fetch_discrepancies",
    "fetch_workflows",
    "fetch_watchdog_anomalies",
    "aggregate_samples",
    "aggregate_examples",
]
