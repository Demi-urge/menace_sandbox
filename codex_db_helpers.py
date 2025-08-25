from __future__ import annotations

"""Helper utilities for pulling training samples from Menace databases."""

from typing import Any, Dict, Iterable, List, Sequence

from scope_utils import apply_scope_to_query


SORT_MAP = {
    "score": "score",
    "roi": "roi",
    "confidence": "confidence",
    "ts": "ts",
}


def _execute_query(db: Any, sql: str, params: Sequence[Any]) -> List[Dict[str, Any]]:
    """Execute ``sql`` on ``db`` and return rows as dictionaries."""

    conn = getattr(db, "conn", db)
    conn.row_factory = getattr(conn, "row_factory", None) or __import__(
        "sqlite3"
    ).Row
    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    return [dict(row) for row in rows]


def _attach_embeddings(db: Any, rows: Iterable[Dict[str, Any]]) -> None:
    if not hasattr(db, "vector"):
        return
    for row in rows:
        row["embedding"] = db.vector(row["id"])


def fetch_enhancements(
    db: Any,
    *,
    sort_by: str = "score",
    limit: int = 100,
    with_embeddings: bool = False,
) -> List[Dict[str, Any]]:
    """Return enhancement records sorted and limited as requested."""

    order_col = SORT_MAP.get(sort_by, "ts")
    base = (
        "SELECT id, summary, score, roi, confidence, ts FROM enhancements"
    )
    sql, params = apply_scope_to_query(base, scope="all", menace_id="")
    sql += f" ORDER BY {order_col} DESC LIMIT ?"
    params.append(limit)
    rows = _execute_query(db, sql, params)
    if with_embeddings:
        _attach_embeddings(db, rows)
    return rows


def fetch_summaries(
    db: Any,
    *,
    sort_by: str = "score",
    limit: int = 100,
    with_embeddings: bool = False,
) -> List[Dict[str, Any]]:
    """Return summary records sorted and limited as requested."""

    order_col = SORT_MAP.get(sort_by, "ts")
    base = (
        "SELECT id, summary, score, roi, confidence, ts FROM workflow_summaries"
    )
    sql, params = apply_scope_to_query(base, scope="all", menace_id="")
    sql += f" ORDER BY {order_col} DESC LIMIT ?"
    params.append(limit)
    rows = _execute_query(db, sql, params)
    if with_embeddings:
        _attach_embeddings(db, rows)
    return rows


def fetch_discrepancies(
    db: Any,
    *,
    sort_by: str = "score",
    limit: int = 100,
    with_embeddings: bool = False,
) -> List[Dict[str, Any]]:
    """Return discrepancy records sorted and limited as requested."""

    order_col = SORT_MAP.get(sort_by, "ts")
    base = (
        "SELECT id, message, score, roi, confidence, ts FROM discrepancies"
    )
    sql, params = apply_scope_to_query(base, scope="all", menace_id="")
    sql += f" ORDER BY {order_col} DESC LIMIT ?"
    params.append(limit)
    rows = _execute_query(db, sql, params)
    if with_embeddings:
        _attach_embeddings(db, rows)
    return rows


def fetch_workflow_history(
    db: Any,
    *,
    sort_by: str = "score",
    limit: int = 100,
    with_embeddings: bool = False,
) -> List[Dict[str, Any]]:
    """Return workflow history records sorted and limited as requested."""

    order_col = SORT_MAP.get(sort_by, "ts")
    base = (
        "SELECT id, details, score, roi, confidence, ts FROM workflow_history"
    )
    sql, params = apply_scope_to_query(base, scope="all", menace_id="")
    sql += f" ORDER BY {order_col} DESC LIMIT ?"
    params.append(limit)
    rows = _execute_query(db, sql, params)
    if with_embeddings:
        _attach_embeddings(db, rows)
    return rows


def aggregate_training_samples(
    enhancement_db: Any | None = None,
    summary_db: Any | None = None,
    discrepancy_db: Any | None = None,
    workflow_db: Any | None = None,
    *,
    sort_by: str = "score",
    limit: int = 100,
    with_embeddings: bool = False,
) -> List[Dict[str, Any]]:
    """Combine samples from all helpers and return the top results."""

    records: List[Dict[str, Any]] = []
    if enhancement_db is not None:
        records.extend(
            fetch_enhancements(
                enhancement_db,
                sort_by=sort_by,
                limit=limit,
                with_embeddings=with_embeddings,
            )
        )
    if summary_db is not None:
        records.extend(
            fetch_summaries(
                summary_db,
                sort_by=sort_by,
                limit=limit,
                with_embeddings=with_embeddings,
            )
        )
    if discrepancy_db is not None:
        records.extend(
            fetch_discrepancies(
                discrepancy_db,
                sort_by=sort_by,
                limit=limit,
                with_embeddings=with_embeddings,
            )
        )
    if workflow_db is not None:
        records.extend(
            fetch_workflow_history(
                workflow_db,
                sort_by=sort_by,
                limit=limit,
                with_embeddings=with_embeddings,
            )
        )
    order_col = SORT_MAP.get(sort_by, "ts")
    records.sort(key=lambda r: r.get(order_col) or 0, reverse=True)
    return records[:limit]


__all__ = [
    "fetch_enhancements",
    "fetch_summaries",
    "fetch_discrepancies",
    "fetch_workflow_history",
    "aggregate_training_samples",
]
