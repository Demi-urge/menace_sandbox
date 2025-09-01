from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from db_router import DBRouter, LOCAL_TABLES
from llm_interface import Completion, Prompt

# ---------------------------------------------------------------------------
# Module level helpers
# ---------------------------------------------------------------------------

# Ensure the prompts table is treated as local by DBRouter
LOCAL_TABLES.add("prompts")


DB_PATH = Path(os.getenv("PROMPT_DB_PATH", "prompts.db"))
_CONN: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    """Return a cached SQLite connection and initialise the schema."""

    global _CONN
    if _CONN is None:
        _CONN = sqlite3.connect(DB_PATH)
        _init_db(_CONN)
    return _CONN


def _init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prompts(
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            completion_raw TEXT,
            completion_parsed TEXT,
            examples TEXT,
            vector_confidence REAL,
            outcome_tags TEXT,
            model TEXT,
            timestamp TEXT
        )
        """,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Public functional API
# ---------------------------------------------------------------------------


def log_interaction(
    prompt_obj: Prompt, raw: Dict[str, Any], parsed: Any, tags: List[str] | None
) -> None:
    """Record a single prompt/completion pair in the SQLite log."""

    conn = _get_conn()
    try:
        raw_json = json.dumps(raw)
    except Exception:  # pragma: no cover - defensive
        raw_json = json.dumps(None)
    try:
        parsed_json = json.dumps(parsed)
    except Exception:  # pragma: no cover - defensive
        parsed_json = json.dumps(None)

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO prompts(
            prompt, completion_raw, completion_parsed, examples,
            vector_confidence, outcome_tags, model, timestamp
        )
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            prompt_obj.user,
            raw_json,
            parsed_json,
            json.dumps(getattr(prompt_obj, "examples", [])),
            getattr(prompt_obj, "vector_confidence", None),
            json.dumps(list(tags or [])),
            raw.get("model") or getattr(prompt_obj, "model", None),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()


def fetch_logs(
    limit: int | None = None,
    *,
    model: str | None = None,
    tag: str | None = None,
) -> List[Dict[str, Any]]:
    """Return log entries optionally filtered by *model* or *tag*.

    Parameters
    ----------
    limit:
        Maximum number of rows to return. ``None`` returns all rows.
    model:
        If provided only entries for the given model are returned.
    tag:
        If provided only entries containing the JSON encoded tag are
        returned.
    """

    conn = _get_conn()
    query = (
        "SELECT prompt, completion_parsed, examples, vector_confidence, "
        "outcome_tags, model, timestamp, completion_raw FROM prompts"
    )
    clauses: List[str] = []
    params: List[Any] = []
    if model:
        clauses.append("model = ?")
        params.append(model)
    if tag:
        clauses.append("outcome_tags LIKE ?")
        params.append(f'%"{tag}"%')
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(query, params).fetchall()
    results: List[Dict[str, Any]] = []
    for row in rows:
        prompt, completion, examples, vc, tags_json, mdl, ts, raw_json = row
        results.append(
            {
                "prompt": prompt,
                "completion": completion,
                "examples": json.loads(examples) if examples else [],
                "vector_confidence": vc,
                "outcome_tags": json.loads(tags_json) if tags_json else [],
                "model": mdl,
                "timestamp": ts,
                "raw": json.loads(raw_json) if raw_json else None,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Backwards compatible class wrapper
# ---------------------------------------------------------------------------


class PromptDB:
    """Compatibility wrapper mimicking the old class based API."""

    def __init__(
        self, model: str, path: str | None = None, router: DBRouter | None = None
    ) -> None:
        self.model = model
        db_path = Path(path or os.getenv("PROMPT_DB_PATH", "prompts.db"))
        self.router = router or DBRouter("prompts", str(db_path), str(db_path))
        self.conn = self.router.get_connection("prompts", operation="write")
        # Ensure schema exists for this connection as well
        _init_db(self.conn)

    # ------------------------------------------------------------------
    def log(self, prompt: Prompt, result: Completion) -> None:
        """Persist *prompt* and *result* to the underlying SQLite store."""

        log_interaction(prompt, result.raw, result.parsed, prompt.outcome_tags)

    # ------------------------------------------------------------------
    def log_prompt(
        self,
        prompt: Prompt,
        result: Completion,
        outcome_tags: List[str] | None = None,
        vector_confidences: List[float] | None = None,
    ) -> None:
        """Backward compatible wrapper around :meth:`log`."""

        if outcome_tags is not None:
            prompt.outcome_tags = list(outcome_tags)
        if vector_confidences is not None:
            prompt.vector_confidence = vector_confidences[0] if vector_confidences else None
        self.log(prompt, result)


__all__ = ["log_interaction", "fetch_logs", "PromptDB"]

