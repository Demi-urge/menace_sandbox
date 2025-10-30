from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import closing
from datetime import datetime
from typing import Any, Dict, List

from dynamic_path_router import resolve_path
from db_router import DBRouter, LOCAL_TABLES
from llm_interface import Completion, Prompt

# ---------------------------------------------------------------------------
# Module level helpers
# ---------------------------------------------------------------------------

# Ensure the prompts table is treated as local by DBRouter
LOCAL_TABLES.add("prompts")

_DB_ENV = os.environ.get("PROMPT_DB_PATH", "prompts.db")
try:
    DB_PATH = resolve_path(_DB_ENV)
except FileNotFoundError:
    DB_PATH = resolve_path(".") / _DB_ENV
_SCHEMA_INITIALISED = False
_SCHEMA_LOCK = threading.Lock()


def _ensure_schema() -> None:
    """Initialise the prompts schema once per process."""

    global _SCHEMA_INITIALISED
    if _SCHEMA_INITIALISED:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_INITIALISED:
            return
        with sqlite3.connect(DB_PATH) as conn:  # noqa: SQL001
            _init_db(conn)
        _SCHEMA_INITIALISED = True


def _init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prompts(
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            text TEXT,
            completion_raw TEXT,
            completion_parsed TEXT,
            response_text TEXT,
            response_parsed TEXT,
            examples TEXT,
            vector_confidence REAL,
            vector_confidences TEXT,
            outcome_tags TEXT,
            model TEXT,
            timestamp TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            latency_ms REAL,
            backend TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cost REAL
        )
        """,
    )
    for stmt in (
        "ALTER TABLE prompts ADD COLUMN prompt_tokens INTEGER",
        "ALTER TABLE prompts ADD COLUMN completion_tokens INTEGER",
        "ALTER TABLE prompts ADD COLUMN latency_ms REAL",
        "ALTER TABLE prompts ADD COLUMN backend TEXT",
        "ALTER TABLE prompts ADD COLUMN input_tokens INTEGER",
        "ALTER TABLE prompts ADD COLUMN output_tokens INTEGER",
        "ALTER TABLE prompts ADD COLUMN cost REAL",
    ):
        try:
            cur.execute(stmt)
        except sqlite3.OperationalError:
            pass
    conn.commit()


# ---------------------------------------------------------------------------
# Public functional API
# ---------------------------------------------------------------------------


def log_interaction(
    prompt_obj: Prompt,
    raw: Dict[str, Any],
    text: str,
    tags: List[str] | None,
    backend: str | None = None,
) -> None:
    """Record a single prompt/completion pair in the SQLite log."""

    _ensure_schema()
    with sqlite3.connect(DB_PATH) as conn:  # noqa: SQL001
        with closing(conn.cursor()) as cur:
            try:
                raw_json = json.dumps(raw)
            except Exception:  # pragma: no cover - defensive
                raw_json = json.dumps(None)
            parsed_obj: Any = None
            try:
                parsed_obj = json.loads(text)
            except Exception:
                pass
            try:
                parsed_json = json.dumps(parsed_obj)
            except Exception:  # pragma: no cover - defensive
                parsed_json = json.dumps(None)

            usage = (raw.get("usage") if isinstance(raw, dict) else {}) or {}
            cur.execute(
                """
                INSERT INTO prompts(
                    prompt, text, completion_raw, completion_parsed, response_text, response_parsed,
                    examples, vector_confidence, vector_confidences, outcome_tags, model, timestamp,
                    prompt_tokens, completion_tokens, latency_ms, backend, input_tokens, output_tokens, cost
                )
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    prompt_obj.user,
                    prompt_obj.user,
                    raw_json,
                    parsed_json,
                    text,
                    parsed_json,
                    json.dumps(getattr(prompt_obj, "examples", [])),
                    getattr(prompt_obj, "vector_confidence", None),
                    json.dumps(getattr(prompt_obj, "vector_confidences", [])),
                    json.dumps(list(tags or [])),
                    raw.get("model") or getattr(prompt_obj, "model", None),
                    datetime.utcnow().isoformat(),
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                    raw.get("latency_ms") if isinstance(raw, dict) else None,
                    backend,
                    usage.get("input_tokens") or usage.get("prompt_tokens"),
                    usage.get("output_tokens") or usage.get("completion_tokens"),
                    usage.get("cost"),
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

    _ensure_schema()
    query = (
        "SELECT prompt, completion_parsed, examples, vector_confidence, "
        "outcome_tags, model, backend, timestamp, completion_raw, "
        "prompt_tokens, completion_tokens, latency_ms, input_tokens, "
        "output_tokens, cost FROM prompts"
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

    with sqlite3.connect(DB_PATH) as conn:  # noqa: SQL001
        with closing(conn.cursor()) as cur:
            rows = cur.execute(query, params).fetchall()
    results: List[Dict[str, Any]] = []
    for row in rows:
        (
            prompt,
            completion,
            examples,
            vc,
            tags_json,
            mdl,
            backend,
            ts,
            raw_json,
            p_tokens,
            c_tokens,
            latency,
            i_tokens,
            o_tokens,
            cost,
        ) = row
        results.append(
            {
                "prompt": prompt,
                "completion": completion,
                "examples": json.loads(examples) if examples else [],
                "vector_confidence": vc,
                "outcome_tags": json.loads(tags_json) if tags_json else [],
                "model": mdl,
                "backend": backend,
                "timestamp": ts,
                "raw": json.loads(raw_json) if raw_json else None,
                "prompt_tokens": p_tokens,
                "completion_tokens": c_tokens,
                "latency_ms": latency,
                "input_tokens": i_tokens,
                "output_tokens": o_tokens,
                "cost": cost,
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
        db_path = resolve_path(path or os.environ.get("PROMPT_DB_PATH", "prompts.db"))
        self.router = router or DBRouter("prompts", str(db_path), str(db_path))
        self.conn = self.router.get_connection("prompts", operation="write")
        # Ensure schema exists for this connection as well
        _init_db(self.conn)

    # ------------------------------------------------------------------
    def log(self, prompt: Prompt, result: Completion, backend: str | None = None) -> None:
        """Persist *prompt* and *result* to the underlying SQLite store."""

        raw = result.raw or {}
        if backend is None:
            backend = raw.get("backend") or getattr(prompt, "metadata", {}).get("backend")
        try:
            raw_json = json.dumps(raw)
        except Exception:  # pragma: no cover - defensive
            raw_json = json.dumps(None)
        try:
            parsed_json = json.dumps(result.parsed)
        except Exception:  # pragma: no cover - defensive
            parsed_json = json.dumps(None)
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO prompts(
                prompt, text, completion_raw, completion_parsed, response_text, response_parsed,
                examples, vector_confidence, vector_confidences, outcome_tags, model, timestamp,
                prompt_tokens, completion_tokens, latency_ms, backend, input_tokens,
                output_tokens, cost
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                prompt.user,
                prompt.user,
                raw_json,
                parsed_json,
                result.text,
                parsed_json,
                json.dumps(getattr(prompt, "examples", [])),
                getattr(prompt, "vector_confidence", None),
                json.dumps(getattr(prompt, "vector_confidences", [])),
                json.dumps(list(prompt.outcome_tags)),
                raw.get("model") or getattr(prompt, "model", None) or self.model,
                datetime.utcnow().isoformat(),
                result.prompt_tokens,
                result.completion_tokens,
                result.latency_ms,
                backend,
                result.input_tokens or result.prompt_tokens,
                result.output_tokens or result.completion_tokens,
                result.cost,
            ),
        )
        self.conn.commit()

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
