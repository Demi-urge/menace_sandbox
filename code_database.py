# flake8: noqa
"""Database for storing code snippets and relationships.

Embeddings are refreshed on database events via
``EmbeddingBackfill.watch_events``. To backfill manually run::

    menace embed --db code
"""

from __future__ import annotations

import os
import sqlite3
import logging
import threading
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Callable, Iterator, TypeVar, Sequence, Tuple, Literal, Mapping
import json
from collections import deque, Counter
import re
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
import os
import sys

from pydantic.dataclasses import dataclass as pydantic_dataclass
from dataclasses import asdict

import license_detector
try:  # pragma: no cover - allow running without vector_service
    from vector_service import EmbeddableDBMixin, EmbeddingBackfill
except Exception:  # pragma: no cover - lightweight stub for tests
    class EmbeddingBackfill:  # type: ignore
        def watch_events(self, *a, **k) -> None:
            return None

    class EmbeddableDBMixin:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def add_embedding(self, *a, **k):  # pragma: no cover - simple stub
            pass

        def try_add_embedding(self, *a, **k):  # pragma: no cover - simple stub
            pass

        def encode_text(self, text):  # pragma: no cover - simple stub
            return [0.0]
from db_router import DBRouter, GLOBAL_ROUTER, init_db_router
from .scope_utils import Scope, build_scope_clause, apply_scope
from dynamic_path_router import resolve_path

try:  # pragma: no cover - support both package and flat imports
    from embeddable_db_mixin import log_embedding_metrics  # type: ignore
except Exception:  # pragma: no cover - fallback for package context
    try:
        from .embeddable_db_mixin import log_embedding_metrics  # type: ignore
    except Exception:  # pragma: no cover - ultimate fallback
        def log_embedding_metrics(*_a, **_k):  # type: ignore
            return None

try:  # optional dependency for future scalability
    from sqlalchemy.engine import Engine  # type: ignore
except Exception:  # pragma: no cover - optional
    Engine = None  # type: ignore

T = TypeVar("T")

try:  # pragma: no cover - support direct module execution
    from .auto_link import auto_link  # type: ignore
    from .unified_event_bus import UnifiedEventBus  # type: ignore
    from .retry_utils import publish_with_retry, with_retry  # type: ignore
    from .alert_dispatcher import send_discord_alert, CONFIG as ALERT_CONFIG  # type: ignore
except Exception:  # pragma: no cover - fallback for top-level imports
    from auto_link import auto_link  # type: ignore
    from unified_event_bus import UnifiedEventBus  # type: ignore
    from retry_utils import publish_with_retry, with_retry  # type: ignore
    from alert_dispatcher import send_discord_alert, CONFIG as ALERT_CONFIG  # type: ignore

try:  # pragma: no cover - optional dependency
    from .vector_metrics_db import VectorMetricsDB
except Exception:  # pragma: no cover - fallback when module unavailable
    VectorMetricsDB = None  # type: ignore

from chunking import split_into_chunks
from analysis.semantic_diff_filter import find_semantic_risks
from vector_service.text_preprocessor import generalise, get_config, PreprocessingConfig
try:  # pragma: no cover - support both package and flat imports
    from .menace_memory_manager import _summarise_text  # type: ignore
except Exception:  # pragma: no cover - fallback for top-level imports
    from menace_memory_manager import _summarise_text  # type: ignore

logger = logging.getLogger(__name__)

_WATCH_THREAD: threading.Thread | None = None


def _ensure_backfill_watcher(bus: "UnifiedEventBus" | None) -> None:
    """Start ``EmbeddingBackfill.watch_events`` once for this module."""

    global _WATCH_THREAD
    if bus is None or _WATCH_THREAD is not None:
        return
    try:
        thread = threading.Thread(
            target=EmbeddingBackfill().watch_events,
            kwargs={"bus": bus},
            daemon=True,
        )
        thread.start()
        _WATCH_THREAD = thread
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed to start embedding watcher")


_ensure_backfill_watcher(UnifiedEventBus())

SQL_DIR = resolve_path("sql_templates")


def _load_sql(name: str) -> str:
    """Load a SQL template with graceful fallback."""
    path = SQL_DIR / name
    try:
        with path.open() as fh:
            data = fh.read().strip()
            if data:
                return data
            raise ValueError("template empty")
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("failed to load SQL template %s: %s", name, exc)
        alt = Path(name)
        if alt.is_file():
            try:
                with alt.open() as fh:
                    data = fh.read().strip()
                    if data:
                        return data
            except Exception as exc2:  # pragma: no cover - unexpected
                logger.warning("fallback SQL load failed for %s: %s", name, exc2)
        logger.error("using empty SQL for template %s", name)
        return ""


# ---------------------------------------------------------------------------
# SQL statements used by ``CodeDB``
SQL_CREATE_CODE_TABLE = _load_sql("create_code_table.sql")

SQL_CREATE_CODE_BOTS_TABLE = _load_sql("create_code_bots_table.sql")
SQL_CREATE_CODE_ENHANCEMENTS_TABLE = _load_sql("create_code_enhancements_table.sql")
SQL_CREATE_CODE_ERRORS_TABLE = _load_sql("create_code_errors_table.sql")

SQL_CREATE_INDEX_CODE_BOTS_CODE = _load_sql("create_index_code_bots_code.sql")
SQL_CREATE_INDEX_CODE_BOTS_BOT = _load_sql("create_index_code_bots_bot.sql")
SQL_CREATE_INDEX_CODE_ENHANCEMENTS_CODE = _load_sql(
    "create_index_code_enhancements_code.sql"
)
SQL_CREATE_INDEX_CODE_ENHANCEMENTS_ENH = _load_sql(
    "create_index_code_enhancements_enh.sql"
)
SQL_CREATE_INDEX_CODE_ERRORS_CODE = _load_sql("create_index_code_errors_code.sql")
SQL_CREATE_INDEX_CODE_ERRORS_ERROR = _load_sql("create_index_code_errors_error.sql")
SQL_CREATE_INDEX_CODE_SUMMARY = _load_sql("create_index_code_summary.sql")
SQL_CREATE_INDEX_CODE_BODY = _load_sql("create_index_code_body.sql")
SQL_CREATE_INDEX_CODE_SOURCE_MENACE = _load_sql(
    "create_index_code_source_menace.sql"
)

SQL_CREATE_FTS = _load_sql("create_fts.sql")
SQL_POPULATE_FTS = _load_sql("populate_fts.sql")

SQL_INSERT_CODE = _load_sql("insert_code.sql")
SQL_INSERT_FTS = _load_sql("insert_fts.sql")
SQL_SELECT_REVISION = _load_sql("select_revision.sql")
SQL_DELETE_FTS_ROW = _load_sql("delete_fts_row.sql")
SQL_DELETE_CODE = _load_sql("delete_code.sql")
SQL_SEARCH_FTS = _load_sql("search_fts.sql")
SQL_SEARCH_FALLBACK = _load_sql("search_fallback.sql")
SQL_SELECT_ALL = _load_sql("select_all.sql")
SQL_SELECT_CODE_BOTS = _load_sql("select_code_bots.sql")
SQL_INSERT_CODE_BOT = _load_sql("insert_code_bot.sql")
SQL_INSERT_CODE_ENHANCEMENT = _load_sql("insert_code_enhancement.sql")
SQL_INSERT_CODE_ERROR = _load_sql("insert_code_error.sql")
SQL_DELETE_REL = {
    "code_bots": _load_sql("delete_code_bots.sql"),
    "code_enhancements": _load_sql("delete_code_enhancements.sql"),
    "code_errors": _load_sql("delete_code_errors.sql"),
}
SQL_SELECT_BY_COMPLEXITY = _load_sql("select_by_complexity.sql")

SQL_CREATE_LICENSE_VIOLATIONS_TABLE = (
    "CREATE TABLE IF NOT EXISTS license_violations("\
    "id INTEGER PRIMARY KEY AUTOINCREMENT,"\
    "path TEXT,"\
    "license TEXT,"\
    "hash TEXT,"\
    "detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
)
SQL_INSERT_LICENSE_VIOLATION = (
    "INSERT INTO license_violations(path, license, hash) VALUES (?, ?, ?)"
)

MIGRATIONS: list[tuple[int, list[str]]] = [
    (
        1,
        [
            SQL_CREATE_CODE_TABLE,
            SQL_CREATE_CODE_BOTS_TABLE,
            SQL_CREATE_CODE_ENHANCEMENTS_TABLE,
            SQL_CREATE_CODE_ERRORS_TABLE,
            SQL_CREATE_INDEX_CODE_BOTS_CODE,
            SQL_CREATE_INDEX_CODE_BOTS_BOT,
            SQL_CREATE_INDEX_CODE_ENHANCEMENTS_CODE,
            SQL_CREATE_INDEX_CODE_ENHANCEMENTS_ENH,
            SQL_CREATE_INDEX_CODE_ERRORS_CODE,
            SQL_CREATE_INDEX_CODE_ERRORS_ERROR,
        ],
    ),
    (
        2,
        [
            "ALTER TABLE code ADD COLUMN revision INTEGER DEFAULT 0",
        ],
    ),
    (
        3,
        [
            SQL_CREATE_INDEX_CODE_SUMMARY,
            SQL_CREATE_INDEX_CODE_BODY,
        ],
    ),
    (
        4,
        [
            "DROP INDEX IF EXISTS idx_code_summary",
            "DROP INDEX IF EXISTS idx_code_body",
            SQL_CREATE_INDEX_CODE_SUMMARY,
            SQL_CREATE_INDEX_CODE_BODY,
        ],
    ),
    (
        5,
        [
            SQL_CREATE_LICENSE_VIOLATIONS_TABLE,
        ],
    ),
    (
        6,
        [
            "ALTER TABLE code ADD COLUMN source_menace_id TEXT NOT NULL DEFAULT ''",
            SQL_CREATE_INDEX_CODE_SOURCE_MENACE,
        ],
    ),
]


def _hash_code(text: str | bytes) -> str:
    """Return a SHA-256 hex digest for the given code text."""
    if isinstance(text, bytes):
        data = text
    else:
        data = text.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _default_db_path(env_var: str, filename: str) -> Path:
    """Return a Path from ``env_var`` falling back to ``MENACE_DATA_DIR``."""
    base = os.getenv("MENACE_DATA_DIR", "")
    default = Path(base) / filename if base else Path(filename)
    candidate = os.getenv(env_var, default)
    try:
        return Path(resolve_path(str(candidate)))
    except FileNotFoundError:
        return Path(candidate)


def _current_menace_id(router: DBRouter | None) -> str:
    return router.menace_id if router else os.getenv("MENACE_ID", "")


@pydantic_dataclass
class CodeRecord:
    """Representation of a code template."""

    code: str
    template_type: str = ""
    language: str = "python"
    version: str = "1.0"
    complexity_score: float = 0.0
    summary: str = ""
    cid: int = 0
    revision: int = 0

    def __post_init__(self) -> None:
        assert self.code, "code cannot be empty"


class CodeDB(EmbeddableDBMixin):
    """SQLite storage for code templates and relationships."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        event_bus: Optional[UnifiedEventBus] = None,
        engine: "Engine" | None = None,
        router: DBRouter | None = None,
    ) -> None:
        """Create a new ``CodeDB`` instance.

        Parameters
        ----------
        path:
            Optional path to the SQLite database. Uses ``CODE_DB_PATH`` env var
            or defaults to ``code.db``.
        event_bus:
            Bus used to emit events for create/update/delete actions.
        engine:
            Optional SQLAlchemy engine to use instead of sqlite3.
        router:
            ``DBRouter`` providing connections.  When ``None`` the
            global ``GLOBAL_ROUTER`` is used.
        """
        self.engine = engine
        self.router = router or GLOBAL_ROUTER
        if self.engine is None and self.router is None:
            raise ValueError("router must be provided when engine is None")

        if self.engine is None:
            if path is None:
                self.path = _default_db_path("CODE_DB_PATH", "code.db")
            else:
                try:
                    self.path = Path(resolve_path(str(path)))
                except FileNotFoundError:
                    self.path = Path(path)
        else:
            self.path = Path("")
        self.event_bus = event_bus
        _ensure_backfill_watcher(self.event_bus)
        self._lock = threading.Lock()
        self.has_fts = False
        self.fts_retry_limit = 3
        self._fts_failures = 0
        self._fts_failure_times: deque[datetime] = deque(maxlen=10)
        self._fts_disabled_until: datetime | None = None

        self._allowed_update_fields = {
            "code",
            "template_type",
            "language",
            "version",
            "complexity",
            "complexity_score",
            "summary",
        }

        if self.engine is not None:
            self._init_engine()
        else:
            with self._connect() as conn:
                self._ensure_schema(conn)

        index_path = (
            self.path.with_suffix(".index")
            if self.path
            else resolve_path(".") / "code_embeddings.index"
        )
        meta_path = index_path.with_suffix(".json")
        EmbeddableDBMixin.__init__(
            self,
            index_path=index_path,
            metadata_path=meta_path,
            backend="annoy",
        )

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        with self._lock:
            if self.engine is not None:
                with self.engine.begin() as conn:
                    yield conn
            else:
                if self.router is None:
                    raise RuntimeError("DBRouter not configured")
                conn = self.router.get_connection("code")
                conn.execute("PRAGMA foreign_keys = ON")
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

    def _execute(self, conn: Any, sql: str, params: Iterable[Any] | None = None) -> Any:
        if hasattr(conn, "exec_driver_sql"):
            return conn.exec_driver_sql(sql, params or ())
        return conn.execute(sql, params or ())

    def _execute_fts(self, conn: Any, sql: str, params: Iterable[Any]) -> Any:
        result: Any = None

        def op() -> None:
            nonlocal result
            result = self._execute(conn, sql, params)

        try:
            prev = self._fts_failures
            with_retry(op, exc=sqlite3.Error, logger=logger)
            if prev:
                logger.info("fts operations recovered after %s failures", prev)
            self._fts_failures = 0
            self._fts_failure_times.clear()
        except Exception as exc:  # pragma: no cover - best effort
            self._fts_failures += 1
            now = datetime.utcnow()
            self._fts_failure_times.append(now)
            recent = [
                t for t in self._fts_failure_times if now - t < timedelta(minutes=30)
            ]
            backoff = min(60, 2 ** (len(recent) - 1))
            logger.warning(
                "fts operation failed (%s/%s): %s",
                self._fts_failures,
                self.fts_retry_limit,
                exc,
                exc_info=True,
                extra={"exc_class": exc.__class__.__name__, "sql": sql},
            )
            self._fts_disabled_until = now + timedelta(minutes=backoff)
            if self._fts_failures >= self.fts_retry_limit:
                self.has_fts = False
                logger.error(
                    "disabling FTS after repeated failures for %s minutes",
                    backoff,
                    extra={"exc_class": exc.__class__.__name__},
                )
                webhook = ALERT_CONFIG.get("discord_webhook")
                if webhook:
                    try:
                        send_discord_alert(
                            f"FTS disabled after {self._fts_failures} failures: {exc}",
                            webhook,
                        )
                    except Exception:
                        logger.exception("discord alert failed")
        return result

    def _insert(self, conn: Any, sql: str, params: Iterable[Any]) -> int:
        cur = self._execute(conn, sql, params)
        if hasattr(cur, "lastrowid"):
            return int(cur.lastrowid)
        if hasattr(cur, "inserted_primary_key"):
            return int(cur.inserted_primary_key[0])
        return 0

    def _maybe_init_fts(self, conn: Any) -> None:
        if self.has_fts:
            return
        if self._fts_disabled_until and datetime.utcnow() < self._fts_disabled_until:
            return
        try:
            self._execute(conn, SQL_CREATE_FTS)
            self._execute_fts(conn, SQL_POPULATE_FTS, ())
            self.has_fts = True
            self._fts_failures = 0
            self._fts_disabled_until = None
            logger.info("fts initialised/recovered")
        except Exception as exc:
            now = datetime.utcnow()
            self._fts_failure_times.append(now)
            recent = [
                t for t in self._fts_failure_times if now - t < timedelta(minutes=30)
            ]
            backoff = min(60, 2 ** (len(recent) - 1))
            logger.warning(
                "fts reinitialisation failed: %s",
                exc,
                exc_info=True,
                extra={"exc_class": exc.__class__.__name__},
            )
            self.has_fts = False
            self._fts_disabled_until = now + timedelta(minutes=backoff)

    def _ensure_schema(self, conn: Any) -> None:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        cols = [r[1] for r in conn.execute("PRAGMA table_info(code)").fetchall()]
        for target, stmts in MIGRATIONS:
            if version < target:
                for stmt in stmts:
                    if stmt.startswith("ALTER TABLE code"):
                        col = stmt.split()[5]
                        if col in cols:
                            continue
                    self._execute(conn, stmt)
                version = target
                conn.execute(f"PRAGMA user_version = {version}")
                cols = [
                    r[1] for r in conn.execute("PRAGMA table_info(code)").fetchall()
                ]

        try:
            self._maybe_init_fts(conn)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("fts initialisation failed: %s", exc)

    def _with_retry(self, func: Callable[[Any], T]) -> T:
        return with_retry(func, exc=sqlite3.Error, logger=logger)

    def _conn_wrapper(self, func: Callable[[Any], T]) -> T:
        with self._connect() as conn:
            return func(conn)

    def _init_engine(self) -> None:
        if self.engine is None:
            return
        with self.engine.begin() as conn:
            self._ensure_schema(conn)

    # ------------------------------------------------------------------
    @auto_link(
        {"bots": "link_bot", "enhancements": "link_enhancement", "errors": "link_error"}
    )
    def add(
        self,
        rec: CodeRecord,
        *,
        bots: Iterable[str] | None = None,
        enhancements: Iterable[int] | None = None,
        errors: Iterable[int] | None = None,
        embedding: Optional[List[float]] = None,
    ) -> int:
        """Insert ``rec`` into the database and link to related tables."""

        def op(conn: Any) -> int:
            if embedding is not None:
                if not isinstance(embedding, list) or not all(
                    isinstance(x, float) for x in embedding
                ):
                    raise ValueError("embedding must be a list of floats")
            if not rec.code or not rec.code.strip():
                raise ValueError("code cannot be empty")
            if not rec.summary or not rec.summary.strip():
                raise ValueError("summary cannot be empty")
            lic = license_detector.detect(rec.code)
            if lic:
                hash_ = license_detector.fingerprint(rec.code)
                try:
                    self.log_license_violation("", lic, hash_)
                except Exception:
                    pass
                log_embedding_metrics(
                    self.__class__.__name__, 0, 0.0, 0.0, vector_id=""
                )
                raise ValueError(f"disallowed license detected: {lic}")
            menace_id = self.router.menace_id
            rec.cid = self._insert(
                conn,
                SQL_INSERT_CODE,
                (
                    menace_id,
                    rec.code,
                    rec.template_type,
                    rec.language,
                    rec.version,
                    rec.complexity_score,
                    rec.summary,
                ),
            )
            if not self.has_fts:
                self._maybe_init_fts(conn)
            if self.has_fts:
                try:
                    self._execute_fts(
                        conn, SQL_INSERT_FTS, (rec.cid, rec.summary, rec.code)
                    )
                except Exception:
                    self.has_fts = False
                    self._fts_disabled_until = datetime.utcnow() + timedelta(minutes=5)
            return rec.cid

        cid = self._with_retry(lambda: self._conn_wrapper(op))
        self.try_add_embedding(cid, rec, "code", source_id=str(cid))
        if self.event_bus:
            if not publish_with_retry(self.event_bus, "code:new", asdict(rec)):
                logger.exception("failed to publish code:new event")
            else:
                publish_with_retry(
                    self.event_bus,
                    "db:record_added",
                    {"db": "code"},
                )
                publish_with_retry(
                    self.event_bus,
                    "db.record_changed",
                    {"db": "code"},
                )
                publish_with_retry(
                    self.event_bus,
                    "embedding:backfill",
                    {"db": self.__class__.__name__},
                )
        return cid

    def update(
        self,
        code_id: int,
        *,
        expected_revision: int | None = None,
        source_menace_id: str | None = None,
        **fields: Any,
    ) -> None:
        """Update fields of a code record.

        Parameters
        ----------
        code_id:
            ID of the record to update.
        expected_revision:
            If provided, update succeeds only when current revision matches.
        fields:
            Whitelisted columns to update.
        """
        if not fields:
            return

        def op(conn: Any) -> CodeRecord | None:
            menace_id = source_menace_id or _current_menace_id(self.router)
            if expected_revision is not None:
                clause, params = build_scope_clause("code", Scope.LOCAL, menace_id)
                query = apply_scope(SQL_SELECT_REVISION, clause)
                row = self._execute(conn, query, [code_id, *params]).fetchone()
                if not row or row[0] != expected_revision:
                    raise RuntimeError("revision mismatch")

            mapping = {
                "code": "code",
                "template_type": "template_type",
                "language": "language",
                "version": "version",
                "complexity": "complexity",
                "complexity_score": "complexity",
                "summary": "summary",
            }
            invalid = set(fields) - self._allowed_update_fields
            if invalid:
                allowed = ", ".join(sorted(self._allowed_update_fields))
                msg = f"Fields not permitted for update: {', '.join(sorted(invalid))}. Allowed fields: {allowed}"
                raise ValueError(msg)
            columns: list[str] = []
            params: list[Any] = []
            for key, value in fields.items():
                col = mapping[key]
                if col in {"code", "summary"} and (
                    value is None or not str(value).strip()
                ):
                    raise ValueError(f"{col} cannot be empty")
                columns.append(f"{col}=?")
                params.append(value)
            sets = ", ".join(columns)
            if expected_revision is not None:
                sets += ", revision=revision+1"
                params.extend([code_id, expected_revision, menace_id])
                self._execute(
                    conn,
                    f"UPDATE code SET {sets} WHERE id=? AND revision=? AND source_menace_id=?",
                    params,
                )
            else:
                params.extend([code_id, menace_id])
                self._execute(
                    conn,
                    f"UPDATE code SET {sets} WHERE id=? AND source_menace_id=?",
                    params,
                )
            if not self.has_fts:
                self._maybe_init_fts(conn)
            rec_row = None
            if self.has_fts:
                try:
                    rec_row = self._execute(
                        conn,
                        "SELECT code, template_type, language, version, complexity_score, summary, revision FROM code WHERE id=? AND source_menace_id=?",
                        (code_id, menace_id),
                    ).fetchone()
                    if rec_row:
                        self._execute_fts(conn, SQL_DELETE_FTS_ROW, (code_id,))
                        self._execute_fts(
                            conn,
                            SQL_INSERT_FTS,
                            (code_id, rec_row[5], rec_row[0]),
                        )
                except Exception as exc:
                    self.has_fts = False
                    self._fts_disabled_until = datetime.utcnow() + timedelta(minutes=5)
                    logger.warning("fts update failed: %s", exc, exc_info=True)
            if rec_row and isinstance(rec_row, sqlite3.Row):
                return CodeRecord(
                    code=rec_row[0],
                    template_type=rec_row[1],
                    language=rec_row[2],
                    version=rec_row[3],
                    complexity_score=rec_row[4],
                    summary=rec_row[5],
                    cid=code_id,
                    revision=rec_row[6],
                )
            return None

        rec = self._with_retry(lambda: self._conn_wrapper(op))
        if rec:
            self.try_add_embedding(code_id, rec, "code", source_id=str(code_id))
        if self.event_bus:
            payload = {"code_id": code_id, **fields}
            if not publish_with_retry(self.event_bus, "code:update", payload):
                logger.exception("failed to publish code:update event")
            else:
                publish_with_retry(
                    self.event_bus,
                    "db:record_updated",
                    {"db": "code"},
                )
                publish_with_retry(
                    self.event_bus,
                    "db.record_changed",
                    {"db": "code"},
                )
                publish_with_retry(
                    self.event_bus,
                    "embedding:backfill",
                    {"db": self.__class__.__name__},
                )

    def fetch_all(
        self,
        *,
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> List[Dict[str, Any]]:
        """Return all code records as a list of dictionaries."""

        menace_id = source_menace_id or _current_menace_id(self.router)

        def op(conn: Any) -> List[Dict[str, Any]]:
            if isinstance(conn, sqlite3.Connection):
                conn.row_factory = sqlite3.Row
            clause, params = build_scope_clause("code", Scope(scope), menace_id)
            query = apply_scope("SELECT * FROM code", clause)
            rows = self._execute(conn, query, params).fetchall()
            return [dict(r) for r in rows]

        return self._with_retry(lambda: self._conn_wrapper(op))

    def by_complexity(
        self,
        min_score: float = 0.0,
        limit: int = 5,
        *,
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> List[Dict[str, Any]]:
        """Return code records sorted by complexity score."""

        menace_id = source_menace_id or _current_menace_id(self.router)

        def op(conn: Any) -> List[Dict[str, Any]]:
            if isinstance(conn, sqlite3.Connection):
                conn.row_factory = sqlite3.Row
            clause, scope_params = build_scope_clause("code", Scope(scope), menace_id)
            query = apply_scope(
                "SELECT * FROM code WHERE complexity >= ?", clause
            )
            params: list[Any] = [min_score, *scope_params, limit]
            query += " ORDER BY complexity DESC LIMIT ?"
            rows = self._execute(conn, query, params).fetchall()
            return [dict(r) for r in rows]

        return self._with_retry(lambda: self._conn_wrapper(op))

    def search(
        self,
        term: str,
        *,
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> List[Dict[str, Any]]:
        """Search code records by summary or code."""
        pattern = f"%{term}%"
        menace_id = source_menace_id or _current_menace_id(self.router)

        def fallback(conn: Any) -> List[Any]:
            clause, scope_params = build_scope_clause("code", Scope(scope), menace_id)
            query = apply_scope(SQL_SEARCH_FALLBACK, clause)
            params = [pattern, pattern, *scope_params]
            return self._execute(conn, query, params).fetchall()

        def op(conn: Any) -> List[Dict[str, Any]]:
            if isinstance(conn, sqlite3.Connection):
                conn.row_factory = sqlite3.Row
            if not self.has_fts:
                self._maybe_init_fts(conn)
            if self.has_fts:
                try:
                    clause, scope_params = build_scope_clause("c", Scope(scope), menace_id)
                    query = apply_scope(SQL_SEARCH_FTS, clause)
                    params = [f"{term}*", *scope_params]
                    rows = self._execute_fts(conn, query, params).fetchall()
                except Exception as exc:
                    self.has_fts = False
                    self._fts_disabled_until = datetime.utcnow() + timedelta(minutes=5)
                    logger.warning(
                        "fts search failed: %s",
                        exc,
                        extra={"exc_class": exc.__class__.__name__},
                        exc_info=True,
                    )
                    logger.info("falling back to non-fts search")
                    rows = fallback(conn)
            else:
                logger.info("fts disabled, using fallback search")
                rows = fallback(conn)
            return [dict(r) for r in rows]

        return self._with_retry(lambda: self._conn_wrapper(op))

    def search_fallback(
        self,
        term: str,
        *,
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> List[Dict[str, Any]]:
        """Search using the non-FTS fallback query directly."""
        pattern = f"%{term}%"
        menace_id = source_menace_id or _current_menace_id(self.router)

        def op(conn: Any) -> List[Dict[str, Any]]:
            if isinstance(conn, sqlite3.Connection):
                conn.row_factory = sqlite3.Row
            clause, scope_params = build_scope_clause("code", Scope(scope), menace_id)
            query = apply_scope(SQL_SEARCH_FALLBACK, clause)
            params = [pattern, pattern, *scope_params]
            rows = self._execute(conn, query, params).fetchall()
            return [dict(r) for r in rows]

        return self._with_retry(lambda: self._conn_wrapper(op))

    def codes_for_bot(
        self,
        bot_id: str,
        *,
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> List[int]:
        """Return IDs of code templates associated with a bot."""

        menace_id = source_menace_id or _current_menace_id(self.router)

        def op(conn: Any) -> List[int]:
            clause, params = build_scope_clause("c", Scope(scope), menace_id)
            query = apply_scope(SQL_SELECT_CODE_BOTS, clause)
            rows = self._execute(conn, query, [bot_id, *params]).fetchall()
            return [int(r[0]) for r in rows]

        return self._with_retry(lambda: self._conn_wrapper(op))

    # linking -----------------------------------------------------------
    def link_bot(
        self, code_id: int, bot_id: str, *, source_menace_id: str | None = None
    ) -> None:
        """Associate a code record with a bot."""

        menace_id = source_menace_id or self.router.menace_id

        def op(conn: Any) -> None:
            self._insert(conn, SQL_INSERT_CODE_BOT, (menace_id, code_id, bot_id))

        self._with_retry(lambda: self._conn_wrapper(op))

    def link_enhancement(
        self, code_id: int, enh_id: int, *, source_menace_id: str | None = None
    ) -> None:
        """Associate a code record with an enhancement."""

        menace_id = source_menace_id or self.router.menace_id

        def op(conn: Any) -> None:
            self._insert(
                conn, SQL_INSERT_CODE_ENHANCEMENT, (menace_id, code_id, enh_id)
            )

        self._with_retry(lambda: self._conn_wrapper(op))

    def link_error(
        self, code_id: int, err_id: int, *, source_menace_id: str | None = None
    ) -> None:
        """Associate a code record with an error."""

        menace_id = source_menace_id or self.router.menace_id

        def op(conn: Any) -> None:
            self._insert(conn, SQL_INSERT_CODE_ERROR, (menace_id, code_id, err_id))

        self._with_retry(lambda: self._conn_wrapper(op))

    def log_license_violation(self, path: str, license_name: str, hash: str) -> None:
        """Record a disallowed license fingerprint for auditing."""

        def op(conn: Any) -> None:
            self._insert(conn, SQL_INSERT_LICENSE_VIOLATION, (path, license_name, hash))

        self._with_retry(lambda: self._conn_wrapper(op))

    def delete(self, code_id: int, *, source_menace_id: str | None = None) -> None:
        """Safely remove a code record and its relationships."""

        menace_id = source_menace_id or _current_menace_id(self.router)

        def op(conn: Any) -> None:
            tables = ["code_bots", "code_enhancements", "code_errors"]
            for table in tables:
                self._execute(conn, SQL_DELETE_REL[table], (code_id,))
            clause, params = build_scope_clause("code", Scope.LOCAL, menace_id)
            query = apply_scope(SQL_DELETE_CODE, clause)
            self._execute(conn, query, [code_id, *params])
            if not self.has_fts:
                self._maybe_init_fts(conn)
            if self.has_fts:
                try:
                    self._execute_fts(conn, SQL_DELETE_FTS_ROW, (code_id,))
                except Exception as exc:
                    self.has_fts = False
                    self._fts_disabled_until = datetime.utcnow() + timedelta(minutes=5)
                    logger.warning("fts delete failed: %s", exc, exc_info=True)

        self._with_retry(lambda: self._conn_wrapper(op))
        if self.event_bus:
            publish_with_retry(self.event_bus, "code:delete", {"code_id": code_id})

    # embedding --------------------------------------------------------
    def _record_text(self, data: dict[str, Any]) -> str:
        """Build a textual representation of ``data`` including metadata."""
        parts: list[str] = []
        summary = str(data.get("summary", ""))
        if summary:
            parts.append(f"summary: {summary}")
        code = str(data.get("code", ""))
        if code:
            parts.append(code)
        template = str(data.get("template_type", ""))
        if template:
            parts.append(f"template_type: {template}")
        language = str(data.get("language", ""))
        if language:
            parts.append(f"language: {language}")
        return "\n".join(parts)

    def _embed_text(
        self,
        data: dict[str, Any],
        *,
        config: PreprocessingConfig | None = None,
    ) -> str:
        """Return a condensed, safe representation of code ``data``."""

        cfg = config or get_config("code")
        parts: list[str] = []
        summary = str(data.get("summary", "")).strip()
        if summary:
            parts.append(generalise(summary, config=cfg))
        code = str(data.get("code", ""))
        if code:
            try:
                chunks = split_into_chunks(code, cfg.chunk_size or 400)
            except Exception:  # pragma: no cover - fallback
                chunks = [type("C", (), {"text": code})()]
            summaries: list[str] = []
            for ch in chunks:
                if cfg.filter_semantic_risks and find_semantic_risks(ch.text.splitlines()):
                    continue
                try:
                    summ = _summarise_text(ch.text)
                except Exception:  # pragma: no cover - best effort
                    summ = ch.text
                summ = generalise(summ, config=cfg)
                if summ:
                    summaries.append(summ)
            if summaries:
                parts.append(" ".join(summaries))
        template = str(data.get("template_type", "")).strip()
        if template:
            parts.append(generalise(f"template_type: {template}", config=cfg))
        language = str(data.get("language", "")).strip()
        if language:
            parts.append(generalise(f"language: {language}", config=cfg))
        return "\n".join(parts).strip()

    def license_text(self, rec: CodeRecord | dict[str, Any]) -> str | None:
        if isinstance(rec, CodeRecord):
            return rec.code
        if isinstance(rec, dict):
            return rec.get("code")
        return None

    def add_embedding(
        self,
        record_id: Any,
        record: Any,
        kind: str,
        *,
        source_id: str = "",
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> None:
        """Embed ``record`` and store the vector and metadata."""
        if not isinstance(record, (CodeRecord, dict)):
            menace_id = source_menace_id or _current_menace_id(self.router)
            clause, params = build_scope_clause("code", Scope(scope), menace_id)
            query = apply_scope(
                "SELECT summary, code, template_type, language FROM code WHERE id= ?",
                clause,
            )
            with self._connect() as conn:
                row = conn.execute(query, [record_id, *params]).fetchone()
            if not row:
                return
            record = (
                dict(row)
                if isinstance(row, sqlite3.Row)
                else {
                    "summary": row[0],
                    "code": row[1],
                    "template_type": row[2],
                    "language": row[3],
                }
            )
        EmbeddableDBMixin.add_embedding(
            self, record_id, record, kind, source_id=source_id
        )

    def backfill_embeddings(self) -> None:
        """Delegate to :class:`EmbeddableDBMixin` for compatibility."""
        EmbeddableDBMixin.backfill_embeddings(self)

    def iter_records(
        self,
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> Iterator[tuple[int, dict[str, Any], str]]:
        """Yield code rows for embedding backfill."""
        menace_id = source_menace_id or _current_menace_id(self.router)
        clause, params = build_scope_clause("code", Scope(scope), menace_id)
        query = apply_scope(
            "SELECT id, summary, code, template_type, language FROM code",
            clause,
        )
        with self._connect() as conn:
            cur = conn.execute(query, params)
            rows = cur.fetchall()
        for row in rows:
            data = dict(row) if isinstance(row, sqlite3.Row) else {
                "id": row[0],
                "summary": row[1],
                "code": row[2],
                "template_type": row[3],
                "language": row[4],
            }
            yield data["id"], data, "code"

    def vector(
        self, rec: Any, *, source_menace_id: str | None = None
    ) -> List[float] | None:
        """Return an embedding vector for ``rec`` or record id."""
        if isinstance(rec, (int, str)):
            rid = str(rec)
            meta = getattr(self, "_metadata", {}).get(rid)
            if meta and "vector" in meta:
                return meta["vector"]
            try:
                rec_id = int(rec)
            except (TypeError, ValueError):
                return None
            menace_id = source_menace_id or _current_menace_id(self.router)
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT summary, code, template_type, language FROM code WHERE id=? AND source_menace_id=?",
                    (rec_id, menace_id),
                ).fetchone()
            if not row:
                return None
            data = dict(row) if isinstance(row, sqlite3.Row) else {
                "summary": row[0],
                "code": row[1],
                "template_type": row[2],
                "language": row[3],
            }
            text = self._embed_text(data)
            return self.encode_text(text) if text else None
        if isinstance(rec, CodeRecord):
            data = asdict(rec)
        elif isinstance(rec, dict):
            data = rec
        else:
            return None
        text = self._embed_text(data)
        return self.encode_text(text) if text else None

    def search_by_vector(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        *,
        source_menace_id: str | None = None,
    ) -> List[dict[str, Any]]:
        matches = EmbeddableDBMixin.search_by_vector(self, vector, top_k)
        results: List[dict[str, Any]] = []
        menace_id = source_menace_id or _current_menace_id(self.router)
        with self._connect() as conn:
            for cid, dist in matches:
                row = conn.execute(
                    "SELECT id, summary, code FROM code WHERE id=? AND source_menace_id=?",
                    (cid, menace_id),
                ).fetchone()
                if row:
                    data = dict(row) if isinstance(row, sqlite3.Row) else {
                        "id": row[0],
                        "summary": row[1],
                        "code": row[2],
                    }
                    data["_distance"] = dist
                    results.append(data)
        return results


@pydantic_dataclass
class PatchRecord:
    """History entry for a code patch."""

    filename: str
    description: str
    roi_before: float
    roi_after: float
    errors_before: int = 0
    errors_after: int = 0
    roi_delta: float = 0.0
    complexity_before: float = 0.0
    complexity_after: float = 0.0
    complexity_delta: float = 0.0
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    entropy_delta: float = 0.0
    predicted_roi: float = 0.0
    predicted_errors: float = 0.0
    reverted: bool = False
    trending_topic: str | None = None
    ts: str = datetime.utcnow().isoformat()
    code_id: int | None = None
    code_hash: str | None = None
    source_bot: str | None = None
    version: str | None = None
    parent_patch_id: int | None = None
    reason: str | None = None
    trigger: str | None = None
    diff: str | None = None
    summary: str | None = None
    outcome: str | None = None
    prompt_headers: str | None = None
    prompt_order: str | None = None
    prompt_tone: str | None = None

    def __post_init__(self) -> None:
        assert self.filename, "filename cannot be empty"


class PatchHistoryDB:
    """SQLite-backed store for patch history."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        code_db: CodeDB | None = None,
        router: DBRouter | None = None,
    ) -> None:
        """Initialise the patch history database."""
        self.path = Path(
            path or _default_db_path("PATCH_HISTORY_DB_PATH", "patch_history.db")
        )
        self.code_db = code_db
        self._lock = threading.Lock()
        self.keyword_counts: Counter[str] = Counter()
        self.keyword_recent: Dict[str, float] = {}
        self._vec_db = VectorMetricsDB() if VectorMetricsDB is not None else None
        # Always initialise a router for the provided path unless an explicit
        # router instance is supplied.  This ensures tests using temporary
        # databases do not share state via ``GLOBAL_ROUTER``.
        self.router = router or init_db_router(
            "patch_history", str(self.path), str(self.path)
        )
        conn = self.router.get_connection("patch_history")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patch_history(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                description TEXT,
                roi_before REAL,
                roi_after REAL,
                errors_before INTEGER DEFAULT 0,
                errors_after INTEGER DEFAULT 0,
                roi_delta REAL DEFAULT 0,
                complexity_before REAL DEFAULT 0,
                complexity_after REAL DEFAULT 0,
                complexity_delta REAL DEFAULT 0,
                entropy_before REAL DEFAULT 0,
                entropy_after REAL DEFAULT 0,
                entropy_delta REAL DEFAULT 0,
                predicted_roi REAL DEFAULT 0,
                predicted_errors REAL DEFAULT 0,
                reverted INTEGER DEFAULT 0,
                trending_topic TEXT,
                ts TEXT,
                source_bot TEXT,
                version TEXT,
                code_id INTEGER,
                code_hash TEXT,
                parent_patch_id INTEGER,
                reason TEXT,
                trigger TEXT,
                diff TEXT,
                summary TEXT,
                outcome TEXT,
                lines_changed INTEGER DEFAULT 0,
                tests_passed INTEGER DEFAULT 0,
                context_tokens INTEGER DEFAULT 0,
                patch_difficulty INTEGER DEFAULT 0,
                effort_estimate REAL,
                enhancement_name TEXT,
                start_time REAL,
                time_to_completion REAL,
                timestamp REAL,
                roi_deltas TEXT,
                errors TEXT,
                error_trace_count INTEGER DEFAULT 0,
                roi_tag TEXT,
                enhancement_score REAL,
                prompt_headers TEXT,
                prompt_order TEXT,
                prompt_tone TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS score_weights(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                w1 REAL,
                w2 REAL,
                w3 REAL,
                w4 REAL,
                w5 REAL,
                w6 REAL,
                ts TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS flakiness_history(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                flakiness REAL,
                ts TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS failed_strategies(
                code_hash TEXT PRIMARY KEY,
                ts TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patch_provenance(
                patch_id INTEGER,
                origin TEXT,
                vector_id TEXT,
                influence REAL,
                retrieved_at TEXT,
                position INTEGER,
                license TEXT,
                license_fingerprint TEXT,
                semantic_alerts TEXT,
                FOREIGN KEY(patch_id) REFERENCES patch_history(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patch_ancestry(
                patch_id INTEGER,
                origin TEXT,
                vector_id TEXT,
                influence REAL,
                license TEXT,
                license_fingerprint TEXT,
                semantic_alerts TEXT,
                FOREIGN KEY(patch_id) REFERENCES patch_history(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patch_contributors(
                patch_id INTEGER,
                vector_id TEXT,
                influence REAL,
                session_id TEXT,
                FOREIGN KEY(patch_id) REFERENCES patch_history(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS branch_log(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patch_id TEXT,
                branch TEXT,
                action TEXT,
                ts TEXT
            )
            """
        )
        cols = [r[1] for r in conn.execute("PRAGMA table_info(patch_history)").fetchall()]
        migrations = {
            "errors_before": "ALTER TABLE patch_history ADD COLUMN errors_before INTEGER DEFAULT 0",
            "errors_after": "ALTER TABLE patch_history ADD COLUMN errors_after INTEGER DEFAULT 0",
            "roi_delta": "ALTER TABLE patch_history ADD COLUMN roi_delta REAL DEFAULT 0",
            "complexity_before": "ALTER TABLE patch_history ADD COLUMN complexity_before REAL DEFAULT 0",
            "complexity_after": "ALTER TABLE patch_history ADD COLUMN complexity_after REAL DEFAULT 0",
            "complexity_delta": "ALTER TABLE patch_history ADD COLUMN complexity_delta REAL DEFAULT 0",
            "entropy_before": "ALTER TABLE patch_history ADD COLUMN entropy_before REAL DEFAULT 0",
            "entropy_after": "ALTER TABLE patch_history ADD COLUMN entropy_after REAL DEFAULT 0",
            "entropy_delta": "ALTER TABLE patch_history ADD COLUMN entropy_delta REAL DEFAULT 0",
            "predicted_roi": "ALTER TABLE patch_history ADD COLUMN predicted_roi REAL DEFAULT 0",
            "predicted_errors": "ALTER TABLE patch_history ADD COLUMN predicted_errors REAL DEFAULT 0",
            "reverted": "ALTER TABLE patch_history ADD COLUMN reverted INTEGER DEFAULT 0",
            "trending_topic": "ALTER TABLE patch_history ADD COLUMN trending_topic TEXT",
            "source_bot": "ALTER TABLE patch_history ADD COLUMN source_bot TEXT",
            "version": "ALTER TABLE patch_history ADD COLUMN version TEXT",
            "code_id": "ALTER TABLE patch_history ADD COLUMN code_id INTEGER",
            "code_hash": "ALTER TABLE patch_history ADD COLUMN code_hash TEXT",
            "parent_patch_id": "ALTER TABLE patch_history ADD COLUMN parent_patch_id INTEGER",
            "reason": "ALTER TABLE patch_history ADD COLUMN reason TEXT",
            "trigger": "ALTER TABLE patch_history ADD COLUMN trigger TEXT",
            "diff": "ALTER TABLE patch_history ADD COLUMN diff TEXT",
            "summary": "ALTER TABLE patch_history ADD COLUMN summary TEXT",
            "outcome": "ALTER TABLE patch_history ADD COLUMN outcome TEXT",
            "lines_changed": "ALTER TABLE patch_history ADD COLUMN lines_changed INTEGER DEFAULT 0",
            "tests_passed": "ALTER TABLE patch_history ADD COLUMN tests_passed INTEGER DEFAULT 0",
            "context_tokens": "ALTER TABLE patch_history ADD COLUMN context_tokens INTEGER DEFAULT 0",
            "patch_difficulty": "ALTER TABLE patch_history ADD COLUMN patch_difficulty INTEGER DEFAULT 0",
            "effort_estimate": "ALTER TABLE patch_history ADD COLUMN effort_estimate REAL",
            "enhancement_name": "ALTER TABLE patch_history ADD COLUMN enhancement_name TEXT",
            "start_time": "ALTER TABLE patch_history ADD COLUMN start_time REAL",
            "time_to_completion": "ALTER TABLE patch_history ADD COLUMN time_to_completion REAL",
            "timestamp": "ALTER TABLE patch_history ADD COLUMN timestamp REAL",
            "roi_deltas": "ALTER TABLE patch_history ADD COLUMN roi_deltas TEXT",
            "errors": "ALTER TABLE patch_history ADD COLUMN errors TEXT",
            "error_trace_count": "ALTER TABLE patch_history ADD COLUMN error_trace_count INTEGER DEFAULT 0",
            "roi_tag": "ALTER TABLE patch_history ADD COLUMN roi_tag TEXT",
            "enhancement_score": "ALTER TABLE patch_history ADD COLUMN enhancement_score REAL",
            "prompt_headers": "ALTER TABLE patch_history ADD COLUMN prompt_headers TEXT",
            "prompt_order": "ALTER TABLE patch_history ADD COLUMN prompt_order TEXT",
            "prompt_tone": "ALTER TABLE patch_history ADD COLUMN prompt_tone TEXT",
        }
        for name, stmt in migrations.items():
            if name not in cols:
                conn.execute(stmt)
        # Ensure patch_provenance and patch_ancestry have alert columns
        cols = [r[1] for r in conn.execute("PRAGMA table_info(patch_provenance)").fetchall()]
        if "license" not in cols:
            conn.execute("ALTER TABLE patch_provenance ADD COLUMN license TEXT")
        if "license_fingerprint" not in cols:
            conn.execute(
                "ALTER TABLE patch_provenance ADD COLUMN license_fingerprint TEXT"
            )
        if "semantic_alerts" not in cols:
            conn.execute(
                "ALTER TABLE patch_provenance ADD COLUMN semantic_alerts TEXT"
            )
        cols = [r[1] for r in conn.execute("PRAGMA table_info(patch_ancestry)").fetchall()]
        if "license" not in cols:
            conn.execute("ALTER TABLE patch_ancestry ADD COLUMN license TEXT")
        if "license_fingerprint" not in cols:
            conn.execute(
                "ALTER TABLE patch_ancestry ADD COLUMN license_fingerprint TEXT"
            )
        if "semantic_alerts" not in cols:
            conn.execute(
                "ALTER TABLE patch_ancestry ADD COLUMN semantic_alerts TEXT"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patch_filename ON patch_history(filename)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_patch_ts ON patch_history(ts)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patch_roi_delta ON patch_history(roi_delta)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patch_parent ON patch_history(parent_patch_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_flaky_file ON flakiness_history(filename)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_provenance_patch ON patch_provenance(patch_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ancestry_patch ON patch_ancestry(patch_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ancestry_vector ON patch_ancestry(vector_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_contributors_patch ON patch_contributors(patch_id)"
        )
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        # expose connection for diagnostics and tests
        self.conn = self.router.get_connection("patch_history")
        self.conn.execute("PRAGMA foreign_keys = ON")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self.router.get_connection("patch_history")
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _with_conn(self, func: Callable[[sqlite3.Connection], T]) -> T:
        with self._connect() as conn:
            return func(conn)

    # ------------------------------------------------------------------
    def _extract_keywords(self, text: str) -> List[str]:
        """Return simple keyword list from ``text``."""
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return words

    # ------------------------------------------------------------------
    def _record_text(self, data: dict[str, Any]) -> str:
        """Return a descriptive text representation for embedding."""
        parts: list[str] = []
        summary = str(data.get("summary", ""))
        if summary:
            parts.append(f"summary: {summary}")
        code = str(data.get("code", ""))
        if code:
            parts.append(code)
        template = str(data.get("template_type", ""))
        if template:
            parts.append(f"template_type: {template}")
        language = str(data.get("language", ""))
        if language:
            parts.append(f"language: {language}")
        return "\n".join(parts)

    def license_text(self, rec: CodeRecord | dict[str, Any]) -> str | None:
        if isinstance(rec, CodeRecord):
            return rec.code
        if isinstance(rec, dict):
            return rec.get("code")
        return None

    def backfill_embeddings(self, batch_size: int = 100) -> None:
        """Delegate to :class:`EmbeddableDBMixin` for compatibility."""
        EmbeddableDBMixin.backfill_embeddings(self)

    def iter_records(self) -> Iterator[tuple[int, dict[str, Any], str]]:
        """Yield code rows for embedding backfill."""
        menace_id = _current_menace_id(self.router)
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT id, summary, code, template_type, language FROM code WHERE source_menace_id=?",
                (menace_id,),
            )
            rows = cur.fetchall()
        for row in rows:
            data = dict(row) if isinstance(row, sqlite3.Row) else {
                "id": row[0],
                "summary": row[1],
                "code": row[2],
                "template_type": row[3],
                "language": row[4],
            }
            yield data["id"], data, "code"

    def vector(self, rec: Any) -> List[float] | None:
        """Return an embedding vector for ``rec`` or record id."""
        if isinstance(rec, (int, str)):
            rid = str(rec)
            meta = getattr(self, "_metadata", {}).get(rid)
            if meta and "vector" in meta:
                return meta["vector"]
            try:
                rec_id = int(rec)
            except (TypeError, ValueError):
                return None
            menace_id = _current_menace_id(self.router)
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT summary, code, template_type, language FROM code WHERE id=? AND source_menace_id=?",
                    (rec_id, menace_id),
                ).fetchone()
            if not row:
                return None
            data = dict(row) if isinstance(row, sqlite3.Row) else {
                "summary": row[0],
                "code": row[1],
                "template_type": row[2],
                "language": row[3],
            }
            text = self._record_text(data)
            prepared = self._prepare_text_for_embedding(text)
            return self.encode_text(prepared) if prepared else None
        if isinstance(rec, CodeRecord):
            data = asdict(rec)
        elif isinstance(rec, dict):
            data = rec
        else:
            return None
        text = self._record_text(data)
        prepared = self._prepare_text_for_embedding(text)
        return self.encode_text(prepared) if prepared else None

    def search_by_vector(
        self, vector: Sequence[float], top_k: int = 5
    ) -> List[dict[str, Any]]:
        matches = EmbeddableDBMixin.search_by_vector(self, vector, top_k)
        results: List[dict[str, Any]] = []
        menace_id = _current_menace_id(self.router)
        with self._connect() as conn:
            for cid, dist in matches:
                row = conn.execute(
                    "SELECT id, summary, code FROM code WHERE id=? AND source_menace_id=?",
                    (cid, menace_id),
                ).fetchone()
                if row:
                    data = dict(row) if isinstance(row, sqlite3.Row) else {
                        "id": row[0],
                        "summary": row[1],
                        "code": row[2],
                    }
                    data["_distance"] = dist
                    results.append(data)
        return results

    def add(
        self, rec: PatchRecord, vectors: Sequence[tuple[str, float]] | None = None
    ) -> int:
        """Store a patch record and optional retrieval provenance."""

        def op(conn: sqlite3.Connection) -> int:
            if rec.code_hash is None and rec.filename:
                try:
                    with open(rec.filename, "rb") as fh:
                        rec.code_hash = _hash_code(fh.read())
                except Exception:
                    rec.code_hash = None

            if rec.code_id is None and self.code_db and rec.code_hash:
                try:
                    matches = [
                        row["id"]
                        for row in self.code_db.fetch_all(scope="all")
                        if _hash_code(row["code"]) == rec.code_hash
                    ]
                    if matches:
                        rec.code_id = matches[0]
                        if len(matches) > 1:
                            logger.warning(
                                "multiple code records share hash",
                                extra={"hash": rec.code_hash},
                            )
                except Exception:
                    pass

            cur = conn.execute(
                "INSERT INTO patch_history(filename, description, roi_before, roi_after, errors_before, errors_after, roi_delta, complexity_before, complexity_after, complexity_delta, entropy_before, entropy_after, entropy_delta, predicted_roi, predicted_errors, reverted, trending_topic, ts, code_id, code_hash, source_bot, version, parent_patch_id, reason, trigger, diff, summary, outcome, prompt_headers, prompt_order, prompt_tone) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rec.filename,
                    rec.description,
                    rec.roi_before,
                    rec.roi_after,
                    rec.errors_before,
                    rec.errors_after,
                    rec.roi_delta,
                    rec.complexity_before,
                    rec.complexity_after,
                    rec.complexity_delta,
                    rec.entropy_before,
                    rec.entropy_after,
                    rec.entropy_delta,
                    rec.predicted_roi,
                    rec.predicted_errors,
                    int(rec.reverted),
                    rec.trending_topic,
                    rec.ts,
                    rec.code_id,
                    rec.code_hash,
                    rec.source_bot,
                    rec.version,
                    rec.parent_patch_id,
                    rec.reason,
                    rec.trigger,
                    rec.diff,
                    rec.summary,
                    rec.outcome,
                    rec.prompt_headers,
                    rec.prompt_order,
                    rec.prompt_tone,
                ),
            )

            keywords: List[str] = []
            if rec.description:
                keywords.extend(self._extract_keywords(rec.description))
            if rec.trending_topic:
                keywords.extend(self._extract_keywords(rec.trending_topic))
            now = time.time()
            for kw in keywords:
                self.keyword_counts[kw] += 1
                self.keyword_recent[kw] = now

            patch_id = int(cur.lastrowid)
            if vectors:
                parsed: List[Tuple[str, str, float]] = []
                for vid, score in vectors:
                    if ":" in vid:
                        origin, vec_id = vid.split(":", 1)
                    else:
                        origin, vec_id = "", vid
                    parsed.append((origin, vec_id, float(score)))
                self._insert_provenance(conn, patch_id, parsed)
            logger.info(
                "patch stored",
                extra={
                    "patch_id": patch_id,
                    "file": rec.filename,
                    "code_id": rec.code_id,
                },
            )
            return patch_id

        return with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def get(self, patch_id: int) -> PatchRecord | None:
        """Return patch record for ``patch_id`` if present."""

        def op(conn: sqlite3.Connection) -> PatchRecord | None:
            row = conn.execute(
                "SELECT filename, description, roi_before, roi_after, errors_before, errors_after, roi_delta, complexity_before, complexity_after, complexity_delta, entropy_before, entropy_after, entropy_delta, predicted_roi, predicted_errors, reverted, trending_topic, ts, code_id, code_hash, source_bot, version, parent_patch_id, reason, trigger, diff, summary, outcome, prompt_headers, prompt_order, prompt_tone FROM patch_history WHERE id=?",
                (patch_id,),
            ).fetchone()
            return PatchRecord(*row) if row else None

        return with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def top_patches(self, limit: int = 5) -> List[PatchRecord]:
        """Return the highest ROI patches."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT filename, description, roi_before, roi_after, errors_before, errors_after, roi_delta, complexity_before, complexity_after, complexity_delta, entropy_before, entropy_after, entropy_delta, predicted_roi, predicted_errors, reverted, trending_topic, ts, code_id, code_hash, source_bot, version, parent_patch_id, reason, trigger, diff, summary, outcome, prompt_headers, prompt_order, prompt_tone FROM patch_history ORDER BY roi_delta DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        patches = [PatchRecord(*row) for row in rows]
        logger.info("patch query", extra={"count": len(patches)})
        return patches

    def success_rate(self, limit: int = 50) -> float:
        """Return the fraction of patches that improved ROI."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT roi_delta FROM patch_history ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        if not rows:
            return 0.0
        success = sum(1 for d, in rows if d > 0)
        return float(success) / len(rows)

    def record_vector_metrics(
        self,
        session_id: str,
        vectors: list[tuple[str, str]],
        *,
        patch_id: int,
        contribution: float,
        roi_delta: float | None = None,
        win: bool,
        regret: bool,
        lines_changed: int | None = None,
        tests_passed: bool | None = None,
        context_tokens: int | None = None,
        patch_difficulty: int | None = None,
        effort_estimate: float | None = None,
        enhancement_name: str | None = None,
        timestamp: float | None = None,
        start_time: float | None = None,
        time_to_completion: float | None = None,
        roi_deltas: Mapping[str, float] | None = None,
        errors: Sequence[Mapping[str, Any]] | None = None,
        error_trace_count: int | None = None,
        roi_tag: str | None = None,
        enhancement_score: float | None = None,
    ) -> None:
        """Update :class:`VectorMetricsDB` rows for *session_id* and *vectors*."""

        if self._vec_db is None:
            return
        try:
            self._vec_db.update_outcome(
                session_id,
                vectors,
                contribution=contribution,
                patch_id=str(patch_id),
                win=win,
                regret=regret,
            )
        except Exception:
            logger.exception("vector metrics update failed")

        try:
            conn = self.router.get_connection("patch_history")
            roi_json = "{}"
            if roi_deltas:
                try:
                    roi_json = json.dumps(roi_deltas)
                except Exception:
                    logger.exception("failed to serialise roi_deltas")
            err_json = "[]"
            if errors:
                try:
                    err_json = json.dumps(list(errors))
                except Exception:
                    logger.exception("failed to serialise errors")
            conn.execute(
                "UPDATE patch_history SET lines_changed=?, tests_passed=?, context_tokens=?, patch_difficulty=?, effort_estimate=?, enhancement_name=?, start_time=?, time_to_completion=?, timestamp=?, diff=COALESCE(?, diff), summary=COALESCE(?, summary), outcome=COALESCE(?, outcome), roi_delta=COALESCE(?, roi_delta), roi_deltas=COALESCE(?, roi_deltas), errors=COALESCE(?, errors), error_trace_count=COALESCE(?, error_trace_count), roi_tag=COALESCE(?, roi_tag), enhancement_score=COALESCE(?, enhancement_score) WHERE id=?",
                (
                    lines_changed,
                    None if tests_passed is None else int(bool(tests_passed)),
                    context_tokens,
                    patch_difficulty,
                    effort_estimate,
                    enhancement_name,
                    start_time,
                    time_to_completion,
                    timestamp,
                    diff,
                    summary,
                    outcome,
                    roi_delta,
                    roi_json,
                    err_json,
                    error_trace_count,
                    roi_tag,
                    enhancement_score,
                    patch_id,
                ),
            )
            conn.commit()
        except Exception:
            logger.exception("failed to update patch history with metrics")

        if self._vec_db is not None:
            try:
                self._vec_db.record_patch_summary(
                    str(patch_id),
                    errors=errors,
                    tests_passed=tests_passed,
                    lines_changed=lines_changed,
                    context_tokens=context_tokens,
                    patch_difficulty=patch_difficulty,
                    effort_estimate=effort_estimate,
                    start_time=start_time,
                    time_to_completion=time_to_completion,
                    error_trace_count=error_trace_count,
                    roi_tag=roi_tag,
                    enhancement_score=enhancement_score,
                )
            except Exception:
                logger.exception("vector metrics patch summary failed")

    # ------------------------------------------------------------------
    def _insert_provenance(
        self,
        conn: sqlite3.Connection,
        patch_id: int,
        vectors: Sequence[tuple],
        *,
        ts: str | None = None,
    ) -> None:
        ts = ts or datetime.utcnow().isoformat()
        for pos, vec in enumerate(vectors):
            parts = list(vec)
            origin = parts[0] if len(parts) > 0 else None
            vec_id = parts[1] if len(parts) > 1 else None
            score = parts[2] if len(parts) > 2 else None
            lic = parts[3] if len(parts) > 3 else None
            fp = None
            alerts = None
            if len(parts) > 5:
                fp = parts[4]
                alerts = parts[5]
            elif len(parts) > 4:
                alerts = parts[4]
            conn.execute(
                "INSERT INTO patch_provenance(patch_id, origin, vector_id, influence, retrieved_at, position, license, license_fingerprint, semantic_alerts) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    patch_id,
                    origin,
                    vec_id,
                    score,
                    ts,
                    pos,
                    lic,
                    fp,
                    json.dumps(alerts) if alerts is not None else None,
                ),
            )

    def record_provenance(self, patch_id: int, vectors: Sequence[tuple]) -> None:
        """Persist retrieval provenance for a patch."""

        def op(conn: sqlite3.Connection) -> None:
            self._insert_provenance(conn, patch_id, vectors)

        with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def _insert_ancestry(
        self,
        conn: sqlite3.Connection,
        patch_id: int,
        vectors: Sequence[tuple],
    ) -> None:
        for vec in vectors:
            parts = list(vec)
            origin = parts[0] if len(parts) > 0 else None
            vec_id = parts[1] if len(parts) > 1 else None
            influence = parts[2] if len(parts) > 2 else None
            lic = parts[3] if len(parts) > 3 else None
            fp = None
            alerts = None
            if len(parts) > 5:
                fp = parts[4]
                alerts = parts[5]
            elif len(parts) > 4:
                alerts = parts[4]
            conn.execute(
                "INSERT INTO patch_ancestry(patch_id, origin, vector_id, influence, license, license_fingerprint, semantic_alerts) VALUES(?,?,?,?,?,?,?)",
                (
                    patch_id,
                    origin,
                    vec_id,
                    influence,
                    lic,
                    fp,
                    json.dumps(alerts) if alerts is not None else None,
                ),
            )

    def _insert_contributors(
        self,
        conn: sqlite3.Connection,
        patch_id: int,
        vectors: Sequence[tuple[str, str, float]],
        session_id: str,
    ) -> None:
        for origin, vec_id, influence in vectors:
            conn.execute(
                "INSERT INTO patch_contributors(patch_id, vector_id, influence, session_id) VALUES(?,?,?,?)",
                (patch_id, f"{origin}:{vec_id}" if origin else vec_id, influence, session_id),
            )

    def log_ancestry(self, patch_id: int, vectors: Sequence[tuple]) -> None:
        """Persist vector ancestry for a patch."""

        def op(conn: sqlite3.Connection) -> None:
            self._insert_ancestry(conn, patch_id, vectors)

        with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def log_contributors(
        self,
        patch_id: int,
        vectors: Sequence[tuple[str, str, float]],
        session_id: str,
    ) -> None:
        """Persist contributor vectors for a patch."""

        def op(conn: sqlite3.Connection) -> None:
            self._insert_contributors(conn, patch_id, vectors, session_id)

        with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def get_contributors(
        self, patch_id: int
    ) -> List[Tuple[str, float, str]]:
        """Return contributors for ``patch_id`` ordered by influence."""

        def op(conn: sqlite3.Connection) -> List[Tuple[str, float, str]]:
            rows = conn.execute(
                "SELECT vector_id, influence, session_id FROM patch_contributors WHERE patch_id=? ORDER BY influence DESC",
                (patch_id,),
            ).fetchall()
            return [(v, float(i), s) for v, i, s in rows]

        return with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def log_branch_event(self, patch_id: str, branch: str, action: str) -> None:
        """Record branch-related actions for a patch."""

        def op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO branch_log(patch_id, branch, action, ts) VALUES(?,?,?,?)",
                (patch_id, branch, action, datetime.utcnow().isoformat()),
            )

        with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def get_ancestry(self, patch_id: int) -> List[Tuple[str, str, float, str | None, str | None, str | None]]:
        """Return ancestry rows for ``patch_id`` ordered by influence."""

        def op(
            conn: sqlite3.Connection,
        ) -> List[Tuple[str, str, float, str | None, str | None, str | None]]:
            rows = conn.execute(
                "SELECT origin, vector_id, influence, license, license_fingerprint, semantic_alerts FROM patch_ancestry WHERE patch_id=? ORDER BY influence DESC",
                (patch_id,),
            ).fetchall()
            return [
                (o, v, float(i), lic, fp, alerts) for o, v, i, lic, fp, alerts in rows
            ]

        return with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def get_ancestry_chain(self, patch_id: int) -> List[Tuple[int, PatchRecord]]:
        """Return the ancestry chain for ``patch_id``.

        Each entry in the returned list is a ``(id, PatchRecord)`` tuple.  The
        list starts with ``patch_id`` and follows ``parent_patch_id`` links until
        reaching the root of the chain.
        """

        chain: List[Tuple[int, PatchRecord]] = []
        current: Optional[int] = patch_id
        while current is not None:
            rec = self.get(current)
            if rec is None:
                break
            chain.append((current, rec))
            current = rec.parent_patch_id
        return chain

    def find_patches_by_vector(
        self,
        vector_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
        index_hint: str | None = None,
    ) -> List[Tuple[int, float, str, str]]:
        """Return patches influenced by ``vector_id`` ordered by influence.

        Supports pagination via ``limit`` and ``offset`` and allows forcing a
        specific SQLite ``index_hint`` for the ``patch_ancestry`` table.
        Results are joined with :class:`PatchRecord` metadata.
        """

        pattern = f"%{vector_id}%"

        def op(conn: sqlite3.Connection) -> List[Tuple[int, float, str, str]]:
            query = (
                "SELECT a.patch_id, a.influence, h.filename, h.description "
                "FROM patch_ancestry"
            )
            if index_hint:
                query += f" INDEXED BY {index_hint}"
            query += " a JOIN patch_history h ON h.id=a.patch_id "
            query += "WHERE a.vector_id LIKE ? ORDER BY a.influence DESC"
            params: List[Any] = [pattern]
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            if offset:
                query += " OFFSET ?"
                params.append(offset)
            rows = conn.execute(query, params).fetchall()
            return [
                (int(pid), float(infl), fname, desc)
                for pid, infl, fname, desc in rows
            ]

        return with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def find_patches_by_hash(self, code_hash: str) -> List[Tuple[int, str, str]]:
        """Return patches matching ``code_hash``.

        Results include the patch id, filename and description.
        """

        def op(conn: sqlite3.Connection) -> List[Tuple[int, str, str]]:
            rows = conn.execute(
                "SELECT id, filename, description FROM patch_history WHERE code_hash=?",
                (code_hash,),
            ).fetchall()
            return [(int(pid), fname, desc) for pid, fname, desc in rows]

        return with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def find_patches_by_provenance(
        self,
        license: str | None = None,
        semantic_alert: str | None = None,
        license_fingerprint: str | None = None,
    ) -> List[Tuple[int, str, str]]:
        """Return patches filtered by license, fingerprint and/or semantic alert.

        ``license_fingerprint`` matches the exact fingerprint string stored in
        the ancestry tables, while ``semantic_alert`` performs a substring match
        against the JSON encoded alerts list.
        """

        def op(conn: sqlite3.Connection) -> List[Tuple[int, str, str]]:
            query = (
                "SELECT DISTINCT a.patch_id, h.filename, h.description "
                "FROM patch_ancestry a JOIN patch_history h ON h.id=a.patch_id"
            )
            conditions: List[str] = []
            params: List[str] = []
            if license is not None:
                conditions.append("a.license=?")
                params.append(license)
            if license_fingerprint is not None:
                conditions.append("a.license_fingerprint=?")
                params.append(license_fingerprint)
            if semantic_alert is not None:
                conditions.append("a.semantic_alerts LIKE ?")
                params.append(f'%"{semantic_alert}"%')
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY a.patch_id"
            rows = conn.execute(query, params).fetchall()
            return [(int(pid), fname, desc) for pid, fname, desc in rows]

        return with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def get_provenance(self, patch_id: int) -> List[Tuple[str, str, float, str]]:
        """Return provenance rows for ``patch_id`` ordered by original ranking."""

        def op(conn: sqlite3.Connection) -> List[Tuple[str, str, float, str]]:
            rows = conn.execute(
                "SELECT origin, vector_id, influence, retrieved_at FROM patch_provenance WHERE patch_id=? ORDER BY position",
                (patch_id,),
            ).fetchall()
            return [(o, v, float(s), t) for o, v, s, t in rows]

        return with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def keyword_features(self) -> tuple[int, int]:
        """Return count and recency (hrs) for the most common keyword."""
        if not self.keyword_counts:
            return 0, 0
        kw, count = self.keyword_counts.most_common(1)[0]
        recency = int((time.time() - self.keyword_recent.get(kw, time.time())) / 3600)
        return count, recency

    def filter(
        self,
        *,
        filename: str | None = None,
        reverted: bool | None = None,
        parent_patch_id: int | None = None,
    ) -> List[PatchRecord]:
        """Filter patches by filename, reverted flag, and lineage."""

        def op(conn: sqlite3.Connection) -> List[PatchRecord]:
            base = (
                "SELECT filename, description, roi_before, roi_after, errors_before, errors_after, "
                "roi_delta, complexity_before, complexity_after, complexity_delta, entropy_before, entropy_after, entropy_delta, predicted_roi, "
                "predicted_errors, reverted, trending_topic, ts, code_id, code_hash, source_bot, version, parent_patch_id, reason, trigger, diff, summary, outcome, prompt_headers, prompt_order, prompt_tone FROM patch_history"
            )
            clauses: List[str] = []
            params: List[Any] = []
            if filename:
                clauses.append("filename = ?")
                params.append(filename)
            if reverted is not None:
                clauses.append("reverted = ?")
                params.append(int(reverted))
            if parent_patch_id is not None:
                clauses.append("parent_patch_id = ?")
                params.append(parent_patch_id)
            if clauses:
                base += " WHERE " + " AND ".join(clauses)
            rows = conn.execute(base, params).fetchall()
            patches = [PatchRecord(*row) for row in rows]
            logger.info(
                "patch filter",
                extra={
                    "file": filename,
                    "reverted": reverted,
                    "count": len(patches),
                },
            )
            return patches

        return with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def search(self, term: str) -> List[PatchRecord]:
        """Search patch descriptions and filenames."""
        pattern = f"%{term}%"

        def op(conn: sqlite3.Connection) -> List[PatchRecord]:
            rows = conn.execute(
                "SELECT filename, description, roi_before, roi_after, errors_before, errors_after, roi_delta, complexity_before, complexity_after, complexity_delta, entropy_before, entropy_after, entropy_delta, predicted_roi, predicted_errors, reverted, trending_topic, ts, code_id, code_hash, source_bot, version, parent_patch_id, reason, trigger, diff, summary, outcome, prompt_headers, prompt_order, prompt_tone FROM patch_history WHERE description LIKE ? COLLATE NOCASE OR filename LIKE ? COLLATE NOCASE",
                (pattern, pattern),
            ).fetchall()
            patches = [PatchRecord(*row) for row in rows]
            logger.info("patch search", extra={"term": term, "count": len(patches)})
            return patches

        return with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def between_dates(self, start: datetime, end: datetime) -> List[PatchRecord]:
        """Return patches recorded between ``start`` and ``end``."""

        def op(conn: sqlite3.Connection) -> List[PatchRecord]:
            rows = conn.execute(
                "SELECT filename, description, roi_before, roi_after, errors_before, errors_after, roi_delta, complexity_before, complexity_after, complexity_delta, entropy_before, entropy_after, entropy_delta, predicted_roi, predicted_errors, reverted, trending_topic, ts, code_id, code_hash, source_bot, version, parent_patch_id, reason, trigger, diff, summary, outcome, prompt_headers, prompt_order, prompt_tone FROM patch_history WHERE ts BETWEEN ? AND ?",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
            patches = [PatchRecord(*row) for row in rows]
            logger.info(
                "patch range",
                extra={
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "count": len(patches),
                },
            )
            return patches

        return with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def by_hash(self, code_hash: str) -> List[PatchRecord]:
        """Return all patches matching *code_hash*."""

        def op(conn: sqlite3.Connection) -> List[PatchRecord]:
            rows = conn.execute(
                "SELECT filename, description, roi_before, roi_after, errors_before, errors_after, roi_delta, complexity_before, complexity_after, complexity_delta, entropy_before, entropy_after, entropy_delta, predicted_roi, predicted_errors, reverted, trending_topic, ts, code_id, code_hash, source_bot, version, parent_patch_id, reason, trigger, diff, summary, outcome, prompt_headers, prompt_order, prompt_tone FROM patch_history WHERE code_hash=?",
                (code_hash,),
            ).fetchall()
            patches = [PatchRecord(*row) for row in rows]
            logger.info(
                "patch by_hash",
                extra={"hash": code_hash, "count": len(patches)},
            )
            return patches

        return with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    # ------------------------------------------------------------------
    def record_failed_strategy(self, code_hash: str) -> None:
        """Persist a failed strategy hash for future exclusion."""

        def op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR IGNORE INTO failed_strategies(code_hash, ts) VALUES(?, ?)",
                (code_hash, datetime.utcnow().isoformat()),
            )

        with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def has_failed_strategy(self, code_hash: str) -> bool:
        """Return ``True`` if ``code_hash`` was previously marked as failed."""

        def op(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                "SELECT 1 FROM failed_strategies WHERE code_hash=? LIMIT 1",
                (code_hash,),
            ).fetchone()
            return bool(row)

        return with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def failed_strategy_hashes(self) -> List[str]:
        """Return all recorded failed strategy hashes."""

        def op(conn: sqlite3.Connection) -> List[str]:
            rows = conn.execute(
                "SELECT code_hash FROM failed_strategies"
            ).fetchall()
            return [r[0] for r in rows]

        return with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    # ------------------------------------------------------------------
    def list_patches(self, limit: int | None = None) -> List[tuple[int, PatchRecord]]:
        """Return ``(id, PatchRecord)`` tuples for recent patches."""

        def op(conn: sqlite3.Connection) -> List[tuple[int, PatchRecord]]:
            base = (
                "SELECT id, filename, description, roi_before, roi_after, errors_before, errors_after, "
                "roi_delta, complexity_before, complexity_after, complexity_delta, entropy_before, entropy_after, entropy_delta, "
                "predicted_roi, predicted_errors, reverted, trending_topic, ts, code_id, code_hash, source_bot, version, parent_patch_id, reason, trigger, diff, summary, outcome, prompt_headers, prompt_order, prompt_tone FROM patch_history ORDER BY id DESC"
            )
            rows = conn.execute(base + (" LIMIT ?" if limit is not None else ""),
                                 (() if limit is None else (limit,))).fetchall()
            return [(row[0], PatchRecord(*row[1:])) for row in rows]

        return with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    # ------------------------------------------------------------------
    def search_with_ids(
        self,
        term: str,
        *,
        limit: int | None = None,
        offset: int = 0,
        index_hint: str | None = None,
    ) -> List[tuple[int, PatchRecord]]:
        """Search patches and include their IDs."""
        pattern = f"%{term}%"

        def op(conn: sqlite3.Connection) -> List[tuple[int, PatchRecord]]:
            query = (
                "SELECT id, filename, description, roi_before, roi_after, errors_before, errors_after, roi_delta, complexity_before, complexity_after, complexity_delta, entropy_before, entropy_after, entropy_delta, predicted_roi, predicted_errors, reverted, trending_topic, ts, code_id, code_hash, source_bot, version, parent_patch_id, reason, trigger, diff, summary, outcome, prompt_headers, prompt_order, prompt_tone FROM patch_history"
            )
            if index_hint:
                query += f" INDEXED BY {index_hint}"
            query += " WHERE description LIKE ? COLLATE NOCASE OR filename LIKE ? COLLATE NOCASE"
            params: List[Any] = [pattern, pattern]
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            if offset:
                query += " OFFSET ?"
                params.append(offset)
            rows = conn.execute(query, params).fetchall()
            return [(row[0], PatchRecord(*row[1:])) for row in rows]

        return with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    # ------------------------------------------------------------------
    def find_by_vector(self, vector: str) -> List[tuple[int, PatchRecord]]:
        """Return patches that include ``vector`` in their provenance."""

        def op(conn: sqlite3.Connection) -> List[tuple[int, PatchRecord]]:
            if ":" in vector:
                origin, vid = vector.split(":", 1)
                rows = conn.execute(
                    "SELECT h.id, h.filename, h.description, h.roi_before, h.roi_after, h.errors_before, h.errors_after, h.roi_delta, h.complexity_before, h.complexity_after, h.complexity_delta, h.entropy_before, h.entropy_after, h.entropy_delta, h.predicted_roi, h.predicted_errors, h.reverted, h.trending_topic, h.ts, h.code_id, h.code_hash, h.source_bot, h.version, h.parent_patch_id, h.reason, h.trigger, h.diff, h.summary, h.outcome FROM patch_provenance p JOIN patch_history h ON h.id=p.patch_id WHERE p.origin=? AND p.vector_id=?",
                    (origin, vid),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT h.id, h.filename, h.description, h.roi_before, h.roi_after, h.errors_before, h.errors_after, h.roi_delta, h.complexity_before, h.complexity_after, h.complexity_delta, h.entropy_before, h.entropy_after, h.entropy_delta, h.predicted_roi, h.predicted_errors, h.reverted, h.trending_topic, h.ts, h.code_id, h.code_hash, h.source_bot, h.version, h.parent_patch_id, h.reason, h.trigger, h.diff, h.summary, h.outcome FROM patch_provenance p JOIN patch_history h ON h.id=p.patch_id WHERE p.vector_id=?",
                    (vector,),
                ).fetchall()
            return [(row[0], PatchRecord(*row[1:])) for row in rows]

        return with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def store_weights(
        self, weights: tuple[float, float, float, float, float, float]
    ) -> None:
        """Persist score weights for later reuse."""

        def op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO score_weights(w1, w2, w3, w4, w5, w6, ts) VALUES(?,?,?,?,?,?,?)",
                (*weights, datetime.utcnow().isoformat()),
            )

        with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def get_weights(self) -> tuple[float, float, float, float, float, float] | None:
        """Return the most recently stored score weights."""

        def op(conn: sqlite3.Connection) -> tuple | None:
            return conn.execute(
                "SELECT w1, w2, w3, w4, w5, w6 FROM score_weights ORDER BY id DESC LIMIT 1"
            ).fetchone()

        row = with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)
        if row:
            return tuple(float(x) for x in row)
        return None

    def record_flakiness(self, filename: str, flakiness: float) -> None:
        """Store flakiness measurement for *filename*."""

        def op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO flakiness_history(filename, flakiness, ts) VALUES(?,?,?)",
                (filename, float(flakiness), datetime.utcnow().isoformat()),
            )

        with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)

    def average_flakiness(self, filename: str, limit: int = 20) -> float:
        """Return average flakiness for *filename* from recent history."""

        def op(conn: sqlite3.Connection) -> list[tuple[float]]:
            return conn.execute(
                "SELECT flakiness FROM flakiness_history WHERE filename=? ORDER BY id DESC LIMIT ?",
                (filename, limit),
            ).fetchall()

        rows = with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)
        vals = [float(r[0]) for r in rows]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    def delete(self, patch_id: int) -> None:
        """Remove a patch record if it exists."""

        def op(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM patch_history WHERE id=?", (patch_id,))

        with_retry(lambda: self._with_conn(op), exc=sqlite3.Error, logger=logger)
        logger.info("patch deleted", extra={"patch_id": patch_id})


sys.modules["code_database"] = sys.modules[__name__]

__all__ = ["CodeRecord", "CodeDB", "PatchRecord", "PatchHistoryDB"]
