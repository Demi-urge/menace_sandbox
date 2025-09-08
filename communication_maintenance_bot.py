# flake8: noqa
"""Communication Maintenance Bot for updates and resource optimisation."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from loguru import logger as loguru_logger
import os
import atexit
import tempfile
import time
try:  # optional dependency
    from sqlalchemy import Column, String, Text, Table, MetaData, create_engine
    from sqlalchemy.engine import Engine
    from sqlalchemy.exc import SQLAlchemyError
except Exception:  # pragma: no cover - fallback when SQLAlchemy missing
    Column = String = Text = Table = MetaData = create_engine = None  # type: ignore
    Engine = SQLAlchemyError = None  # type: ignore
from dataclasses import dataclass, asdict, field
from pydantic import BaseModel, ValidationError, Field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, TypeVar
import concurrent.futures
from urllib.parse import urlparse

from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError, NoSuchPathError
try:  # optional redis for locking
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional
    redis = None  # type: ignore
from filelock import FileLock, Timeout
from threading import Lock
from .retry_utils import with_retry
from .audit_logger import log_event as audit_log_event
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore
try:
    import aiohttp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    aiohttp = None  # type: ignore
try:  # optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None  # type: ignore
import asyncio
from . import env_config
from .env_config import MENACE_MODE
from .db_router import DBRouter, GLOBAL_ROUTER, init_db_router
from .admin_bot_base import AdminBotBase
from .unified_event_bus import UnifiedEventBus
from .menace_memory_manager import MenaceMemoryManager, MemoryEntry

logger = logging.getLogger(__name__)
router = GLOBAL_ROUTER or init_db_router("communication_maintenance")

SESSION_NOW = datetime.utcnow()

T = TypeVar("T")


def ensure_directory(path: Path) -> None:
    """Create ``path`` if missing with a file lock to prevent races."""
    lock = FileLock(str(path) + ".lock", timeout=FILE_LOCK_TIMEOUT)
    with lock:
        path.mkdir(parents=True, exist_ok=True)


def archive_file(path: Path, *, suffix: str = ".bak") -> Path:
    """Rename ``path`` adding ``suffix`` and return the new path."""
    dest = path.with_suffix(path.suffix + suffix)
    counter = 1
    while dest.exists():
        dest = path.with_suffix(path.suffix + f"{suffix}{counter}")
        counter += 1
    lock = FileLock(str(path) + ".lock", timeout=FILE_LOCK_TIMEOUT)
    with lock:
        path.rename(dest)
    return dest


def safe_write(path: Path, data: str, mode: str = "w") -> None:
    """Write ``data`` to ``path`` using retries and locking."""

    def op() -> None:
        lock = FileLock(str(path) + ".lock", timeout=FILE_LOCK_TIMEOUT)
        with lock:
            try:
                with open(path, mode, encoding="utf-8") as fh:
                    fh.write(data)
            except OSError as exc:
                logger.error("file write failed: %s", exc, exc_info=True)
                raise FileOperationError(str(exc)) from exc

    with_retry(
        op,
        attempts=FILE_ATTEMPTS,
        delay=FILE_DELAY,
        exc=FileOperationError,
        logger=logger,
    )

def configure_logger(
    target: logging.Logger | None = None,
    level: str | int | None = None,
    *,
    json_format: bool | None = None,
) -> logging.Logger:
    """Configure logging with a basic formatter or JSON output."""
    log_obj = target or logger
    if getattr(log_obj, "_configured", False):
        return log_obj
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if json_format is None:
        json_format = os.getenv("LOG_JSON", "0") == "1"
    if not log_obj.handlers:
        handlers: List[logging.Handler] = [logging.StreamHandler()]
        if LOG_FILE_PATH:
            ensure_directory(Path(LOG_FILE_PATH).parent)
            handlers.append(
                RotatingFileHandler(
                    LOG_FILE_PATH,
                    maxBytes=LOG_ROTATE_BYTES,
                    backupCount=LOG_ROTATE_COUNT,
                )
            )

        for handler in handlers:
            if json_format:
                class JsonFormatter(logging.Formatter):
                    def format(self, record: logging.LogRecord) -> str:  # type: ignore
                        payload = {
                            "ts": datetime.utcnow().isoformat(),
                            "level": record.levelname,
                            "name": record.name,
                            "message": record.getMessage(),
                            "bot": getattr(record, "bot_name", os.getenv("MAINTENANCE_BOT_NAME", "maintenance_bot")),
                        }
                        if hasattr(record, "task"):
                            payload["task"] = record.task
                        if hasattr(record, "tag"):
                            payload["tag"] = record.tag
                        if record.exc_info:
                            payload["exc"] = self.formatException(record.exc_info)
                        return json.dumps(payload)

                handler.setFormatter(JsonFormatter())
            else:
                class SafeFormatter(logging.Formatter):
                    def format(self, record: logging.LogRecord) -> str:  # type: ignore
                        if not hasattr(record, "tag"):
                            record.tag = "-"
                        return super().format(record)

                handler.setFormatter(
                    SafeFormatter("%(asctime)s %(levelname)s %(name)s [%(tag)s]: %(message)s")
                )
            log_obj.addHandler(handler)
    log_obj.setLevel(level)
    setattr(log_obj, "_configured", True)
    return log_obj

try:
    from celery import Celery
except ImportError:  # pragma: no cover - optional
    Celery = None  # type: ignore

CHECK_INTERVAL = float(os.getenv("MAINTENANCE_CHECK_INTERVAL", "3600"))
OPTIMISE_INTERVAL = float(os.getenv("MAINTENANCE_OPTIMISE_INTERVAL", "86400"))

# Retry and scheduling configuration
LOCK_RETRY_ATTEMPTS = int(os.getenv("MAINTENANCE_LOCK_RETRY_ATTEMPTS", "3"))
LOCK_RETRY_DELAY = float(os.getenv("MAINTENANCE_LOCK_RETRY_DELAY", "1"))
TASK_RETRY_DELAY = float(os.getenv("MAINTENANCE_TASK_RETRY_DELAY", "30"))
TASK_TIME_LIMIT = int(os.getenv("MAINTENANCE_TASK_TIME_LIMIT", "60"))
HEARTBEAT_INTERVAL = float(os.getenv("MAINTENANCE_HEARTBEAT_INTERVAL", "3600"))
MESSAGE_RETRY_ATTEMPTS = int(os.getenv("MAINTENANCE_MESSAGE_RETRY_ATTEMPTS", "3"))
MESSAGE_RETRY_DELAY = float(os.getenv("MAINTENANCE_MESSAGE_RETRY_DELAY", "1"))
DIAGNOSTICS_WEBHOOK = os.getenv("MAINTENANCE_DISCORD_WEBHOOK")
DIAGNOSTICS_WEBHOOKS = os.getenv(
    "MAINTENANCE_DISCORD_WEBHOOKS",
    DIAGNOSTICS_WEBHOOK or "",
)
MESSAGE_TIMEOUT = float(os.getenv("MAINTENANCE_MESSAGE_TIMEOUT", "5"))
MESSAGE_MAX_LENGTH = int(os.getenv("MAINTENANCE_MESSAGE_MAX_LENGTH", "2000"))
LOG_DIR = os.getenv("MAINTENANCE_LOG_DIR", "maintenance-logs")
MESSAGE_QUEUE_PATH = os.getenv(
    "MAINTENANCE_MESSAGE_QUEUE",
    os.path.join(LOG_DIR, "maintenance_msg_queue.jsonl"),
)

# Logging enhancement configuration
LOG_FILE_PATH = os.getenv(
    "MAINTENANCE_LOG_FILE", os.path.join(LOG_DIR, "maintenance.log")
)
LOG_ROTATE_BYTES = int(
    os.getenv("MAINTENANCE_LOG_ROTATE_BYTES", str(5 * 1024 * 1024))
)
LOG_ROTATE_COUNT = int(os.getenv("MAINTENANCE_LOG_ROTATE_COUNT", "5"))

# Fallback webhook URLs for critical alerts
FALLBACK_WEBHOOKS = os.getenv("MAINTENANCE_FALLBACK_WEBHOOKS", "")

# Message templates
from string import Template

DEFAULT_TEMPLATES: dict[str, Template] = {
    "heartbeat": Template("$bot heartbeat $ts"),
    "update_status": Template("$bot update status: $status"),
    "hotfix_applied": Template("$bot hotfix applied: $desc"),
    "optimise_summary": Template("$bot optimisation: $summary"),
    "allocation_updated": Template("$bot resources adjusted: $actions"),
}

def load_templates(path: str | None = None) -> dict[str, Template]:
    """Load message templates from ``path`` if present."""
    path = path or os.getenv("MAINTENANCE_TEMPLATE_PATH", "")
    templates = DEFAULT_TEMPLATES.copy()
    if path and Path(path).exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for k, v in data.items():
                templates[k] = Template(str(v))
        except Exception as exc:  # pragma: no cover - malformed template file
            logger.warning("failed loading templates: %s", exc)
    return templates

# Communication log configuration
COMM_LOG_PATH = os.getenv(
    "MAINTENANCE_COMM_LOG_PATH",
    os.path.join(LOG_DIR, "communication_log.json"),
)
COMM_LOG_RETENTION_HOURS = float(os.getenv("MAINTENANCE_COMM_LOG_RETENTION_HOURS", "24"))
COMM_LOG_MAX_SIZE_MB = float(os.getenv("MAINTENANCE_COMM_LOG_MAX_SIZE_MB", "5"))

# Bot monitoring configuration
PING_INTERVAL = float(os.getenv("MAINTENANCE_PING_INTERVAL", "60"))
PING_RETRY_ATTEMPTS = int(os.getenv("MAINTENANCE_PING_RETRY_ATTEMPTS", "3"))
PING_RETRY_DELAY = float(os.getenv("MAINTENANCE_PING_RETRY_DELAY", "1"))
PING_TIMEOUT = float(os.getenv("MAINTENANCE_PING_TIMEOUT", "5"))
PING_FAILURE_LIMIT = int(os.getenv("MAINTENANCE_PING_FAILURE_LIMIT", "3"))
BOT_URLS = os.getenv("MAINTENANCE_BOT_URLS", "")

# Scheduler and safety configuration
SCHEDULER_POLL_INTERVAL = float(
    os.getenv("MAINTENANCE_SCHEDULER_POLL_INTERVAL", "0.1")
)
PING_MAX_DURATION = float(os.getenv("MAINTENANCE_PING_MAX_DURATION", "30"))
MESSAGE_MAX_DURATION = float(os.getenv("MAINTENANCE_MESSAGE_MAX_DURATION", "30"))
KILL_SWITCH_PATH = os.getenv("MAINTENANCE_KILL_SWITCH", "")

# Heartbeat recalibration thresholds
HEARTBEAT_LOAD_HIGH = float(os.getenv("MAINTENANCE_HEARTBEAT_LOAD_HIGH", "0.75"))
HEARTBEAT_LOAD_LOW = float(os.getenv("MAINTENANCE_HEARTBEAT_LOAD_LOW", "0.25"))
HEARTBEAT_MAX_FACTOR = float(os.getenv("MAINTENANCE_HEARTBEAT_MAX_FACTOR", "4"))
HEARTBEAT_MIN_FACTOR = float(os.getenv("MAINTENANCE_HEARTBEAT_MIN_FACTOR", "0.5"))

# Ping response validation
PING_EXPECT_KEY = os.getenv("MAINTENANCE_PING_EXPECT_KEY", "pong")
_ping_val = os.getenv("MAINTENANCE_PING_EXPECT_VALUE", "true").lower()
if _ping_val in {"1", "true", "yes"}:
    PING_EXPECT_VALUE = True
elif _ping_val in {"0", "false", "no"}:
    PING_EXPECT_VALUE = False
else:
    PING_EXPECT_VALUE = _ping_val

# File operation configuration
LOCK_TTL = int(os.getenv("MAINTENANCE_LOCK_TTL", "60"))
FILE_LOCK_TIMEOUT = float(os.getenv("MAINTENANCE_FILE_LOCK_TIMEOUT", "10"))
FILE_ATTEMPTS = int(os.getenv("MAINTENANCE_FILE_ATTEMPTS", "3"))
FILE_DELAY = float(os.getenv("MAINTENANCE_FILE_DELAY", "1"))
ERROR_DB_PATH = os.getenv("ERROR_DB_PATH", os.path.join(LOG_DIR, "errors.db"))
# Task execution retry configuration
RUN_TASK_ATTEMPTS = int(os.getenv("MAINTENANCE_TASK_ATTEMPTS", "3"))
RUN_TASK_BACKOFF = float(os.getenv("MAINTENANCE_TASK_BACKOFF", "1"))
# Admin tokens for privileged command execution
ADMIN_TOKENS = [
    t.strip() for t in os.getenv("MAINTENANCE_ADMIN_TOKENS", "").split(",") if t.strip()
]
# Minimum delay in seconds between successive commands of the same type
COMMAND_COOLDOWN = float(os.getenv("MAINTENANCE_COMMAND_COOLDOWN", "5"))

if os.getenv("MENACE_LIGHT_IMPORTS") and MENACE_MODE.lower() == "production":
    raise RuntimeError(
        "MENACE_LIGHT_IMPORTS cannot be enabled when MENACE_MODE=production"
    )

if os.getenv("MENACE_LIGHT_IMPORTS"):
    class ErrorDB:
        def __init__(self, path: Path | str = ERROR_DB_PATH, *_, **__):
            self.records: List[str] = []

        def log_discrepancy(self, message: str) -> None:
            self.records.append(message)

        def discrepancies(self) -> object:
            class _Dummy:
                empty = False

            return _Dummy()


    class ErrorBot:
        def __init__(self, db: ErrorDB, context_builder) -> None:
            self.db = db
            self.logger = logging.getLogger("CommMaintenanceBot.ErrorBot")
            self.context_builder = context_builder

        def handle_error(self, message: str) -> None:  # pragma: no cover - noop
            self.logger.error("noop ErrorBot handling error: %s", message)
            self.db.log_discrepancy(message)
else:
    from .error_bot import ErrorBot, ErrorDB
if os.getenv("MENACE_LIGHT_IMPORTS"):
    ResourceAllocationBot = None  # type: ignore
    AllocationDB = None  # type: ignore

    @dataclass
    class ResourceMetrics:
        cpu: float
        memory: float
        disk: float
        time: float
else:
    from .resource_allocation_bot import ResourceAllocationBot, AllocationDB
    from .resource_prediction_bot import ResourceMetrics


def _default_log_dir() -> Path:
    return Path(os.getenv("MAINTENANCE_LOG_DIR", "maintenance-logs"))


@dataclass
class MaintenanceBotConfig:
    """Configuration for CommunicationMaintenanceBot."""

    log_dir: Path = field(default_factory=_default_log_dir)
    comm_log_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "MAINTENANCE_COMM_LOG_PATH",
                str(_default_log_dir() / "communication_log.json"),
            )
        )
    )
    message_queue: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "MAINTENANCE_MESSAGE_QUEUE",
                str(_default_log_dir() / "maintenance_msg_queue.jsonl"),
            )
        )
    )
    error_db_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("ERROR_DB_PATH", str(_default_log_dir() / "errors.db"))
        )
    )
    comm_log_max_size_mb: float = field(
        default_factory=lambda: float(os.getenv("MAINTENANCE_COMM_LOG_MAX_SIZE_MB", "5"))
    )


class TaskType(str, Enum):
    """Canonical task identifiers used for maintenance logging."""

    UPDATE_CHECK = "update_check"
    HOTFIX = "hotfix"
    OPTIMISE = "optimise"
    ALLOCATION = "allocation"


class Severity(str, Enum):
    """Severity levels for maintenance events."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DeploymentEvent(BaseModel):
    """Schema for deployment events."""

    id: int


class MaintenanceRecord(BaseModel):
    """Log entry for a maintenance task."""

    task: str
    status: str
    details: str
    ts: str = Field(default_factory=lambda: SESSION_NOW.isoformat())


class CommLogEntry(BaseModel):
    """Structured communication log entry."""

    timestamp: str
    message: str


def _validate_positive(name: str, val: float) -> None:
    if not isinstance(val, (int, float)) or val <= 0:
        raise ValueError(f"{name} must be a positive number")


def _validate_config(cfg: MaintenanceBotConfig) -> None:
    if not isinstance(cfg.log_dir, Path):
        cfg.log_dir = Path(cfg.log_dir)
    ensure_directory(cfg.log_dir)
    for attr in ["comm_log_path", "message_queue", "error_db_path"]:
        path = getattr(cfg, attr)
        if not isinstance(path, Path):
            setattr(cfg, attr, Path(path))
        ensure_directory(getattr(cfg, attr).parent)
    if not isinstance(cfg.comm_log_max_size_mb, (int, float)) or cfg.comm_log_max_size_mb < 0:
        raise ValueError("comm_log_max_size_mb must be non-negative")


def _validate_message(content: str) -> str:
    if not isinstance(content, str):
        raise TypeError("message must be a string")
    content = content.strip()
    if not content:
        raise ValueError("message cannot be empty")
    if len(content) > MESSAGE_MAX_LENGTH * 10:
        raise ValueError("message exceeds maximum allowed length")
    return content


class FileOperationError(RuntimeError):
    """Raised when a file operation fails after retries."""


class RateLimitError(RuntimeError):
    """Signal that the request was rate limited."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"rate limited for {retry_after}s")
        self.retry_after = retry_after


class TaskLock:
    """Optional Redis-backed lock with file-based fallback."""

    def __init__(self, url: str | None = None) -> None:
        self.url = url or os.getenv("MAINTENANCE_LOCK_REDIS")
        self.client = None
        if self.url and redis:
            try:
                self.client = redis.from_url(self.url)
                self.client.ping()
            except Exception as exc:  # pragma: no cover - redis unavailable
                logging.getLogger(__name__).warning("redis lock unavailable: %s", exc)
                self.client = None
        self.locks: dict[str, FileLock] = {}
        if MENACE_MODE.lower() == "production" and not self.client:
            raise RuntimeError("Redis locking is required in production")
        self.attempts = LOCK_RETRY_ATTEMPTS
        self.delay = LOCK_RETRY_DELAY

    def _redis_acquire(self, name: str, ttl: int) -> bool:
        def op() -> bool:
            assert self.client
            return bool(self.client.set(name, "1", nx=True, ex=ttl))

        return with_retry(
            op,
            attempts=self.attempts,
            delay=self.delay,
            exc=Exception,
            logger=logging.getLogger(__name__),
        )

    def acquire(self, name: str, ttl: int = LOCK_TTL) -> bool:
        if self.client:
            try:
                return self._redis_acquire(name, ttl)
            except Exception as exc:  # pragma: no cover - redis issues
                msg = f"redis lock acquire failed: {exc}"
                if MENACE_MODE.lower() == "production":
                    raise RuntimeError(msg) from exc
                logging.getLogger(__name__).warning(msg)
        lock = self.locks.get(name)
        if not lock:
            temp_root = os.getenv("MAINTENANCE_TEMP_DIR", tempfile.gettempdir())
            lock_dir = Path(temp_root)
            ensure_directory(lock_dir)
            lock_path = lock_dir / f"{name}.lock"
            lock = FileLock(str(lock_path))
            self.locks[name] = lock
        try:
            lock.acquire(timeout=0)
            return True
        except Timeout:
            return False

    def _redis_release(self, name: str) -> None:
        def op() -> None:
            assert self.client
            self.client.delete(name)

        with_retry(
            op,
            attempts=self.attempts,
            delay=self.delay,
            exc=Exception,
            logger=logging.getLogger(__name__),
        )

    def release(self, name: str) -> None:
        if self.client:
            try:
                self._redis_release(name)
            except Exception as exc:  # pragma: no cover - redis issues
                msg = f"redis lock release failed: {exc}"
                if MENACE_MODE.lower() == "production":
                    raise RuntimeError(msg) from exc
                logging.getLogger(__name__).warning(msg)
        lock = self.locks.get(name)
        if lock and lock.is_locked:
            try:
                lock.release()
            except Exception:
                logger.exception("file lock release failed")


def _alloc_bot_instance() -> "ResourceAllocationBot":
    """Create a ResourceAllocationBot with graceful fallbacks."""
    RAB = ResourceAllocationBot
    ADB = AllocationDB
    if RAB is None or ADB is None:
        try:
            from .resource_allocation_bot import ResourceAllocationBot as _RAB
            from .resource_allocation_bot import AllocationDB as _ADB
        except Exception as exc:  # pragma: no cover - optional dependency missing
            raise RuntimeError("ResourceAllocationBot unavailable") from exc
        RAB = _RAB
        ADB = _ADB
    return RAB(ADB())


class MaintenanceStorageAdapter:
    """Interface for persistence backends used by CommunicationMaintenanceBot."""

    def log(self, rec: MaintenanceRecord) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def fetch(self) -> List[Tuple[str, str, str, str]]:  # pragma: no cover - interface
        raise NotImplementedError

    def set_state(self, key: str, value: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def get_state(self, key: str) -> Optional[str]:  # pragma: no cover - interface
        raise NotImplementedError


class SQLiteMaintenanceDB(MaintenanceStorageAdapter):
    """SQLite-backed persistent store for maintenance logs."""

    def __init__(self, path: Path | str | None = None) -> None:
        db_path = Path(path or os.environ.get("MAINTENANCE_DB", env_config.MAINTENANCE_DB))
        ensure_directory(db_path.parent)
        self.conn = router.get_connection("maintenance")
        self.path = db_path
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS maintenance(
                task TEXT,
                status TEXT,
                details TEXT,
                ts TEXT
            )
            """
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT)"
        )
        self.conn.commit()

    def log(self, rec: MaintenanceRecord) -> None:
        self.conn.execute(
            "INSERT INTO maintenance(task, status, details, ts) VALUES (?,?,?,?)",
            (rec.task, rec.status, rec.details, rec.ts),
        )
        self.conn.commit()

    def fetch(self) -> List[Tuple[str, str, str, str]]:
        cur = self.conn.execute(
            "SELECT task, status, details, ts FROM maintenance"
        )
        return [(r[0], r[1], r[2], r[3]) for r in cur.fetchall()]

    def set_state(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO state(key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    def get_state(self, key: str) -> Optional[str]:
        cur = self.conn.execute(
            "SELECT value FROM state WHERE key=?",
            (key,),
        )
        row = cur.fetchone()
        return row[0] if row else None


class PostgresMaintenanceDB(MaintenanceStorageAdapter):
    """PostgreSQL-backed persistent store for maintenance logs."""

    def __init__(self, dsn: str | None = None) -> None:
        import psycopg2  # type: ignore

        self.dsn = dsn or os.environ.get("MAINTENANCE_DB_URL")
        if not self.dsn:
            raise ValueError("MAINTENANCE_DB_URL is not set")
        self.conn = psycopg2.connect(self.dsn)
        with self.conn, self.conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS maintenance(
                    task TEXT,
                    status TEXT,
                    details TEXT,
                    ts TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS state(
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )

    def log(self, rec: MaintenanceRecord) -> None:
        with self.conn, self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO maintenance(task, status, details, ts) VALUES (%s,%s,%s,%s)",
                (rec.task, rec.status, rec.details, rec.ts),
            )

    def fetch(self) -> List[Tuple[str, str, str, str]]:
        with self.conn, self.conn.cursor() as cur:
            cur.execute("SELECT task, status, details, ts FROM maintenance")
            rows = cur.fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def set_state(self, key: str, value: str) -> None:
        with self.conn, self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO state(key, value)
                VALUES(%s, %s)
                ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value
                """,
                (key, value),
            )

    def get_state(self, key: str) -> Optional[str]:
        with self.conn, self.conn.cursor() as cur:
            cur.execute("SELECT value FROM state WHERE key=%s", (key,))
            row = cur.fetchone()
        return row[0] if row else None


class MaintenanceDB(MaintenanceStorageAdapter):
    """Persistent store for maintenance logs with environment-based backend."""

    def __init__(self, url: Path | str | None = None) -> None:
        path_or_url = url or os.environ.get("MAINTENANCE_DB") or env_config.MAINTENANCE_DB
        if not path_or_url:
            raise ValueError("MAINTENANCE_DB environment variable must be provided")
        path_or_url = str(path_or_url)

        prod = MENACE_MODE.lower() == "production"
        db_path: Path | None = None
        if "://" not in path_or_url:
            db_path = Path(path_or_url)
            if prod:
                raise RuntimeError("Database URL required for production")
            try:
                ensure_directory(db_path.parent)
                safe_write(db_path, "", mode="a")
            except Exception as exc:
                raise RuntimeError(f"maintenance db path not writable: {exc}") from exc
            db_url = f"sqlite:///{db_path}"
        else:
            db_url = path_or_url

        if prod and db_url.startswith("sqlite"):
            raise RuntimeError("SQLite may not be used for MaintenanceDB in production")

        if create_engine is None:
            if prod and not db_url.startswith("sqlite"):
                raise RuntimeError("SQLAlchemy is required for MaintenanceDB in production")

            if db_path is not None and "//" not in db_url:
                self.path = db_path
            else:
                self.path = Path(db_url.split("///")[-1])
            conn = router.get_connection("maintenance")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS maintenance(task TEXT, status TEXT, details TEXT, ts TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT)"
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.commit()
            self.engine = None  # type: ignore
            self.meta = None  # type: ignore
            self.table = None  # type: ignore
            self.state_table = None  # type: ignore
            self._sqlite = True
        else:
            self.engine = create_engine(db_url)
            self.meta = MetaData()
            self.table = Table(
                "maintenance",
                self.meta,
                Column("task", String),
                Column("status", String),
                Column("details", Text),
                Column("ts", String),
            )
            self.state_table = Table(
                "state",
                self.meta,
                Column("key", String, primary_key=True),
                Column("value", Text),
            )
            self.meta.create_all(self.engine)
            self._sqlite = db_url.startswith("sqlite")
            if self._sqlite:
                self.path = db_path or Path(urlparse(db_url).path)

    def log(self, rec: MaintenanceRecord) -> None:
        if self._sqlite:
            try:
                conn = router.get_connection("maintenance")
                conn.execute(
                    "INSERT INTO maintenance(task, status, details, ts) VALUES (?,?,?,?)",
                    (rec.task, rec.status, rec.details, rec.ts),
                )
                conn.commit()
            except Exception as exc:  # pragma: no cover - sqlite failure
                logging.getLogger(__name__).error("sqlite log failed: %s", exc)
        else:
            try:
                with self.engine.begin() as conn:
                    conn.execute(
                        self.table.insert().values(
                            task=rec.task,
                            status=rec.status,
                            details=rec.details,
                            ts=rec.ts,
                        )
                    )
            except SQLAlchemyError as exc:
                logging.getLogger(__name__).error("failed to log maintenance record: %s", exc)
                raise

    def fetch(self) -> List[Tuple[str, str, str, str]]:
        if self._sqlite:
            try:
                conn = router.get_connection("maintenance")
                rows = conn.execute(
                    "SELECT task, status, details, ts FROM maintenance"
                ).fetchall()
            except Exception as exc:  # pragma: no cover - sqlite failure
                logging.getLogger(__name__).error("sqlite fetch failed: %s", exc)
                return []
            return [(r[0], r[1], r[2], r[3]) for r in rows]
        with self.engine.begin() as conn:
            rows = conn.execute(self.table.select()).fetchall()
            return [(row.task, row.status, row.details, row.ts) for row in rows]

    # simple key/value state helpers
    def set_state(self, key: str, value: str) -> None:
        if self._sqlite:
            try:
                conn = router.get_connection("maintenance")
                conn.execute(
                    "INSERT OR REPLACE INTO state(key,value) VALUES (?,?)",
                    (key, value),
                )
                conn.commit()
            except Exception as exc:  # pragma: no cover - sqlite failure
                logging.getLogger(__name__).error("sqlite set_state failed: %s", exc)
        else:
            try:
                with self.engine.begin() as conn:
                    conn.execute(
                        self.state_table.delete().where(self.state_table.c.key == key)
                    )
                    conn.execute(
                        self.state_table.insert().values(key=key, value=value)
                    )
            except Exception as exc:  # pragma: no cover - sql failure
                logging.getLogger(__name__).error("state set failed: %s", exc)

    def get_state(self, key: str) -> Optional[str]:
        if self._sqlite:
            try:
                conn = router.get_connection("maintenance")
                row = conn.execute(
                    "SELECT value FROM state WHERE key=?", (key,)
                ).fetchone()
                return row[0] if row else None
            except Exception as exc:  # pragma: no cover - sqlite failure
                logging.getLogger(__name__).error("sqlite get_state failed: %s", exc)
                return None
        try:
            with self.engine.begin() as conn:
                row = conn.execute(
                    self.state_table.select().where(self.state_table.c.key == key)
                ).fetchone()
                return row.value if row else None
        except Exception as exc:  # pragma: no cover - sql failure
            logging.getLogger(__name__).error("state get failed: %s", exc)
            return None


class CommunicationLogHandler:
    """Manage persistent communication logs with validation and cleanup."""

    def __init__(
        self,
        path: Path | str = COMM_LOG_PATH,
        *,
        retention_hours: float = COMM_LOG_RETENTION_HOURS,
        max_size_mb: float = COMM_LOG_MAX_SIZE_MB,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.path = Path(path)
        ensure_directory(self.path.parent)
        self.retention = timedelta(hours=retention_hours)
        self.max_size = max_size_mb
        self.on_error = on_error
        loguru_logger.add(
            str(self.path),
            rotation=f"{max_size_mb} MB",
            retention=f"{retention_hours} hours",
            serialize=True,
        )

    def _notify_error(self, msg: str) -> None:
        if self.on_error:
            try:
                self.on_error(msg)
            except Exception:
                logger.exception("error callback failed")

    def load(self) -> List[dict]:
        if not self.path.exists():
            return []
        entries: List[dict] = []
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        obj = CommLogEntry.parse_raw(line)
                        entries.append(obj.dict())
                    except ValidationError as exc:
                        self._notify_error(f"invalid log entry: {exc}")
        except Exception as exc:
            logger.warning("failed loading communication log: %s", exc)
            self._notify_error(f"communication log load failed: {exc}")
        return entries

    def save(self, entries: List[dict]) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                for entry in entries:
                    fh.write(CommLogEntry(**entry).json() + "\n")
        except Exception as exc:  # pragma: no cover - filesystem issues
            logger.error("failed writing communication log: %s", exc, exc_info=True)
            self._notify_error(f"communication log save failed: {exc}")
            raise FileOperationError(str(exc)) from exc

    def append(self, message: str) -> None:
        message = _validate_message(message)
        entry = CommLogEntry(timestamp=SESSION_NOW.isoformat(), message=message)
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(entry.json() + "\n")
        except Exception as exc:
            logger.error("failed appending communication log: %s", exc, exc_info=True)
            self._notify_error(f"communication log append failed: {exc}")

    def cleanup(self) -> None:
        entries = self.load()
        fresh = [e for e in entries if not entry_expired(e.get("timestamp", ""), self.retention)]
        if len(fresh) != len(entries):
            self.save(fresh)


def entry_expired(ts: str, threshold: timedelta = timedelta(hours=COMM_LOG_RETENTION_HOURS)) -> bool:
    """Return True if timestamp is older than the provided threshold."""
    try:
        dt = datetime.fromisoformat(ts)
    except Exception:
        return True
    return datetime.utcnow() - dt > threshold


class CommunicationMaintenanceBot(AdminBotBase):
    """Manage updates, patches and resource allocation for communication bots."""

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _ensure_repo(self, repo_path: Path) -> Repo | None:
        """Return a valid Repo object, initialising if necessary."""
        try:
            repo = Repo(str(repo_path))
        except (InvalidGitRepositoryError, NoSuchPathError) as exc:
            self.logger.warning("invalid repo %s: %s", repo_path, exc)
            try:
                ensure_directory(repo_path)
                repo = Repo.init(str(repo_path))
                # simple command to validate repository
                repo.git.rev_parse("--git-dir")
                self.logger.info("initialized new repository at %s", repo_path)
            except GitCommandError as exc2:
                self.logger.error("failed initializing repo at %s: %s", repo_path, exc2)
                return None
            except Exception as exc2:
                self.logger.error("repository validation failed at %s: %s", repo_path, exc2)
                return None
        except GitCommandError as exc:
            self.logger.error("git error opening repo %s: %s", repo_path, exc)
            raise
        return repo

    def _parse_bot_urls(self, spec: str) -> Dict[str, str]:
        """Parse comma separated bot URL specification."""
        urls: Dict[str, str] = {}
        for item in spec.split(','):
            item = item.strip()
            if not item:
                continue
            if '=' in item:
                name, url = item.split('=', 1)
            else:
                url = item
                name = urlparse(url).hostname or f"bot{len(urls)}"
            url = url.strip()
            name = name.strip()
            if not url.startswith(('http://', 'https://')):
                self.logger.warning("invalid bot url %s", url)
                continue
            urls[name] = url
        return urls

    def _parse_webhook_urls(self, spec: Iterable[str] | str) -> List[str]:
        """Parse webhook URL specification into a list."""
        if isinstance(spec, str):
            parts = [p.strip() for p in spec.split(',') if p.strip()]
        else:
            parts = [str(p).strip() for p in spec if str(p).strip()]
        urls: List[str] = []
        for url in parts:
            if not url.startswith(('http://', 'https://')):
                self.logger.warning("invalid webhook url %s", url)
                continue
            urls.append(url)
        return urls

    def _create_scheduler(self, broker: str | None):
        """Return Celery app or a lightweight scheduler for development."""
        if Celery and broker:
            return Celery(
                "maintenance",
                broker=broker,
                backend=os.getenv("CELERY_RESULT_BACKEND", "rpc://"),
            )
        if Celery and MENACE_MODE.lower() == "production":
            raise RuntimeError("CELERY_BROKER_URL must be configured in production")
        if MENACE_MODE.lower() == "production":
            self.logger.warning("Celery unavailable in production, using fallback scheduler")
        else:
            self.logger.warning("Celery unavailable, falling back to simple scheduler")
        import time
        from threading import Event, Thread

        class _SimpleScheduler:
            def __init__(self) -> None:
                self.tasks: list[tuple[Callable[[], float] | float, Callable[[], None]]] = []
                self.stop = Event()
                self.thread: Thread | None = None
                self.logger = logging.getLogger("CommMaintenanceBot.Scheduler")
                self.poll_interval = SCHEDULER_POLL_INTERVAL

            def task(self, func: Callable) -> Callable:
                return func

            def add_periodic_task(
                self, interval: Callable[[], float] | float, func: Callable[[], None]
            ) -> None:
                self.tasks.append((interval, func))
                if not self.thread:
                    self.thread = Thread(target=self._run, daemon=True)
                    self.thread.start()

            def _run(self) -> None:
                next_runs: dict[int, float] = {i: time.time() for i in range(len(self.tasks))}
                while not self.stop.is_set():
                    if KILL_SWITCH_PATH and Path(KILL_SWITCH_PATH).exists():
                        self.logger.error("kill switch activated", extra={"tag": "kill"})
                        self.stop.set()
                        break
                    now = time.time()
                    for idx, (interval, fn) in enumerate(self.tasks):
                        if now >= next_runs[idx]:
                            try:
                                fn()
                            except BaseException:
                                if not self.stop.is_set():
                                    self.logger.exception("task failed")
                                else:
                                    raise
                            iv = interval() if callable(interval) else interval
                            next_runs[idx] = now + iv
                    time.sleep(self.poll_interval)

            def shutdown(self) -> None:
                self.stop.set()
                if self.thread:
                    self.thread.join(timeout=0)

        scheduler = _SimpleScheduler()
        atexit.register(scheduler.shutdown)
        return scheduler

    def __init__(
        self,
        db: MaintenanceStorageAdapter | None = None,
        error_bot: ErrorBot | None = None,
        repo_path: Path | str | None = None,
        broker: str | None = None,
        db_router: DBRouter | None = None,
        *,
        event_bus: Optional[UnifiedEventBus] = None,
        memory_mgr: MenaceMemoryManager | None = None,
        context_builder: "ContextBuilder",
        logger: logging.Logger | None = None,
        webhook_urls: Optional[Iterable[str] | str] = None,
        msg_retry_attempts: Optional[int] = None,
        msg_retry_delay: Optional[float] = None,
        msg_timeout: Optional[float] = None,
        msg_max_length: Optional[int] = None,
        comm_store: CommunicationLogHandler | None = None,
        config: MaintenanceBotConfig | None = None,
        admin_tokens: Optional[Iterable[str] | str] = None,
    ) -> None:
        super().__init__(db_router=db_router)
        self.logger = configure_logger(
            logger or logging.getLogger("CommMaintenanceBot"),
            level=os.getenv("MAINTENANCE_LOG_LEVEL"),
        )
        self.config = config or MaintenanceBotConfig()
        _validate_config(self.config)
        if db is not None:
            self.db = db
        else:
            url = os.getenv("MAINTENANCE_DB_URL")
            if url:
                try:
                    self.db = PostgresMaintenanceDB(url)
                except Exception as exc:
                    self.logger.warning(
                        "PostgreSQL maintenance DB unavailable: %s", exc
                    )
                    self.db = SQLiteMaintenanceDB()
            else:
                self.db = SQLiteMaintenanceDB()
        if context_builder is None:
            raise ValueError("context_builder is required")
        self.context_builder = context_builder
        self.error_bot = error_bot or ErrorBot(
            ErrorDB(self.config.error_db_path), context_builder=self.context_builder
        )
        repo_path = Path(repo_path or os.getenv("MAINTENANCE_REPO_PATH", "."))
        broker = broker or os.getenv("CELERY_BROKER_URL")
        if Celery and not broker and MENACE_MODE.lower() == "production":
            raise RuntimeError("CELERY_BROKER_URL must be configured in production")
        if not Celery and MENACE_MODE.lower() == "production" and not broker:
            self.logger.warning("Running without Celery broker in production mode")
        self.repo = self._ensure_repo(repo_path)
        self.event_bus = event_bus
        self.memory_mgr = memory_mgr
        self.comm_store = comm_store or CommunicationLogHandler(
            self.config.comm_log_path,
            max_size_mb=self.config.comm_log_max_size_mb,
            on_error=self.escalate_error,
        )
        self.lock = TaskLock()
        self.webhook_urls = self._parse_webhook_urls(
            webhook_urls if webhook_urls is not None else DIAGNOSTICS_WEBHOOKS
        )
        if not self.webhook_urls:
            raise ValueError("Valid MAINTENANCE_DISCORD_WEBHOOK(S) must be configured")
        self.webhook_stats: Dict[str, Dict[str, int]] = {
            url: {"success": 0, "failure": 0} for url in self.webhook_urls
        }
        self.fallback_webhook_urls = self._parse_webhook_urls(FALLBACK_WEBHOOKS)
        self.kill_switch_path = KILL_SWITCH_PATH
        self.ping_max_duration = PING_MAX_DURATION
        self.message_max_duration = MESSAGE_MAX_DURATION
        self.msg_retry_attempts = (
            msg_retry_attempts if msg_retry_attempts is not None else MESSAGE_RETRY_ATTEMPTS
        )
        self.msg_retry_delay = (
            msg_retry_delay if msg_retry_delay is not None else MESSAGE_RETRY_DELAY
        )
        self.msg_timeout = msg_timeout if msg_timeout is not None else MESSAGE_TIMEOUT
        self.msg_max_length = msg_max_length if msg_max_length is not None else MESSAGE_MAX_LENGTH
        _validate_positive("msg_retry_attempts", self.msg_retry_attempts)
        _validate_positive("msg_retry_delay", self.msg_retry_delay)
        _validate_positive("msg_timeout", self.msg_timeout)
        _validate_positive("msg_max_length", self.msg_max_length)
        self.queue_file = Path(self.config.message_queue)
        self.flush_queue()
        self.comm_store.cleanup()
        self.notify_lock = Lock()
        self.bot_urls = self._parse_bot_urls(BOT_URLS)
        self.fail_counts: Dict[str, int] = {}
        self.cluster_data: Dict[str, object] = {}
        self.fetch_cluster_data()
        self.ping_interval = PING_INTERVAL
        self.heartbeat_interval = HEARTBEAT_INTERVAL
        _validate_positive("ping_interval", self.ping_interval)
        _validate_positive("heartbeat_interval", self.heartbeat_interval)
        self.bot_name = os.getenv("MAINTENANCE_BOT_NAME", "maintenance_bot")
        self.templates = load_templates()
        if admin_tokens is None:
            tokens = ADMIN_TOKENS
        elif isinstance(admin_tokens, str):
            tokens = [t.strip() for t in admin_tokens.split(',') if t.strip()]
        else:
            tokens = [str(t).strip() for t in admin_tokens if str(t).strip()]
        self.admin_tokens = tokens
        self.last_command_times: Dict[str, float] = {}
        self.command_map: Dict[str, Callable[..., object]] = {
            "check_updates": self.check_updates,
            "optimise": self.optimise_performance,
            "heartbeat": self.heartbeat,
            "ping": self.ping_bots,
        }
        self.last_deployment_event: object | None = None
        self.last_memory_entry: MemoryEntry | None = None
        try:
            val = self.db.get_state("last_deployment_event")
            if val:
                self.last_deployment_event = DeploymentEvent(**json.loads(val))
            val = self.db.get_state("last_memory_entry")
            if val:
                self.last_memory_entry = MemoryEntry(**json.loads(val))
        except Exception as exc:  # pragma: no cover - corrupted state
            self.logger.warning("failed to load previous state: %s", exc)
        self.app = self._create_scheduler(broker)
        if self.event_bus:
            try:
                self.event_bus.subscribe("deployments:new", self._on_deployment_event)
            except Exception as exc:
                self.logger.exception("event bus subscription failed: %s", exc)
        if self.memory_mgr:
            try:
                self.memory_mgr.subscribe(self._on_memory_entry)
            except Exception as exc:
                self.logger.exception("memory subscription failed: %s", exc)

    def query(self, term: str) -> str:
        """Query DB router and context builder for ``term`` and return context."""
        try:
            self.db_router.query_all(term)
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.warning("db query failed: %s", exc)
        snippet = ""
        metadata: Dict[str, object] = {}
        try:
            result = self.context_builder.query(term, return_metadata=True)
            if isinstance(result, dict):
                snips = result.get("snippets") or result.get("context") or []
                if isinstance(snips, list):
                    snippet = snips[0] if snips else ""
                else:
                    snippet = str(snips)
                metadata = result.get("metadata") or {}
        except Exception as exc:  # pragma: no cover - retrieval failures
            self.logger.warning("context retrieval failed: %s", exc)
        parts = []
        if snippet:
            parts.append(f"snippet: {snippet}")
        if metadata:
            parts.append(f"metadata: {metadata}")
        return " | ".join(parts)

    def escalate_error(self, message: str, severity: Severity = Severity.CRITICAL) -> None:
        """Handle failure by logging, auditing and notifying."""
        context = self.query(message)
        enriched = f"{message} | {context}" if context else message
        self.logger.error(
            "escalation: %s",
            enriched,
            extra={"tag": severity.value, "bot_name": self.bot_name},
        )
        try:
            audit_log_event(
                "maintenance_error",
                {"bot": self.bot_name, "message": enriched, "severity": severity.value},
            )
        except Exception:
            self.logger.exception("audit logging failed")
        try:
            self.error_bot.handle_error(enriched)
        except Exception:
            self.logger.exception("error escalation handling failed")
        try:
            self.notify_critical(f"ESCALATION: {enriched}")
        except Exception:
            self.logger.exception("escalation notification failed")

    def set_log_level(self, level: str | int) -> None:
        """Dynamically update logger verbosity."""
        if isinstance(level, str):
            level = getattr(logging, level.upper(), self.logger.level)
        self.logger.setLevel(level)

    def recalibrate_heartbeat_interval(self, load: float | None = None) -> None:
        """Adjust ``heartbeat_interval`` based on system load."""
        try:
            if load is None:
                if 'psutil' in globals() and psutil is not None:
                    cpu = psutil.cpu_percent()
                    mem = psutil.virtual_memory().percent
                    load = max(cpu, mem) / 100.0
                else:
                    load = 0.0
            base = HEARTBEAT_INTERVAL
            new_interval = self.heartbeat_interval
            if load > HEARTBEAT_LOAD_HIGH:
                new_interval = min(self.heartbeat_interval * 2, base * HEARTBEAT_MAX_FACTOR)
            elif load < HEARTBEAT_LOAD_LOW and self.heartbeat_interval > base:
                new_interval = max(base * HEARTBEAT_MIN_FACTOR, self.heartbeat_interval / 2)
            if new_interval != self.heartbeat_interval:
                self.heartbeat_interval = new_interval
                self.logger.info(
                    "heartbeat interval adjusted to %s",
                    new_interval,
                    extra={"tag": "recalibrate", "bot_name": self.bot_name},
                )
        except Exception as exc:  # pragma: no cover - metrics failure
            self.logger.warning("recalibration failed: %s", exc)

    def _run_with_timeout(self, func: Callable[[], None], timeout: float, name: str) -> None:
        """Execute ``func`` with a timeout and escalate if it exceeds ``timeout``."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(func)
            try:
                fut.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                self.escalate_error(f"{name} timed out after {timeout}s")
                raise

    def report_metric(self, name: str, value: float) -> None:
        """Report metrics to the event bus or via Discord."""
        payload = {"metric": name, "value": value, "ts": datetime.utcnow().isoformat(), "bot": self.bot_name}
        if self.event_bus:
            try:
                self.event_bus.publish("metrics:new", payload)
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.warning("metric publish failed: %s", exc)
        else:
            self.notify(json.dumps(payload))

    def _run_task(self, func: Callable[[], T], name: str, timeout: float = TASK_TIME_LIMIT) -> T:
        """Run ``func`` with retries, timeout and alerting."""
        backoff = RUN_TASK_BACKOFF
        for attempt in range(RUN_TASK_ATTEMPTS):
            try:
                result: T | None = None
                def call() -> T:
                    nonlocal result
                    result = func()
                    return result

                self._run_with_timeout(call, timeout, name)
                self.report_metric(f"task_{name}_success", 1.0)
                return result  # type: ignore[return-value]
            except Exception as exc:
                self.logger.warning(
                    "%s attempt %s failed: %s",
                    name,
                    attempt + 1,
                    exc,
                    extra={"task": name, "bot_name": self.bot_name},
                )
                if attempt == RUN_TASK_ATTEMPTS - 1:
                    self.escalate_error(f"{name} persistent failure: {exc}")
                    self.report_metric(f"task_{name}_failure", 1.0)
                    raise
                time.sleep(backoff)
                backoff *= 2

    # ------------------------------------------------------------------
    # Command execution helpers
    # ------------------------------------------------------------------

    def _check_cooldown(self, name: str) -> None:
        last = self.last_command_times.get(name, 0)
        now = time.time()
        if now - last < COMMAND_COOLDOWN:
            raise RateLimitError(COMMAND_COOLDOWN - (now - last))
        self.last_command_times[name] = now

    def execute_command(self, token: str, command: str, *args, **kwargs) -> object:
        """Execute a registered command if authorised and not rate limited."""
        if token not in self.admin_tokens:
            self.logger.error("unauthorised command token", extra={"tag": "auth"})
            raise PermissionError("invalid token")
        command = str(command).strip().lower()
        if command not in self.command_map:
            self.logger.error("unknown command %s", command, extra={"tag": "cmd"})
            raise ValueError(f"unknown command {command}")
        self._check_cooldown(command)
        func = self.command_map[command]
        return func(*args, **kwargs)

    def check_updates(self) -> None:
        """Check for uncommitted changes and log status."""
        context = self.query("update")

        def _impl() -> None:
            if not self.repo:
                self.db.log(
                    MaintenanceRecord(
                        task=TaskType.UPDATE_CHECK.value,
                        status="no_repo",
                        details="",
                    )
                )
                return
            dirty = self.repo.is_dirty()
            status = "dirty" if dirty else "clean"
            self.db.log(
                MaintenanceRecord(
                    task=TaskType.UPDATE_CHECK.value,
                    status=status,
                    details="",
                )
            )
            msg = self.templates["update_status"].safe_substitute(
                bot=self.bot_name, status=status
            )
            if context:
                msg = f"{msg} | {context}"
            self.notify(msg)

        self._run_task(_impl, "check_updates")

    def apply_hotfix(
        self,
        description: str,
        fix: Callable[[], None],
        *,
        commit: bool | None = None,
        soft_rollback: bool = False,
        diff_log: bool = False,
    ) -> None:
        """Apply hotfix and optionally commit the changes."""
        context = self.query(description)

        def _impl() -> None:
            commit_flag = commit
            if commit_flag is None:
                commit_flag = MENACE_MODE.lower() != "production"
            rollback_point = None
            if self.repo:
                rollback_point = self.repo.head.commit.hexsha
            exc: Exception | None = None
            try:
                fix()
                status = "applied"
                if self.repo and commit_flag:
                    self.repo.git.add(A=True)
                    self.repo.index.commit(f"hotfix: {description}")
                elif not self.repo:
                    self.logger.warning("no git repository available; hotfix not committed")
                    status = "logged"
                elif not commit_flag:
                    status = "logged"
            except Exception as e:  # pragma: no cover - runtime failures
                exc = e
                status = "failed"
                self.error_bot.handle_error(str(e))
                self.logger.exception("hotfix application failed: %s", e)
            finally:
                if exc and self.repo and rollback_point:
                    try:
                        if diff_log:
                            diff = self.repo.git.diff(rollback_point)
                            self.logger.warning("rollback diff:\n%s", diff)
                        reset_arg = "--soft" if soft_rollback else "--hard"
                        self.repo.git.reset(reset_arg, rollback_point)
                        self.logger.info("rolled back hotfix to %s", rollback_point)
                    except Exception as rex:
                        self.logger.error("rollback failed: %s", rex)
                        self.db.log(
                            MaintenanceRecord(
                                task=TaskType.HOTFIX.value,
                                status="rollback_failed",
                                details=str(rex),
                            )
                        )
            self.db.log(
                MaintenanceRecord(
                    task=TaskType.HOTFIX.value,
                    status=status,
                    details=description if not exc else str(exc),
                )
            )
            if not exc:
                self.error_bot.db.log_discrepancy(description)
                msg = self.templates["hotfix_applied"].safe_substitute(
                    bot=self.bot_name, desc=description
                )
                if context:
                    msg = f"{msg} | {context}"
                self.notify(msg)
            if exc:
                raise exc

        self._run_task(_impl, "apply_hotfix")

    def optimise_performance(self) -> None:
        """Perform lightweight git maintenance to keep repository efficient."""
        context = self.query("optimise")

        def _impl() -> None:
            if not self.repo:
                self.db.log(
                    MaintenanceRecord(
                        task=TaskType.OPTIMISE.value,
                        status="no_repo",
                        details="",
                    )
                )
                return
            try:
                self.repo.git.gc(auto=True)
                summary = "git gc run"
            except Exception as exc:  # pragma: no cover - git failures
                summary = f"gc failed: {exc}"
            self.db.log(
                MaintenanceRecord(
                    task=TaskType.OPTIMISE.value,
                    status="done",
                    details=summary,
                )
            )
            msg = self.templates["optimise_summary"].safe_substitute(
                bot=self.bot_name, summary=summary
            )
            if context:
                msg = f"{msg} | {context}"
            self.notify(msg)

        self._run_task(_impl, "optimise_performance")

    def adjust_resources(self, metrics: Dict[str, ResourceMetrics]) -> List[Tuple[str, bool]]:
        """Delegate to ResourceAllocationBot for efficiency-based allocation."""
        self.query("allocation")

        def _impl() -> List[Tuple[str, bool]]:
            alloc_bot = _alloc_bot_instance()
            if alloc_bot is None:
                raise RuntimeError("ResourceAllocationBot missing")
            actions = alloc_bot.allocate(metrics)
            self.db.log(
                MaintenanceRecord(
                    task=TaskType.ALLOCATION.value,
                    status="updated",
                    details=str(actions),
                )
            )
            msg = self.templates["allocation_updated"].safe_substitute(
                bot=self.bot_name, actions=actions
            )
            self.notify(msg)
            return actions

        return self._run_task(_impl, "adjust_resources")

    # ------------------------------------------------------------------
    # Communication monitoring helpers
    # ------------------------------------------------------------------

    def fetch_cluster_data(self) -> Dict[str, object]:
        """Retrieve latest cluster information from external source."""
        url = os.getenv("MAINTENANCE_CLUSTER_INFO_URL")
        if url and requests is not None:
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    self.cluster_data = resp.json()  # type: ignore[assignment]
                    return self.cluster_data
                self.logger.warning(
                    "cluster info request failed: %s", resp.status_code
                )
            except Exception as exc:  # pragma: no cover - network failure
                self.logger.warning("cluster info fetch failed: %s", exc)
        self.cluster_data = {}
        return self.cluster_data

    def evaluate_status(self) -> Dict[str, float]:
        """Evaluate recent communication logs and return metrics."""
        entries = self.comm_store.load()
        recent = [
            e
            for e in entries
            if not entry_expired(e.get("timestamp", ""), timedelta(hours=1))
        ]
        total = len(recent)
        err = sum(
            1
            for e in recent
            if "error" in str(e.get("message", "")).lower()
        )
        return {
            "messages_last_hour": float(total),
            "error_rate": (err / total) if total else 0.0,
        }

    def monitor_communication(self) -> Dict[str, float]:
        """Monitor communication health and escalate if degraded."""
        self.fetch_cluster_data()
        metrics = self.evaluate_status()
        if metrics.get("error_rate", 0.0) > 0.1:
            self.escalate_error(
                f"High communication error rate: {metrics['error_rate']:.2%}",
                Severity.HIGH,
            )
        return metrics

    def generate_maintenance_report(self) -> str:
        """Build a maintenance report using metrics and cluster info."""
        metrics = self.evaluate_status()
        lines = [f"Communication summary for {self.bot_name}:"]
        lines.append(f"Messages last hour: {int(metrics['messages_last_hour'])}")
        lines.append(f"Error rate: {metrics['error_rate']:.2%}")
        if self.cluster_data:
            details = ", ".join(f"{k}={v}" for k, v in self.cluster_data.items())
            lines.append(f"Cluster info: {details}")
        issues: List[str] = []
        if metrics["error_rate"] > 0.1:
            issues.append("High error rate detected")
        if not self.cluster_data:
            issues.append("Cluster info unavailable")
        if issues:
            lines.append("Issues: " + "; ".join(issues))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Bot ping helpers
    # ------------------------------------------------------------------

    def _ping_bot(self, name: str, url: str) -> bool:
        """Ping a single bot with retry and validation."""
        if requests is None:
            self.logger.warning(
                "requests library missing, cannot ping %s",
                name,
                extra={"bot_name": self.bot_name},
            )
            return False
        delay = PING_RETRY_DELAY
        start = time.time()
        for attempt in range(1, PING_RETRY_ATTEMPTS + 1):
            if self.kill_switch_path and Path(self.kill_switch_path).exists():
                self.logger.error(
                    "kill switch activated during ping",
                    extra={"tag": "kill", "bot_name": self.bot_name},
                )
                return False
            if time.time() - start > self.ping_max_duration:
                self.logger.error(
                    "ping %s exceeded max duration",
                    name,
                    extra={"bot_name": self.bot_name},
                )
                return False
            try:
                resp = requests.get(url, timeout=PING_TIMEOUT)
                if resp.status_code != 200:
                    self.logger.warning(
                        "ping %s attempt %s failed status %s",
                        name,
                        attempt,
                        resp.status_code,
                        extra={"bot_name": self.bot_name},
                    )
                else:
                    try:
                        data = resp.json()
                        val = data.get(PING_EXPECT_KEY)
                        expected = PING_EXPECT_VALUE
                        match = False
                        if isinstance(expected, bool):
                            match = bool(val) is expected
                        else:
                            match = str(val).lower() == str(expected).lower()
                        if match:
                            self.logger.info(
                                "ping %s succeeded",
                                name,
                                extra={"bot_name": self.bot_name},
                            )
                            return True
                        self.logger.warning("ping %s invalid response %s", name, resp.text)
                    except Exception:
                        self.logger.info(
                            "ping %s succeeded (no json)",
                            name,
                            extra={"bot_name": self.bot_name},
                        )
                        return True
            except Exception as exc:
                self.logger.warning(
                    "ping %s attempt %s error: %s",
                    name,
                    attempt,
                    exc,
                    extra={"bot_name": self.bot_name},
                )
            if attempt < PING_RETRY_ATTEMPTS:
                time.sleep(delay)
                delay *= 2
        return False

    def ping_bots(self) -> Dict[str, bool]:
        """Ping all configured bots concurrently."""
        results: Dict[str, bool] = {}
        if not self.bot_urls:
            return results
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.bot_urls)) as ex:
            future_map = {
                ex.submit(self._ping_bot, name, url): name
                for name, url in self.bot_urls.items()
            }
            for fut in concurrent.futures.as_completed(future_map):
                name = future_map[fut]
                ok = fut.result()
                results[name] = ok
                if not ok:
                    self.fail_counts[name] = self.fail_counts.get(name, 0) + 1
                    self.escalate_error(f"Ping to {name} failed", Severity.HIGH)
                    if self.fail_counts[name] >= PING_FAILURE_LIMIT:
                        self.logger.error(
                            "bot %s removed after %s failed pings",
                            name,
                            self.fail_counts[name],
                        )
                        self.bot_urls.pop(name, None)
                else:
                    self.fail_counts[name] = 0
                self.report_metric(f"ping_{name}", 1.0 if ok else 0.0)
        return results

    # ------------------------------------------------------------------
    # Messaging helpers
    # ------------------------------------------------------------------

    def _split_message(self, content: str) -> List[str]:
        if not content:
            raise ValueError("message content required")
        content = content.replace("@", "@\u200b")
        max_len = self.msg_max_length
        if len(content) <= max_len:
            return [content]
        chunks: List[str] = []
        start = 0
        while start < len(content):
            chunks.append(content[start : start + max_len])
            start += max_len
        return chunks

    def _send_requests(self, url: str, content: str) -> tuple[bool, Optional[float]]:
        if requests is None:
            return False, None
        try:
            resp = requests.post(url, json={"content": content}, timeout=self.msg_timeout)
            if resp.status_code in {200, 204}:
                return True, None
            if resp.status_code == 429:
                retry_after = self.msg_retry_delay
                try:
                    retry_after = float(resp.json().get("retry_after", retry_after))
                except Exception:
                    self.logger.exception("failed parsing retry_after from discord response")
                return False, retry_after
            self.logger.error(
                "discord error %s: %s",
                resp.status_code,
                resp.text,
                extra={"bot_name": self.bot_name},
            )
        except requests.Timeout:
            self.logger.warning("discord request timed out", extra={"bot_name": self.bot_name})
        except requests.RequestException as exc:
            self.logger.warning("discord request failed: %s", exc, extra={"bot_name": self.bot_name})
        except Exception as exc:
            self.logger.warning("discord send failed: %s", exc, extra={"bot_name": self.bot_name})
        return False, None

    async def _send_aiohttp(self, url: str, content: str) -> tuple[bool, Optional[float]]:
        assert aiohttp is not None
        timeout = aiohttp.ClientTimeout(total=self.msg_timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json={"content": content}) as resp:
                    if resp.status in {200, 204}:
                        return True, None
                    if resp.status == 429:
                        retry_after = self.msg_retry_delay
                        try:
                            data = await resp.json()
                            retry_after = float(data.get("retry_after", retry_after))
                        except Exception:
                            self.logger.exception("failed parsing async retry_after from discord response")
                        return False, retry_after
                    text = await resp.text()
                    self.logger.error(
                        "discord error %s: %s",
                        resp.status,
                        text,
                        extra={"bot_name": self.bot_name},
                    )
        except asyncio.TimeoutError:
            self.logger.warning("discord request timed out", extra={"bot_name": self.bot_name})
        except Exception as exc:
            self.logger.warning("discord request failed: %s", exc, extra={"bot_name": self.bot_name})
        return False, None

    def _send_message(self, url: str, content: str) -> bool:
        delay = self.msg_retry_delay
        start = time.time()
        for attempt in range(1, self.msg_retry_attempts + 1):
            if self.kill_switch_path and Path(self.kill_switch_path).exists():
                self.logger.error("kill switch activated during message send", extra={"tag": "kill"})
                return False
            if time.time() - start > self.message_max_duration:
                self.logger.error("message send exceeded max duration")
                return False
            if aiohttp is not None:
                ok, retry_after = asyncio.run(self._send_aiohttp(url, content))
            else:
                ok, retry_after = self._send_requests(url, content)
            if ok:
                return True
            if retry_after is not None:
                time.sleep(retry_after)
                delay = retry_after
                continue
            time.sleep(delay)
            delay *= 2
        self.logger.error(
            "failed to send message after %s attempts", self.msg_retry_attempts
        )
        self.escalate_error("failed to send discord message")
        return False

    def _send_discord(self, content: str) -> bool:
        if not self.webhook_urls:
            return False
        segments = self._split_message(content)
        success = True
        for segment in segments:
            sent = False
            ordered = sorted(
                self.webhook_urls,
                key=lambda u: (
                    -self.webhook_stats[u]["success"],
                    self.webhook_stats[u]["failure"],
                ),
            )
            for url in ordered:
                if self._send_message(url, segment):
                    self.webhook_stats[url]["success"] += 1
                    sent = True
                    break
                else:
                    self.webhook_stats[url]["failure"] += 1
            success = success and sent
        return success

    def _queue_message(self, content: str) -> None:
        try:
            safe_write(self.queue_file, json.dumps({"content": content}) + "\n", mode="a")
        except Exception as exc:
            self.logger.error("failed queuing message: %s", exc, exc_info=True)

    def flush_queue(self) -> None:
        if not self.queue_file.exists():
            return
        lock = FileLock(str(self.queue_file) + ".lock", timeout=FILE_LOCK_TIMEOUT)
        try:
            with lock:
                lines = self.queue_file.read_text().splitlines()
        except Exception as exc:
            self.logger.error("failed reading queue: %s", exc, exc_info=True)
            return
        remaining: List[str] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                content = str(payload.get("content", ""))
            except Exception:
                continue
            if not self._send_discord(content):
                remaining.append(json.dumps({"content": content}))
        if remaining:
            try:
                safe_write(self.queue_file, "\n".join(remaining) + "\n")
            except Exception as exc:
                self.logger.error("failed writing queue: %s", exc, exc_info=True)
        else:
            try:
                with lock:
                    self.queue_file.unlink()
            except Exception:
                self.logger.exception("failed deleting queue file")

    def notify(self, content: str) -> None:
        content = _validate_message(content)
        with self.notify_lock:
            self.flush_queue()
            if not self._send_discord(content):
                self._queue_message(content)
            self.comm_store.append(content)
        self.logger.info(
            "notify: %s",
            content,
            extra={"bot_name": self.bot_name, "tag": "notify"},
        )

    def notify_critical(self, content: str) -> None:
        content = _validate_message(content)
        with self.notify_lock:
            self.flush_queue()
            sent = self._send_discord(content)
            if not sent and self.fallback_webhook_urls:
                for url in self.fallback_webhook_urls:
                    if self._send_message(url, content):
                        sent = True
                        break
            if not sent:
                self._queue_message(content)
            self.comm_store.append(content)
        self.logger.info(
            "critical_notify: %s",
            content,
            extra={"bot_name": self.bot_name, "tag": "critical"},
        )

    def heartbeat(self) -> None:
        msg = self.templates["heartbeat"].safe_substitute(
            bot=self.bot_name, ts=datetime.utcnow().isoformat()
        )
        self.notify(msg)
        self.recalibrate_heartbeat_interval()

    def schedule(self) -> None:
        """Set up periodic maintenance tasks if Celery is available."""
        if not Celery:
            self.logger.warning("Celery unavailable, skipping schedule")
            return
        lock = self.lock

        @self.app.task(
            bind=True,
            autoretry_for=(Exception,),
            retry_backoff=True,
            retry_jitter=True,
            default_retry_delay=TASK_RETRY_DELAY,
            soft_time_limit=TASK_TIME_LIMIT,
        )
        def _check_updates_task(_self: object) -> None:
            if lock.acquire("maintenance_check_updates"):
                try:
                    self.check_updates()
                finally:
                    lock.release("maintenance_check_updates")

        @self.app.task(
            bind=True,
            autoretry_for=(Exception,),
            retry_backoff=True,
            retry_jitter=True,
            default_retry_delay=TASK_RETRY_DELAY,
            soft_time_limit=TASK_TIME_LIMIT,
        )
        def _optimise_task(_self: object) -> None:
            if lock.acquire("maintenance_optimise"):
                try:
                    self.optimise_performance()
                finally:
                    lock.release("maintenance_optimise")

        @self.app.task(
            bind=True,
            autoretry_for=(Exception,),
            retry_backoff=True,
            retry_jitter=True,
            default_retry_delay=TASK_RETRY_DELAY,
            soft_time_limit=TASK_TIME_LIMIT,
        )
        def _heartbeat_task(_self: object) -> None:
            if lock.acquire("maintenance_heartbeat"):
                try:
                    self.heartbeat()
                finally:
                    lock.release("maintenance_heartbeat")

        @self.app.task(
            bind=True,
            autoretry_for=(Exception,),
            retry_backoff=True,
            retry_jitter=True,
            default_retry_delay=TASK_RETRY_DELAY,
            soft_time_limit=TASK_TIME_LIMIT,
        )
        def _ping_task(_self: object) -> None:
            if lock.acquire("maintenance_ping"):
                try:
                    self.ping_bots()
                finally:
                    lock.release("maintenance_ping")

        self.app.add_periodic_task(CHECK_INTERVAL, _check_updates_task.s())
        self.app.add_periodic_task(OPTIMISE_INTERVAL, _optimise_task.s())
        if Celery is not None and isinstance(self.app, Celery):
            self.app.add_periodic_task(self.heartbeat_interval, _heartbeat_task.s())
            if self.bot_urls:
                self.app.add_periodic_task(self.ping_interval, _ping_task.s())
        else:
            self.app.add_periodic_task(lambda: self.heartbeat_interval, _heartbeat_task)
            if self.bot_urls:
                self.app.add_periodic_task(lambda: self.ping_interval, _ping_task)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_deployment_event(self, topic: str, payload: object) -> None:
        """Store deployment events received via the event bus."""
        if not isinstance(payload, dict):
            raise TypeError("deployment event payload must be a dict")
        if set(payload.keys()) != {"id"}:
            raise ValueError(f"malformed deployment payload: {payload}")
        try:
            self.last_deployment_event = DeploymentEvent(**payload)
            self.db.set_state("last_deployment_event", json.dumps(payload))
        except Exception as exc:
            self.logger.error("invalid deployment event payload: %s", exc)
            raise

    def _on_memory_entry(self, entry: MemoryEntry) -> None:
        """Record memory entries tagged for maintenance review."""
        if not isinstance(entry, MemoryEntry):
            raise TypeError("memory entry must be a MemoryEntry instance")
        required = {"key", "data", "version", "tags"}
        if not required.issubset(vars(entry).keys()):
            raise ValueError("malformed memory entry")
        tags = entry.tags
        tag_list: List[str] = []
        if isinstance(tags, str):
            tag_list = [t.strip().lower() for t in tags.split(",") if t.strip()]
        else:
            try:
                tag_list = [str(t).lower() for t in tags]
            except Exception:
                tag_list = []
        if "maintenance" in tag_list:
            self.last_memory_entry = entry
            try:
                self.db.set_state("last_memory_entry", json.dumps(asdict(entry)))
            except Exception as exc:  # pragma: no cover - state persistence fail
                self.logger.warning("failed to persist memory entry: %s", exc)


__all__ = [
    "TaskType",
    "Severity",
    "DeploymentEvent",
    "MaintenanceRecord",
    "MaintenanceBotConfig",
    "TaskLock",
    "RateLimitError",
    "MaintenanceStorageAdapter",
    "SQLiteMaintenanceDB",
    "PostgresMaintenanceDB",
    "MaintenanceDB",
    "CommunicationLogHandler",
    "entry_expired",
    "CommunicationMaintenanceBot",
]
