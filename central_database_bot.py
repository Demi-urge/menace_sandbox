from __future__ import annotations

from .coding_bot_interface import self_coding_managed
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Callable, TypeVar
import uuid

from .logging_utils import set_correlation_id
from .resilience import (
    CircuitBreaker,
    CircuitOpenError,
    RetryError,
    ResilienceError,
    retry_with_backoff,
)

from pydantic import BaseModel
from .db_router import DBRouter
from .admin_bot_base import AdminBotBase
try:
    from sqlalchemy import MetaData, Table, Column, Integer, Text, create_engine, select, text as sqtext  # type: ignore
    from sqlalchemy.engine import Engine  # type: ignore
    from sqlalchemy.exc import SQLAlchemyError  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    MetaData = Table = Column = Integer = Text = create_engine = select = sqtext = Engine = SQLAlchemyError = None  # type: ignore

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None  # type: ignore

try:
    from redlock import Redlock  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Redlock = None  # type: ignore


class Proposal(BaseModel):
    """Incoming write intent."""

    operation: str
    target_table: str
    payload: Dict[str, Any]
    origin_bot_id: str
    priority: int = 1
    ts: str = datetime.utcnow().isoformat()


@dataclass
class Result:
    proposal: Proposal
    status: str
    detail: str = ""


class CentralDatabaseError(ResilienceError):
    """Base class for CentralDatabaseBot errors."""


class RedisUnavailableError(CentralDatabaseError):
    """Raised when Redis operations fail or circuit is open."""


@self_coding_managed
class CentralDatabaseBot(AdminBotBase):
    """Serialize schema-changing operations via a FIFO queue."""

    def __init__(
        self,
        db_url: str = "sqlite:///:memory:",
        redis_url: str | None = None,
        stream: str = "menace.db_queue",
        results_stream: str = "menace.db_results",
        lock_key: str = "menace.db_lock",
        db_router: "DBRouter" | None = None,
        fallback_local: bool = True,
    ) -> None:
        super().__init__(db_router=db_router)
        self.engine: Engine = create_engine(db_url)
        self.meta = MetaData()
        self.meta.reflect(bind=self.engine)
        self.redis_url = redis_url
        self.stream = stream
        self.results_stream = results_stream
        self.lock_key = lock_key
        self.fallback_local = fallback_local
        self.logger = logging.getLogger("CentralDatabaseBot")

        if redis and redis_url:
            try:
                self.client = retry_with_backoff(
                    lambda: redis.from_url(redis_url),
                    attempts=3,
                    delay=1.0,
                    logger=self.logger,
                )
            except ResilienceError as exc:
                self.logger.exception("redis connection failed")
                if not self.fallback_local:
                    raise RedisUnavailableError(str(exc)) from exc
                self.client = None
                self.logger.warning("using local queue due to redis failure")
        else:
            self.client = None
        self.queue: list[Proposal] = []
        self.results: list[Result] = []
        self.lock = Redlock([redis_url]) if Redlock and redis_url else None
        self._prepare_invalid_table()
        self.circuit = CircuitBreaker(max_failures=3, reset_timeout=30.0)

    # ------------------------------------------------------------------
    def _prepare_invalid_table(self) -> None:
        if "invalid_proposals" not in self.meta.tables:
            self.invalid_table = Table(
                "invalid_proposals",
                self.meta,
                Column("id", Integer, primary_key=True),
                Column("proposal", Text),
            )
            self.meta.create_all(self.engine)
        else:
            self.invalid_table = self.meta.tables["invalid_proposals"]

    T = TypeVar("T")

    def _redis_op(self, func: Callable[[], T]) -> T:
        if not self.client:
            raise RedisUnavailableError("redis client missing")
        try:
            return self.circuit.call(
                lambda: retry_with_backoff(func, attempts=3, delay=0.5, logger=self.logger)
            )
        except CircuitOpenError as exc:
            raise RedisUnavailableError("redis circuit open") from exc
        except RetryError as exc:
            raise RedisUnavailableError(str(exc)) from exc

    def enqueue(self, proposal: Proposal) -> None:
        """Push a proposal onto the queue."""
        if self.client:
            try:
                self._redis_op(lambda: self.client.xadd(self.stream, proposal.dict()))
                return
            except RedisUnavailableError:
                self.logger.exception("enqueue to redis failed")
                if self.fallback_local:
                    self.logger.warning("falling back to local queue")
                else:
                    raise
        self.queue.append(proposal)

    # ------------------------------------------------------------------
    def _fetch_one(self) -> Proposal | None:
        if self.client:
            try:
                res = self._redis_op(lambda: self.client.xrange(self.stream, count=1))
                if res:
                    key, fields = res[0]
                    self._redis_op(lambda: self.client.xdel(self.stream, key))
                    payload = {k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v for k, v in fields.items()}
                    return Proposal.parse_obj(payload)
            except RedisUnavailableError:
                self.logger.exception("redis fetch failed")
                if not self.fallback_local:
                    raise
                self.logger.warning("falling back to local queue")
        if self.queue:
            return self.queue.pop(0)
        return None

    def _publish_result(self, result: Result) -> None:
        data = {"proposal": result.proposal.dict(), "status": result.status, "detail": result.detail}
        if self.client:
            try:
                self._redis_op(lambda: self.client.xadd(self.results_stream, data))
                return
            except RedisUnavailableError:
                self.logger.exception("publish result failed")
                if self.fallback_local:
                    self.logger.warning("falling back to local queue")
                else:
                    raise
        self.results.append(result)

    # ------------------------------------------------------------------
    def _apply(self, prop: Proposal) -> Result:
        self.query(prop.target_table)
        table = self.meta.tables.get(prop.target_table)
        if table is None:
            with self.engine.begin() as conn:
                conn.execute(self.invalid_table.insert().values(proposal=prop.json()))
            res = Result(prop, "invalid", "unknown_table")
            self._publish_result(res)
            return res
        try:
            with self.engine.begin() as conn:
                if self.engine.dialect.name == "postgresql":
                    conn.execute(sqtext("SELECT pg_advisory_xact_lock(hashtext(:t))"), {"t": prop.target_table})
                if prop.operation == "insert":
                    conn.execute(table.insert().values(**prop.payload))
                elif prop.operation == "update":
                    pk = prop.payload.get("id")
                    if pk is None:
                        raise KeyError("id")
                    data = {k: v for k, v in prop.payload.items() if k != "id"}
                    conn.execute(table.update().where(table.c.id == pk).values(**data))
                elif prop.operation == "delete":
                    pk = prop.payload.get("id")
                    if pk is None:
                        raise KeyError("id")
                    conn.execute(table.delete().where(table.c.id == pk))
                else:
                    raise ValueError("unknown operation")
            res = Result(prop, "committed")
        except (SQLAlchemyError, Exception) as exc:  # pragma: no cover - runtime
            with self.engine.begin() as conn:
                conn.execute(self.invalid_table.insert().values(proposal=prop.json()))
            res = Result(prop, "failed", str(exc))
        self._publish_result(res)
        return res

    # ------------------------------------------------------------------
    def process_once(self) -> None:
        prop = self._fetch_one()
        if not prop:
            return
        if self.lock:
            try:
                handle = self.lock.lock(self.lock_key, 1000)
            except Exception:  # pragma: no cover - lock issue
                self.enqueue(prop)
                return
        else:
            handle = None
        cid = uuid.uuid4().hex
        set_correlation_id(cid)
        try:
            self._apply(prop)
        finally:
            set_correlation_id(None)
            if self.lock and handle:
                try:
                    self.lock.unlock(handle)
                except Exception:  # pragma: no cover - lock issue
                    self.logger.exception("failed unlocking redlock")

    def process_all(self) -> None:
        while True:
            prop = self._fetch_one()
            if not prop:
                break
            cid = uuid.uuid4().hex
            set_correlation_id(cid)
            try:
                self._apply(prop)
            finally:
                set_correlation_id(None)


__all__ = [
    "Proposal",
    "CentralDatabaseBot",
    "Result",
    "CentralDatabaseError",
    "RedisUnavailableError",
]
