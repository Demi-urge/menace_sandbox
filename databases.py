"""Interconnected SQLite schema for the Menace system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import logging
from typing import Literal

from db_dedup import insert_if_unique, ensure_content_hash_column
from db_router import queue_insert, SHARED_TABLES

from .env_config import DATABASE_URL
from .scope_utils import build_scope_clause, apply_scope
try:
    from sqlalchemy import (
        Boolean,
        Column,
        Float,
        ForeignKey,
        Integer,
        MetaData,
        String,
        Table,
        Text,
        Index,
        create_engine,
        event,
    )  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    (
        Boolean,
        Column,
        Float,
        ForeignKey,
        Integer,
        MetaData,
        String,
        Table,
        Text,
        Index,
        create_engine,
        event,
    ) = (None,) * 12  # type: ignore
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


@dataclass
class MenaceDB:
    """Create and expose tables for Menace's recursive databases."""

    engine: Engine
    meta: MetaData
    queue_path: str | Path | None = None
    use_queue: bool = False

    def __init__(
        self, url: str | None = None, queue_path: str | Path | None = None
    ) -> None:
        url = url or DATABASE_URL
        self.engine = create_engine(url)
        if event is not None and self.engine.url.get_backend_name() == "sqlite":
            event.listen(self.engine, "connect", _configure_sqlite_connection)
        self.meta = MetaData()
        self.queue_path = queue_path
        self.use_queue = bool(os.getenv("USE_DB_QUEUE")) or queue_path is not None

        self.models = Table(
            "models",
            self.meta,
            Column("model_id", Integer, primary_key=True),
            Column("model_name", String, unique=True),
            Column("source", String),
            Column("date_discovered", String),
            Column("tags", Text),
            Column("initial_roi_prediction", Float),
            Column("final_roi_prediction", Float),
            Column("current_status", String),
            Column("enhancement_count", Integer, default=0),
            Column("discrepancy_flag", Boolean, default=False),
            Column("error_flag", Boolean, default=False),
            Column("profitability_score", Float, default=0.0),
        )

        self.workflows = Table(
            "workflows",
            self.meta,
            Column("workflow_id", Integer, primary_key=True),
            Column("workflow_name", String),
            Column("task_tree", Text),
            Column("dependencies", Text),
            Column("resource_allocation_plan", Text),
            Column("created_from", String),
            Column("enhancement_links", Text),
            Column("discrepancy_links", Text),
            Column("status", String),
            Column("estimated_profit_per_bot", Float, default=0.0),
            Column("content_hash", Text, unique=True, nullable=False),
        )
        Index("idx_workflows_content_hash", self.workflows.c.content_hash, unique=True)

        self.model_workflows = Table(
            "model_workflows",
            self.meta,
            Column("model_id", Integer, ForeignKey("models.model_id"), primary_key=True),
            Column("workflow_id", Integer, ForeignKey("workflows.workflow_id"), primary_key=True),
        )

        self.workflow_models = Table(
            "workflow_models",
            self.meta,
            Column("workflow_id", Integer, ForeignKey("workflows.workflow_id"), primary_key=True),
            Column("model_id", Integer, ForeignKey("models.model_id"), primary_key=True),
        )

        self.workflow_bots = Table(
            "workflow_bots",
            self.meta,
            Column("workflow_id", Integer, ForeignKey("workflows.workflow_id"), primary_key=True),
            Column("bot_id", Integer, ForeignKey("bots.bot_id"), primary_key=True),
        )

        self.workflow_summaries = Table(
            "workflow_summaries",
            self.meta,
            Column("workflow_id", Integer, primary_key=True),
            Column("summary", Text),
            Column("source_menace_id", Text, nullable=False, server_default=""),
        )
        Index(
            "idx_workflow_summaries_source_menace_id",
            self.workflow_summaries.c.source_menace_id,
        )

        self.information = Table(
            "information",
            self.meta,
            Column("info_id", Integer, primary_key=True),
            Column("data_type", String),
            Column("source_url", String),
            Column("date_collected", String),
            Column("summary", Text),
            Column("validated", Boolean, default=False),
            Column("validation_notes", Text),
            Column("keywords", Text),
            Column("data_depth_score", Float),
        )

        self.information_workflows = Table(
            "information_workflows",
            self.meta,
            Column("info_id", Integer, ForeignKey("information.info_id"), primary_key=True),
            Column("workflow_id", Integer, ForeignKey("workflows.workflow_id"), primary_key=True),
        )

        self.information_models = Table(
            "information_models",
            self.meta,
            Column("info_id", Integer, ForeignKey("information.info_id"), primary_key=True),
            Column("model_id", Integer, ForeignKey("models.model_id"), primary_key=True),
        )

        self.information_bots = Table(
            "information_bots",
            self.meta,
            Column("info_id", Integer, ForeignKey("information.info_id"), primary_key=True),
            Column("bot_id", Integer, ForeignKey("bots.bot_id"), primary_key=True),
        )

        self.information_embeddings = Table(
            "information_embeddings",
            self.meta,
            Column("record_id", String, primary_key=True),
            Column("vector", Text),
            Column("created_at", String),
            Column("embedding_version", Integer),
            Column("kind", String),
            Column("source_id", String),
        )

        self.bots = Table(
            "bots",
            self.meta,
            Column("bot_id", Integer, primary_key=True),
            Column("bot_name", String),
            Column("bot_type", String),
            Column("assigned_task", String),
            Column("parent_bot_id", Integer, ForeignKey("bots.bot_id")),
            Column("dependencies", Text),
            Column("resource_estimates", Text),
            Column("creation_date", String),
            Column("last_modification_date", String),
            Column("status", String),
            Column("version", String),
            Column("estimated_profit", Float, default=0.0),
            Column(
                "source_menace_id",
                Text,
                nullable=False,
                server_default="",
            ),
            Column("content_hash", Text, unique=True, nullable=False),
        )
        Index("idx_bots_source_menace_id", self.bots.c.source_menace_id)
        Index("idx_bots_content_hash", self.bots.c.content_hash, unique=True)

        self.bot_models = Table(
            "bot_models",
            self.meta,
            Column("bot_id", Integer, ForeignKey("bots.bot_id"), primary_key=True),
            Column("model_id", Integer, ForeignKey("models.model_id"), primary_key=True),
        )

        self.bot_workflows = Table(
            "bot_workflows",
            self.meta,
            Column("bot_id", Integer, ForeignKey("bots.bot_id"), primary_key=True),
            Column("workflow_id", Integer, ForeignKey("workflows.workflow_id"), primary_key=True),
        )

        self.bot_enhancements = Table(
            "bot_enhancements",
            self.meta,
            Column(
                "bot_id",
                Integer,
                ForeignKey("bots.bot_id"),
                primary_key=True,
            ),
            Column(
                "enhancement_id",
                Integer,
                ForeignKey("enhancements.enhancement_id"),
                primary_key=True,
            ),
        )

        self.bot_errors = Table(
            "bot_errors",
            self.meta,
            Column(
                "bot_id",
                Integer,
                ForeignKey("bots.bot_id"),
                primary_key=True,
            ),
            Column(
                "error_id",
                Integer,
                ForeignKey("errors.error_id"),
                primary_key=True,
            ),
        )

        self.enhancements = Table(
            "enhancements",
            self.meta,
            Column("enhancement_id", Integer, primary_key=True),
            Column("description_of_change", Text),
            Column("reason_for_change", Text),
            Column("performance_delta", Float),
            Column("timestamp", String),
            Column("triggered_by", String),
            Column("source_menace_id", Text, nullable=False),
            Column("content_hash", Text, unique=True, nullable=False),
        )
        Index(
            "idx_enhancements_source_menace_id",
            self.enhancements.c.source_menace_id,
        )
        Index("idx_enhancements_content_hash", self.enhancements.c.content_hash, unique=True)

        self.enhancement_models = Table(
            "enhancement_models",
            self.meta,
            Column(
                "enhancement_id",
                Integer,
                ForeignKey("enhancements.enhancement_id"),
                primary_key=True,
            ),
            Column(
                "model_id",
                Integer,
                ForeignKey("models.model_id"),
                primary_key=True,
            ),
        )

        self.enhancement_bots = Table(
            "enhancement_bots",
            self.meta,
            Column(
                "enhancement_id",
                Integer,
                ForeignKey("enhancements.enhancement_id"),
                primary_key=True,
            ),
            Column(
                "bot_id",
                Integer,
                ForeignKey("bots.bot_id"),
                primary_key=True,
            ),
        )

        self.enhancement_workflows = Table(
            "enhancement_workflows",
            self.meta,
            Column(
                "enhancement_id",
                Integer,
                ForeignKey("enhancements.enhancement_id"),
                primary_key=True,
            ),
            Column(
                "workflow_id",
                Integer,
                ForeignKey("workflows.workflow_id"),
                primary_key=True,
            ),
        )

        self.code = Table(
            "code",
            self.meta,
            Column("code_id", Integer, primary_key=True),
            Column("template_type", String),
            Column("language", String),
            Column("version", String),
            Column("complexity_score", Float),
            Column("code_summary", Text),
            Column("source_menace_id", Text, nullable=False, server_default=""),
        )
        Index("idx_code_source_menace_id", self.code.c.source_menace_id)

        self.code_bots = Table(
            "code_bots",
            self.meta,
            Column("code_id", Integer, ForeignKey("code.code_id"), primary_key=True),
            Column("bot_id", Integer, ForeignKey("bots.bot_id"), primary_key=True),
        )

        self.code_enhancements = Table(
            "code_enhancements",
            self.meta,
            Column(
                "code_id",
                Integer,
                ForeignKey("code.code_id"),
                primary_key=True,
            ),
            Column(
                "enhancement_id",
                Integer,
                ForeignKey("enhancements.enhancement_id"),
                primary_key=True,
            ),
        )

        self.code_errors = Table(
            "code_errors",
            self.meta,
            Column(
                "code_id",
                Integer,
                ForeignKey("code.code_id"),
                primary_key=True,
            ),
            Column(
                "error_id",
                Integer,
                ForeignKey("errors.error_id"),
                primary_key=True,
            ),
        )

        self.errors = Table(
            "errors",
            self.meta,
            Column("error_id", Integer, primary_key=True),
            Column("timestamp", String),
            Column("error_type", String),
            Column("error_description", Text),
            Column("resolution_status", String),
            Column("source_menace_id", Text, nullable=False, server_default=""),
            Column("content_hash", Text, unique=True, nullable=False),
        )
        Index("idx_errors_source_menace_id", self.errors.c.source_menace_id)
        Index("idx_errors_content_hash", self.errors.c.content_hash, unique=True)

        self.error_bots = Table(
            "error_bots",
            self.meta,
            Column(
                "error_id",
                Integer,
                ForeignKey("errors.error_id"),
                primary_key=True,
            ),
            Column(
                "bot_id",
                Integer,
                ForeignKey("bots.bot_id"),
                primary_key=True,
            ),
        )

        self.error_models = Table(
            "error_models",
            self.meta,
            Column("error_id", Integer, ForeignKey("errors.error_id"), primary_key=True),
            Column("model_id", Integer, ForeignKey("models.model_id"), primary_key=True),
        )

        self.model_errors = Table(
            "model_errors",
            self.meta,
            Column("model_id", Integer, ForeignKey("models.model_id"), primary_key=True),
            Column("error_id", Integer, ForeignKey("errors.error_id"), primary_key=True),
        )

        self.error_codes = Table(
            "error_codes",
            self.meta,
            Column(
                "error_id",
                Integer,
                ForeignKey("errors.error_id"),
                primary_key=True,
            ),
            Column(
                "code_id",
                Integer,
                ForeignKey("code.code_id"),
                primary_key=True,
            ),
        )

        self.error_discrepancies = Table(
            "error_discrepancies",
            self.meta,
            Column(
                "error_id",
                Integer,
                ForeignKey("errors.error_id"),
                primary_key=True,
            ),
            Column(
                "discrepancy_id",
                Integer,
                ForeignKey("discrepancies.discrepancy_id"),
                primary_key=True,
            ),
        )

        self.discrepancies = Table(
            "discrepancies",
            self.meta,
            Column("discrepancy_id", Integer, primary_key=True),
            Column("description", Text),
            Column("resolution_notes", Text),
            Column("source_menace_id", Text, nullable=False, server_default=""),
        )
        Index(
            "idx_discrepancies_source_menace_id",
            self.discrepancies.c.source_menace_id,
        )

        self.discrepancy_bots = Table(
            "discrepancy_bots",
            self.meta,
            Column(
                "discrepancy_id",
                Integer,
                ForeignKey("discrepancies.discrepancy_id"),
                primary_key=True,
            ),
            Column(
                "bot_id",
                Integer,
                ForeignKey("bots.bot_id"),
                primary_key=True,
            ),
        )

        self.discrepancy_models = Table(
            "discrepancy_models",
            self.meta,
            Column(
                "discrepancy_id",
                Integer,
                ForeignKey("discrepancies.discrepancy_id"),
                primary_key=True,
            ),
            Column(
                "model_id",
                Integer,
                ForeignKey("models.model_id"),
                primary_key=True,
            ),
        )

        self.model_discrepancies = Table(
            "model_discrepancies",
            self.meta,
            Column(
                "model_id",
                Integer,
                ForeignKey("models.model_id"),
                primary_key=True,
            ),
            Column(
                "discrepancy_id",
                Integer,
                ForeignKey("discrepancies.discrepancy_id"),
                primary_key=True,
            ),
        )

        self.discrepancy_workflows = Table(
            "discrepancy_workflows",
            self.meta,
            Column(
                "discrepancy_id",
                Integer,
                ForeignKey("discrepancies.discrepancy_id"),
                primary_key=True,
            ),
            Column(
                "workflow_id",
                Integer,
                ForeignKey("workflows.workflow_id"),
                primary_key=True,
            ),
        )

        self.discrepancy_enhancements = Table(
            "discrepancy_enhancements",
            self.meta,
            Column(
                "discrepancy_id",
                Integer,
                ForeignKey("discrepancies.discrepancy_id"),
                primary_key=True,
            ),
            Column(
                "enhancement_id",
                Integer,
                ForeignKey("enhancements.enhancement_id"),
                primary_key=True,
            ),
        )

        self.audit_log = Table(
            "audit_log",
            self.meta,
            Column("log_id", Integer, primary_key=True),
            Column("timestamp", String),
            Column("bot_name", String),
            Column("action", String),
            Column("details", Text),
        )

        self.memory_embeddings = Table(
            "memory_embeddings",
            self.meta,
            Column("id", Integer, primary_key=True),
            Column("key", String),
            Column("data", Text),
            Column("version", Integer),
            Column("tags", Text),
            Column("ts", String),
            Column("embedding", Text),
        )

        self.replication_checksums = Table(
            "replication_checksums",
            self.meta,
            Column("id", Integer, primary_key=True),
            Column("timestamp", String),
            Column("checksum", String),
        )

        for table in ("bots", "workflows", "enhancements", "errors"):
            ensure_content_hash_column(table, engine=self.engine)

        self.meta.create_all(self.engine)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def _current_menace_id(self, source_menace_id: str | None) -> str:
        return source_menace_id or os.getenv("MENACE_ID", "")

    def search(
        self,
        term: str,
        *,
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> list[dict]:
        """Return rows from key tables containing *term*."""
        term_l = term.lower()
        menace_id = self._current_menace_id(source_menace_id)
        results: list[dict] = []
        tables = [
            self.models,
            self.workflows,
            self.bots,
            self.code,
            self.information,
        ]
        with self.engine.begin() as conn:
            for tbl in tables:
                try:
                    if "source_menace_id" in tbl.c:
                        clause, params = build_scope_clause(tbl.name, scope, menace_id)
                        query = apply_scope(f"SELECT * FROM {tbl.name}", clause)
                        rows = conn.exec_driver_sql(query, params).fetchall()
                    else:
                        rows = conn.execute(tbl.select()).fetchall()
                except Exception:
                    continue
                for row in rows:
                    values = " ".join(str(v) for v in row)
                    if term_l in values.lower():
                        rec = dict(row._mapping)
                        rec["table"] = tbl.name
                        results.append(rec)
        return results

    def query_vector(self, embedding: list[float], limit: int = 5) -> list[dict]:
        """Return memory rows ranked by cosine similarity to *embedding*."""
        with self.engine.begin() as conn:
            rows = (
                conn.execute(self.memory_embeddings.select()).mappings().fetchall()
            )
        scored: list[tuple[float, dict]] = []
        for row in rows:
            try:
                emb = json.loads(row["embedding"])
            except Exception:
                continue
            num = sum(a * b for a, b in zip(embedding, emb))
            denom = (sum(a * a for a in embedding) ** 0.5) * (
                sum(b * b for b in emb) ** 0.5
            )
            score = num / denom if denom else 0.0
            scored.append((score, dict(row)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [row for _, row in scored[:limit]]

    def set_model_status(self, model_id: int, status: str) -> None:
        """Update the current_status field for a model."""
        with self.engine.begin() as conn:
            conn.execute(
                self.models.update()
                .where(self.models.c.model_id == model_id)
                .values(current_status=status)
            )
    # ------------------------------------------------------------------
    # Core record helpers
    # ------------------------------------------------------------------

    def add_bot(
        self,
        bot_name: str,
        bot_type: str,
        assigned_task: str,
        *,
        parent_bot_id: int | None = None,
        dependencies: str = "[]",
        resource_estimates: str = "{}",
        creation_date: str | None = None,
        last_modification_date: str | None = None,
        status: str = "",
        version: str = "",
        estimated_profit: float = 0.0,
        source_menace_id: str | None = None,
    ) -> int:
        """Insert a bot and return its id, deduplicating on key fields."""
        menace_id = self._current_menace_id(source_menace_id)
        values = {
            "bot_name": bot_name,
            "bot_type": bot_type,
            "assigned_task": assigned_task,
            "parent_bot_id": parent_bot_id,
            "dependencies": dependencies,
            "resource_estimates": resource_estimates,
            "creation_date": creation_date,
            "last_modification_date": last_modification_date,
            "status": status,
            "version": version,
            "estimated_profit": estimated_profit,
            "source_menace_id": menace_id,
        }
        values = {k: v for k, v in values.items() if v is not None}
        hash_fields = sorted([
            "bot_name",
            "bot_type",
            "assigned_task",
            "dependencies",
            "resource_estimates",
        ])
        inserted = insert_if_unique(
            self.bots,
            values,
            hash_fields,
            menace_id,
            engine=None if self.use_queue else self.engine,
            logger=logger,
            queue_path=self.queue_path if self.use_queue else None,
        )
        if inserted is None:
            if self.use_queue:
                return 0
            raise RuntimeError("bot record not inserted")
        return int(inserted)

    def add_workflow(
        self,
        workflow_name: str,
        task_tree: str,
        dependencies: str,
        resource_allocation_plan: str,
        status: str,
        *,
        created_from: str = "",
        enhancement_links: str = "",
        discrepancy_links: str = "",
        estimated_profit_per_bot: float = 0.0,
        source_menace_id: str | None = None,
    ) -> int:
        """Insert a workflow and return its id, deduplicating on key fields."""
        menace_id = self._current_menace_id(source_menace_id)
        values = {
            "workflow_name": workflow_name,
            "task_tree": task_tree,
            "dependencies": dependencies,
            "resource_allocation_plan": resource_allocation_plan,
            "created_from": created_from,
            "enhancement_links": enhancement_links,
            "discrepancy_links": discrepancy_links,
            "status": status,
            "estimated_profit_per_bot": estimated_profit_per_bot,
        }
        hash_fields = sorted([
            "workflow_name",
            "task_tree",
            "dependencies",
            "resource_allocation_plan",
            "status",
        ])
        inserted = insert_if_unique(
            self.workflows,
            values,
            hash_fields,
            menace_id,
            engine=None if self.use_queue else self.engine,
            logger=logger,
            queue_path=self.queue_path if self.use_queue else None,
        )
        if inserted is None:
            if self.use_queue:
                return 0
            raise RuntimeError("workflow record not inserted")
        return int(inserted)

    def add_enhancement(
        self,
        description_of_change: str,
        reason_for_change: str,
        performance_delta: float,
        timestamp: str,
        triggered_by: str,
        *,
        source_menace_id: str | None = None,
    ) -> int:
        """Insert an enhancement and return its id, deduplicating on key fields."""
        menace_id = self._current_menace_id(source_menace_id)
        values = {
            "description_of_change": description_of_change,
            "reason_for_change": reason_for_change,
            "performance_delta": performance_delta,
            "timestamp": timestamp,
            "triggered_by": triggered_by,
            "source_menace_id": menace_id,
        }
        hash_fields = sorted([
            "description_of_change",
            "reason_for_change",
            "performance_delta",
            "timestamp",
            "triggered_by",
        ])
        inserted = insert_if_unique(
            self.enhancements,
            values,
            hash_fields,
            menace_id,
            engine=None if self.use_queue else self.engine,
            logger=logger,
            queue_path=self.queue_path if self.use_queue else None,
        )
        if inserted is None:
            if self.use_queue:
                return 0
            raise RuntimeError("enhancement record not inserted")
        return int(inserted)

    # ------------------------------------------------------------------
    # Error helpers
    # ------------------------------------------------------------------

    def add_error(
        self,
        description: str,
        *,
        type_: str = "runtime",
        resolution: str = "fatal",
        source_menace_id: str | None = None,
        scope: Literal["local", "global", "all"] = "local",
    ) -> int:
        """Insert a new error and return its id."""
        menace_id = self._current_menace_id(source_menace_id)
        values = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type_,
            "error_description": description,
            "resolution_status": resolution,
            "source_menace_id": menace_id,
        }
        hash_fields = sorted(["error_type", "error_description", "resolution_status"])
        inserted = insert_if_unique(
            self.errors,
            values,
            hash_fields,
            menace_id,
            engine=None if self.use_queue else self.engine,
            logger=logger,
            queue_path=self.queue_path if self.use_queue else None,
        )
        if inserted is None:
            if self.use_queue:
                return 0
            raise RuntimeError("error record not inserted")
        return int(inserted)

    def link_error_bot(self, error_id: int, bot_id: int) -> None:
        if (
            self.use_queue
            and self.error_bots.name in SHARED_TABLES
            and self.bot_errors.name in SHARED_TABLES
        ):
            payload = {
                "error_id": error_id,
                "bot_id": bot_id,
                "hash_fields": ["error_id", "bot_id"],
            }
            queue_insert(
                self.error_bots.name,
                payload,
                os.getenv("MENACE_ID", ""),
            )
            payload = {
                "bot_id": bot_id,
                "error_id": error_id,
                "hash_fields": ["bot_id", "error_id"],
            }
            queue_insert(
                self.bot_errors.name,
                payload,
                os.getenv("MENACE_ID", ""),
            )
            return
        with self.engine.begin() as conn:
            conn.execute(
                self.error_bots.insert().values(error_id=error_id, bot_id=bot_id)
            )
            conn.execute(
                self.bot_errors.insert().values(bot_id=bot_id, error_id=error_id)
            )

    def link_error_model(self, error_id: int, model_id: int) -> None:
        if (
            self.use_queue
            and self.error_models.name in SHARED_TABLES
            and self.model_errors.name in SHARED_TABLES
        ):
            payload = {
                "error_id": error_id,
                "model_id": model_id,
                "hash_fields": ["error_id", "model_id"],
            }
            queue_insert(
                self.error_models.name,
                payload,
                os.getenv("MENACE_ID", ""),
            )
            payload = {
                "model_id": model_id,
                "error_id": error_id,
                "hash_fields": ["model_id", "error_id"],
            }
            queue_insert(
                self.model_errors.name,
                payload,
                os.getenv("MENACE_ID", ""),
            )
            return
        with self.engine.begin() as conn:
            conn.execute(
                self.error_models.insert().values(error_id=error_id, model_id=model_id)
            )
            conn.execute(
                self.model_errors.insert().values(model_id=model_id, error_id=error_id)
            )

    def link_error_code(self, error_id: int, code_id: int) -> None:
        if self.use_queue and self.error_codes.name in SHARED_TABLES:
            payload = {
                "error_id": error_id,
                "code_id": code_id,
                "hash_fields": ["error_id", "code_id"],
            }
            queue_insert(
                self.error_codes.name,
                payload,
                os.getenv("MENACE_ID", ""),
            )
            return
        with self.engine.begin() as conn:
            conn.execute(
                self.error_codes.insert().values(error_id=error_id, code_id=code_id)
            )

    def flag_model_error(self, model_id: int, flag: bool = True) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                self.models.update()
                .where(self.models.c.model_id == model_id)
                .values(error_flag=flag)
            )

    # ------------------------------------------------------------------
    # Discrepancy helpers
    # ------------------------------------------------------------------

    def add_discrepancy(
        self,
        description: str,
        notes: str = "",
        *,
        source_menace_id: str | None = None,
    ) -> int:
        """Insert a discrepancy and return its id."""
        menace_id = source_menace_id or os.getenv("MENACE_ID", "")
        values = {
            "description": description,
            "resolution_notes": notes,
            "source_menace_id": menace_id,
        }
        if self.use_queue and self.discrepancies.name in SHARED_TABLES:
            payload = dict(values)
            payload["hash_fields"] = [
                "description",
                "resolution_notes",
                "source_menace_id",
            ]
            queue_insert(
                self.discrepancies.name,
                payload,
                menace_id,
            )
            return 0
        with self.engine.begin() as conn:
            res = conn.execute(self.discrepancies.insert().values(**values))
            return int(res.inserted_primary_key[0])

    def link_discrepancy_model(self, disc_id: int, model_id: int) -> None:
        if (
            self.use_queue
            and self.discrepancy_models.name in SHARED_TABLES
            and self.model_discrepancies.name in SHARED_TABLES
        ):
            payload = {
                "discrepancy_id": disc_id,
                "model_id": model_id,
                "hash_fields": ["discrepancy_id", "model_id"],
            }
            queue_insert(
                self.discrepancy_models.name,
                payload,
                os.getenv("MENACE_ID", ""),
            )
            payload = {
                "model_id": model_id,
                "discrepancy_id": disc_id,
                "hash_fields": ["model_id", "discrepancy_id"],
            }
            queue_insert(
                self.model_discrepancies.name,
                payload,
                os.getenv("MENACE_ID", ""),
            )
            return
        with self.engine.begin() as conn:
            conn.execute(
                self.discrepancy_models.insert().values(
                    discrepancy_id=disc_id, model_id=model_id
                )
            )
            conn.execute(
                self.model_discrepancies.insert().values(
                    model_id=model_id, discrepancy_id=disc_id
                )
            )

    def link_discrepancy_bot(self, disc_id: int, bot_id: int) -> None:
        if self.use_queue and self.discrepancy_bots.name in SHARED_TABLES:
            payload = {
                "discrepancy_id": disc_id,
                "bot_id": bot_id,
                "hash_fields": ["discrepancy_id", "bot_id"],
            }
            queue_insert(
                self.discrepancy_bots.name,
                payload,
                os.getenv("MENACE_ID", ""),
            )
            return
        with self.engine.begin() as conn:
            conn.execute(
                self.discrepancy_bots.insert().values(
                    discrepancy_id=disc_id, bot_id=bot_id
                )
            )

    def link_discrepancy_workflow(self, disc_id: int, workflow_id: int) -> None:
        if self.use_queue and self.discrepancy_workflows.name in SHARED_TABLES:
            payload = {
                "discrepancy_id": disc_id,
                "workflow_id": workflow_id,
                "hash_fields": ["discrepancy_id", "workflow_id"],
            }
            queue_insert(
                self.discrepancy_workflows.name,
                payload,
                os.getenv("MENACE_ID", ""),
            )
            return
        with self.engine.begin() as conn:
            conn.execute(
                self.discrepancy_workflows.insert().values(
                    discrepancy_id=disc_id, workflow_id=workflow_id
                )
            )

    def link_discrepancy_enhancement(self, disc_id: int, enh_id: int) -> None:
        if self.use_queue and self.discrepancy_enhancements.name in SHARED_TABLES:
            payload = {
                "discrepancy_id": disc_id,
                "enhancement_id": enh_id,
                "hash_fields": ["discrepancy_id", "enhancement_id"],
            }
            queue_insert(
                self.discrepancy_enhancements.name,
                payload,
                os.getenv("MENACE_ID", ""),
            )
            return
        with self.engine.begin() as conn:
            conn.execute(
                self.discrepancy_enhancements.insert().values(
                    discrepancy_id=disc_id, enhancement_id=enh_id
                )
            )

    def link_workflow_bot(self, workflow_id: int, bot_id: int) -> None:
        """Associate a bot with a workflow."""
        if self.use_queue and self.workflow_bots.name in SHARED_TABLES:
            payload = {
                "workflow_id": workflow_id,
                "bot_id": bot_id,
                "hash_fields": ["workflow_id", "bot_id"],
            }
            queue_insert(
                self.workflow_bots.name,
                payload,
                os.getenv("MENACE_ID", ""),
            )
            return
        with self.engine.begin() as conn:
            conn.execute(
                self.workflow_bots.insert().values(
                    workflow_id=workflow_id, bot_id=bot_id
                )
            )


__all__ = ["MenaceDB"]


def _configure_sqlite_connection(dbapi_connection, _) -> None:
    """Apply performance-focused pragmas when connecting to SQLite."""

    conn = dbapi_connection
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
