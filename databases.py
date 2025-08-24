"""Interconnected SQLite schema for the Menace system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os

from .env_config import DATABASE_URL
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
    ) = (None,) * 11  # type: ignore
from sqlalchemy.engine import Engine


@dataclass
class MenaceDB:
    """Create and expose tables for Menace's recursive databases."""

    engine: Engine
    meta: MetaData

    def __init__(self, url: str | None = None) -> None:
        url = url or DATABASE_URL
        self.engine = create_engine(url)
        self.meta = MetaData()

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
        )

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
        )
        Index("idx_bots_source_menace_id", self.bots.c.source_menace_id)

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
        )
        Index(
            "idx_enhancements_source_menace_id",
            self.enhancements.c.source_menace_id,
        )

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
        )
        Index("idx_errors_source_menace_id", self.errors.c.source_menace_id)

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
        include_cross_instance: bool = False,
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
                    stmt = tbl.select()
                    if not include_cross_instance and "source_menace_id" in tbl.c:
                        stmt = stmt.where(tbl.c.source_menace_id == menace_id)
                    rows = conn.execute(stmt).fetchall()
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
    # Error helpers
    # ------------------------------------------------------------------

    def add_error(
        self,
        description: str,
        *,
        type_: str = "runtime",
        resolution: str = "fatal",
        source_menace_id: str | None = None,
        include_cross_instance: bool = False,
    ) -> int:
        """Insert a new error and return its id."""
        menace_id = self._current_menace_id(source_menace_id)
        with self.engine.begin() as conn:
            stmt = self.errors.select().where(
                self.errors.c.error_description == description
            )
            if not include_cross_instance:
                stmt = stmt.where(self.errors.c.source_menace_id == menace_id)
            row = conn.execute(stmt).fetchone()
            if row:
                return int(row["error_id"])
            res = conn.execute(
                self.errors.insert().values(
                    timestamp=datetime.utcnow().isoformat(),
                    error_type=type_,
                    error_description=description,
                    resolution_status=resolution,
                    source_menace_id=menace_id,
                )
            )
            return int(res.inserted_primary_key[0])

    def link_error_bot(self, error_id: int, bot_id: int) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                self.error_bots.insert().values(error_id=error_id, bot_id=bot_id)
            )
            conn.execute(
                self.bot_errors.insert().values(bot_id=bot_id, error_id=error_id)
            )

    def link_error_model(self, error_id: int, model_id: int) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                self.error_models.insert().values(error_id=error_id, model_id=model_id)
            )
            conn.execute(
                self.model_errors.insert().values(model_id=model_id, error_id=error_id)
            )

    def link_error_code(self, error_id: int, code_id: int) -> None:
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
        with self.engine.begin() as conn:
            res = conn.execute(
                self.discrepancies.insert().values(
                    description=description,
                    resolution_notes=notes,
                    source_menace_id=menace_id,
                )
            )
            return int(res.inserted_primary_key[0])

    def link_discrepancy_model(self, disc_id: int, model_id: int) -> None:
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
        with self.engine.begin() as conn:
            conn.execute(
                self.discrepancy_bots.insert().values(
                    discrepancy_id=disc_id, bot_id=bot_id
                )
            )

    def link_discrepancy_workflow(self, disc_id: int, workflow_id: int) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                self.discrepancy_workflows.insert().values(
                    discrepancy_id=disc_id, workflow_id=workflow_id
                )
            )

    def link_discrepancy_enhancement(self, disc_id: int, enh_id: int) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                self.discrepancy_enhancements.insert().values(
                    discrepancy_id=disc_id, enhancement_id=enh_id
                )
            )

    def link_workflow_bot(self, workflow_id: int, bot_id: int) -> None:
        """Associate a bot with a workflow."""
        with self.engine.begin() as conn:
            conn.execute(
                self.workflow_bots.insert().values(
                    workflow_id=workflow_id, bot_id=bot_id
                )
            )


__all__ = ["MenaceDB"]
