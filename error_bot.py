"""Error Bot for detecting and resolving system issues."""

from __future__ import annotations

import logging
import sqlite3
import json
import hashlib
from dataclasses import dataclass
import dataclasses
from datetime import datetime
from pathlib import Path

from .auto_link import auto_link
from typing import Optional, Iterable, List, TYPE_CHECKING, Sequence

from .unified_event_bus import EventBus
from .menace_memory_manager import MenaceMemoryManager, MemoryEntry
import asyncio
import threading
from jinja2 import Template
import yaml

logger = logging.getLogger(__name__)

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from .data_bot import MetricsDB, DataBot
from .error_forecaster import ErrorForecaster
from .error_logger import TelemetryEvent, ErrorLogger
from .error_ontology import ErrorType
from .knowledge_graph import KnowledgeGraph
from .database_router import DatabaseRouter
from .admin_bot_base import AdminBotBase
from .metrics_exporter import error_bot_exceptions
from .embeddable_db_mixin import EmbeddableDBMixin

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .prediction_manager_bot import PredictionManager
    from .capital_management_bot import CapitalManagementBot
    from .bot_database import BotDB
    from .databases import MenaceDB
    from .contrarian_db import ContrarianDB
    from .conversation_manager_bot import ConversationManagerBot
    from .self_coding_engine import SelfCodingEngine
    from .self_improvement_engine import SelfImprovementEngine

try:
    from prometheus_client import Summary  # type: ignore
except Exception:  # pragma: no cover - optional
    Summary = None  # type: ignore


@dataclass
class ErrorRecord:
    """Representation of an error or anomaly."""

    message: str
    resolved: bool = False
    ts: str = datetime.utcnow().isoformat()


class ErrorDB(EmbeddableDBMixin):
    """SQLite-backed storage for known errors and discrepancies."""

    def __init__(
        self,
        path: Path | str = "errors.db",
        *,
        event_bus: Optional[EventBus] = None,
        graph: KnowledgeGraph | None = None,
        vector_backend: str = "annoy",
        vector_index_path: Path | str = "error_embeddings.index",
        embedding_version: int = 1,
    ) -> None:
        self.path = Path(path)
        # Allow connection sharing across threads
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.event_bus = event_bus
        self.graph = graph
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS known_errors(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT UNIQUE,
                solution TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS discrepancies(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT,
                ts TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS errors(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT UNIQUE,
                type TEXT,
                description TEXT,
                resolution TEXT,
                ts TEXT,
                category TEXT,
                cause TEXT,
                frequency INTEGER
            )
            """
        )
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(errors)").fetchall()]
        if "category" not in cols:
            self.conn.execute("ALTER TABLE errors ADD COLUMN category TEXT")
        if "cause" not in cols:
            self.conn.execute("ALTER TABLE errors ADD COLUMN cause TEXT")
        if "frequency" not in cols:
            self.conn.execute("ALTER TABLE errors ADD COLUMN frequency INTEGER")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS error_model(error_id INTEGER, model_id INTEGER)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS error_bot(error_id INTEGER, bot_id INTEGER)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS bot_error(bot_id INTEGER, error_id INTEGER)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS error_code(error_id INTEGER, code_id INTEGER)"
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                bot_id TEXT,
                error_type TEXT,
                category TEXT,
                cause TEXT,
                stack_trace TEXT,
                root_module TEXT,
                module TEXT,
                module_counts TEXT,
                inferred_cause TEXT,
                ts TEXT,
                resolution_status TEXT,
                patch_id INTEGER,
                deploy_id INTEGER,
                frequency INTEGER
            )
            """
        )
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(telemetry)").fetchall()]
        if "patch_id" not in cols:
            self.conn.execute("ALTER TABLE telemetry ADD COLUMN patch_id INTEGER")
        if "deploy_id" not in cols:
            self.conn.execute("ALTER TABLE telemetry ADD COLUMN deploy_id INTEGER")
        if "module" not in cols:
            self.conn.execute("ALTER TABLE telemetry ADD COLUMN module TEXT")
        if "inferred_cause" not in cols:
            self.conn.execute("ALTER TABLE telemetry ADD COLUMN inferred_cause TEXT")
        if "module_counts" not in cols:
            self.conn.execute("ALTER TABLE telemetry ADD COLUMN module_counts TEXT")
        if "category" not in cols:
            self.conn.execute("ALTER TABLE telemetry ADD COLUMN category TEXT")
        if "cause" not in cols:
            self.conn.execute("ALTER TABLE telemetry ADD COLUMN cause TEXT")
        if "frequency" not in cols:
            self.conn.execute("ALTER TABLE telemetry ADD COLUMN frequency INTEGER")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS error_stats(
                category TEXT,
                module TEXT,
                count INTEGER,
                PRIMARY KEY (category, module)
            )
            """
        )
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(error_stats)").fetchall()]
        if "category" not in cols and "error_type" in cols:
            self.conn.execute(
                "ALTER TABLE error_stats RENAME COLUMN error_type TO category"
            )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS error_clusters(
                error_type TEXT PRIMARY KEY,
                cluster_id INTEGER
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS safe_mode(
                module TEXT PRIMARY KEY,
                active INTEGER
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS test_results(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                passed INTEGER,
                failed INTEGER,
                ts TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS preemptive_patches(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module TEXT,
                risk_score REAL,
                patch_id INTEGER,
                ts TEXT
            )
            """
        )
        self.conn.commit()
        EmbeddableDBMixin.__init__(
            self,
            vector_backend=vector_backend,
            index_path=vector_index_path,
            embedding_version=embedding_version,
        )

    def _publish(self, topic: str, payload: object) -> None:
        if self.event_bus:
            try:
                self.event_bus.publish(topic, payload)
            except Exception as exc:
                logger.exception("publish failed: %s", exc)
                if error_bot_exceptions:
                    error_bot_exceptions.inc()

    def _embed(self, text: str) -> list[float] | None:
        if not hasattr(self, "_embedder"):
            try:  # pragma: no cover - optional dependency
                from sentence_transformers import SentenceTransformer  # type: ignore
            except Exception:  # pragma: no cover
                self._embedder = None
            else:  # pragma: no cover
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        if getattr(self, "_embedder", None):
            try:
                return self._embedder.encode([text])[0].tolist()  # type: ignore
            except Exception:  # pragma: no cover
                return None
        return None

    def add_known(self, message: str, solution: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO known_errors(message, solution) VALUES (?, ?)",
            (message, solution),
        )
        self.conn.commit()
        self._publish("known_errors:new", {"message": message, "solution": solution})

    def find_solution(self, message: str) -> Optional[str]:
        cur = self.conn.execute(
            "SELECT solution FROM known_errors WHERE message = ?",
            (message,),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def log_discrepancy(self, message: str) -> None:
        self.conn.execute(
            "INSERT INTO discrepancies(message, ts) VALUES (?, ?)",
            (message, datetime.utcnow().isoformat()),
        )
        self.conn.commit()
        self._publish("discrepancies:new", {"message": message})

    def discrepancies(self) -> pd.DataFrame:
        return pd.read_sql("SELECT message, ts FROM discrepancies", self.conn)

    def log_preemptive_patch(
        self, module: str, risk_score: float, patch_id: int | None
    ) -> None:
        """Persist information about a preemptive patch action."""

        self.conn.execute(
            "INSERT INTO preemptive_patches(module, risk_score, patch_id, ts) VALUES (?,?,?,?)",
            (module, float(risk_score), patch_id, datetime.utcnow().isoformat()),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Safe mode helpers
    # ------------------------------------------------------------------

    def set_safe_mode(self, module: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO safe_mode(module, active) VALUES (?,1)",
            (module,),
        )
        self.conn.commit()
        self._publish("safe_mode:update", {"module": module, "active": True})

    def clear_safe_mode(self, module: str) -> None:
        self.conn.execute(
            "UPDATE safe_mode SET active=0 WHERE module=?",
            (module,),
        )
        self.conn.commit()
        self._publish("safe_mode:update", {"module": module, "active": False})

    def is_safe_mode(self, module: str) -> bool:
        cur = self.conn.execute(
            "SELECT active FROM safe_mode WHERE module=?",
            (module,),
        )
        row = cur.fetchone()
        return bool(row and row[0])

    def active_safe_modes(self) -> List[str]:
        cur = self.conn.execute(
            "SELECT module FROM safe_mode WHERE active=1"
        )
        return [r[0] for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Expanded error tracking helpers
    # ------------------------------------------------------------------

    def find_error(self, message: str) -> Optional[int]:
        cur = self.conn.execute(
            "SELECT id FROM errors WHERE message = ?",
            (message,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else None

    @auto_link({"models": "link_model", "bots": "link_bot", "codes": "link_code"})
    def add_error(
        self,
        message: str,
        *,
        type_: str = "",
        description: str | None = None,
        resolution: str = "fatal",
        models: Iterable[int] | None = None,
        bots: Iterable[str] | None = None,
        codes: Iterable[int] | None = None,
    ) -> int:
        """Insert a new error if not already present and return its id."""
        found = self.find_error(message)
        if found is not None:
            emb = self._embed(message)
            if emb:
                self.add_embedding(
                    found,
                    emb,
                    metadata={"kind": "error", "source_id": found},
                )
            return found
        cur = self.conn.execute(
            "INSERT INTO errors(message, type, description, resolution, ts) VALUES (?,?,?,?,?)",
            (
                message,
                type_,
                description or message,
                resolution,
                datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()
        err_id = int(cur.lastrowid)
        emb = self._embed(message)
        if emb:
            self.add_embedding(
                err_id,
                emb,
                metadata={"kind": "error", "source_id": err_id},
            )
        self._publish(
            "errors:new",
            {
                "id": err_id,
                "message": message,
                "type": type_,
                "description": description or message,
                "resolution": resolution,
            },
        )
        return err_id

    def backfill_embeddings(self, batch_size: int = 100) -> None:
        """Generate embeddings for errors missing vectors."""
        while True:
            rows = self.conn.execute(
                "SELECT id, message FROM errors WHERE id NOT IN (SELECT record_id FROM embeddings) LIMIT ?",
                (batch_size,),
            ).fetchall()
            if not rows:
                break
            for row in rows:
                emb = self._embed(row["message"])
                if not emb:
                    continue
                try:
                    self.add_embedding(
                        row["id"],
                        emb,
                        metadata={"kind": "error", "source_id": row["id"]},
                    )
                except Exception:  # pragma: no cover - best effort
                    logger.exception("embedding store failed for %s", row["id"])

    def link_model(self, err_id: int, model_id: int) -> None:
        self.conn.execute(
            "INSERT INTO error_model(error_id, model_id) VALUES (?, ?)",
            (err_id, model_id),
        )
        self.conn.commit()
        self._publish("error_model:new", {"error_id": err_id, "model_id": model_id})

    def link_bot(self, err_id: int, bot_id: str) -> None:
        self.conn.execute(
            "INSERT INTO error_bot(error_id, bot_id) VALUES (?, ?)",
            (err_id, bot_id),
        )
        self.conn.execute(
            "INSERT INTO bot_error(bot_id, error_id) VALUES (?, ?)",
            (bot_id, err_id),
        )
        self.conn.commit()
        self._publish("error_bot:new", {"error_id": err_id, "bot_id": bot_id})
        self._publish("bot_error:new", {"bot_id": bot_id, "error_id": err_id})

    def link_code(self, err_id: int, code_id: int) -> None:
        self.conn.execute(
            "INSERT INTO error_code(error_id, code_id) VALUES (?, ?)",
            (err_id, code_id),
        )
        self.conn.commit()
        self._publish("error_code:new", {"error_id": err_id, "code_id": code_id})

    def add_telemetry(self, event: "TelemetryEvent") -> None:
        """Insert a high-resolution error telemetry entry."""
        mods = event.module_counts or ({event.module: 1} if event.module else {})
        freq = sum(mods.values()) if mods else 1
        cur = self.conn.execute(
            """
            INSERT INTO telemetry(
                task_id,
                bot_id,
                error_type,
                category,
                cause,
                stack_trace,
                root_module,
                module,
                module_counts,
                inferred_cause,
                ts,
                resolution_status,
                patch_id,
                deploy_id,
                frequency
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.task_id,
                event.bot_id,
                getattr(event.error_type, "value", event.error_type),
                getattr(event.category, "value", event.category),
                event.root_cause,
                event.stack_trace,
                event.root_module,
                event.module,
                json.dumps(mods),
                event.inferred_cause,
                event.timestamp,
                event.resolution_status,
                event.patch_id,
                event.deploy_id,
                freq,
            ),
        )
        for mod, count in mods.items():
            self.conn.execute(
                """
                INSERT INTO error_stats(category, module, count) VALUES(?,?,?)
                ON CONFLICT(category, module) DO UPDATE SET count=count+excluded.count
                """,
                (
                    getattr(event.category, "value", event.category),
                    mod,
                    count,
                ),
            )
        self.conn.commit()
        emb_text = f"{event.root_cause} {event.stack_trace}".strip()
        emb = self._embed(emb_text)
        if emb:
            self.add_embedding(
                int(cur.lastrowid),
                emb,
                metadata={
                    "kind": "telemetry",
                    "source_id": event.task_id or event.bot_id or "",
                },
            )
        if self.graph:
            try:  # pragma: no cover - best effort
                self.graph.save(self.graph.path)
            except Exception:
                pass
        self._publish(
            "telemetry:new",
            {
                "task_id": event.task_id,
                "bot_id": event.bot_id,
                "error_type": str(event.error_type),
                "category": getattr(event.category, "value", event.category),
                "cause": event.root_cause,
                "stack_trace": event.stack_trace,
                "root_module": event.root_module,
                "module": event.module,
                "module_counts": mods,
                "inferred_cause": event.inferred_cause,
                "ts": event.timestamp,
                "resolution_status": event.resolution_status,
                "patch_id": event.patch_id,
                "deploy_id": event.deploy_id,
                "frequency": freq,
            },
        )

    def search_by_vector(
        self, vector: Sequence[float], top_k: int = 5
    ) -> list[dict[str, object]]:
        matches = EmbeddableDBMixin.search_by_vector(self, vector, top_k)
        results: list[dict[str, object]] = []
        for rec_id, dist in matches:
            row = self.conn.execute(
                "SELECT cause, stack_trace FROM telemetry WHERE id=?", (rec_id,)
            ).fetchone()
            if row:
                results.append(
                    {
                        "id": rec_id,
                        "cause": row[0],
                        "stack_trace": row[1],
                        "_distance": dist,
                    }
                )
                continue
            row = self.conn.execute(
                "SELECT message FROM errors WHERE id=?", (rec_id,)
            ).fetchone()
            if row:
                results.append(
                    {"id": rec_id, "message": row[0], "_distance": dist}
                )
        return results

    def fetch_error_stats(self) -> list[dict[str, int | str | None]]:
        """Return aggregated error counts grouped by category, module and cause."""
        try:
            cur = self.conn.execute(
                """
                SELECT category, module, cause, COUNT(*) AS cnt
                FROM telemetry
                GROUP BY category, module, cause
                """
            )
            rows = cur.fetchall()
            if rows:
                return [
                    {
                        "error_type": row[0],
                        "category": row[0],
                        "module": row[1],
                        "cause": row[2],
                        "count": row[3],
                    }
                    for row in rows
                ]
        except Exception:
            pass

        cur = self.conn.execute(
            "SELECT category, module, count FROM error_stats"
        )
        return [
            {
                "error_type": row[0],
                "category": row[0],
                "module": row[1],
                "cause": None,
                "count": row[2],
            }
            for row in cur.fetchall()
        ]

    def get_error_stats(self) -> list[dict[str, int | str | None]]:
        """Backward compatible wrapper for ``fetch_error_stats``."""
        return self.fetch_error_stats()

    def record_error_occurrence(self, category: str, module: str) -> None:
        """Increment the occurrence count for a ``(category, module)`` pair."""
        self.conn.execute(
            """
            INSERT INTO error_stats(category, module, count) VALUES(?,?,1)
            ON CONFLICT(category, module) DO UPDATE SET count=count+1
            """,
            (category, module),
        )
        self.conn.commit()

    def top_error_module(
        self, bot_id: str | None = None, *, unresolved_only: bool = False
    ) -> tuple[str, str, dict[str, int], int, str] | None:
        """Return the most frequent ``(error_type, module)`` pair.

        Parameters:
            bot_id: optionally restrict to a specific bot.
            unresolved_only: if ``True``, only consider unresolved telemetry.

        Returns a tuple ``(error_type, module, module_counts, count, sample_bot)``
        where ``module_counts`` aggregates counts for the selected
        ``error_type`` across modules and ``sample_bot`` is an example bot id
        for the returned pair.
        """

        where: list[str] = []
        params: list[str] = []
        if bot_id:
            where.append("bot_id=?")
            params.append(bot_id)
        if unresolved_only:
            where.append("resolution_status='unresolved'")
        query = "SELECT bot_id, error_type, module_counts FROM telemetry"
        if where:
            query += " WHERE " + " AND ".join(where)
        cur = self.conn.execute(query, params)
        pair_counts: dict[tuple[str, str], int] = {}
        samples: dict[tuple[str, str], str] = {}
        module_totals: dict[str, dict[str, int]] = {}
        for bot, etype, mods_json in cur.fetchall():
            try:
                mods = json.loads(mods_json or "{}")
            except Exception:
                mods = {}
            for mod, cnt in mods.items():
                key = (str(etype), str(mod))
                pair_counts[key] = pair_counts.get(key, 0) + int(cnt)
                samples.setdefault(key, str(bot or ""))
                mt = module_totals.setdefault(str(etype), {})
                mt[str(mod)] = mt.get(str(mod), 0) + int(cnt)
        if not pair_counts:
            return None
        (etype, module), count = max(pair_counts.items(), key=lambda kv: kv[1])
        return etype, module, module_totals.get(etype, {}), count, samples[(etype, module)]

    # ------------------------------------------------------------------
    def set_error_clusters(self, clusters: dict[str, int]) -> None:
        """Persist ``error_type`` → ``cluster_id`` mappings."""
        self.conn.executemany(
            "INSERT OR REPLACE INTO error_clusters(error_type, cluster_id) VALUES(?, ?)",
            list(clusters.items()),
        )
        self.conn.commit()

    def get_error_clusters(self) -> dict[str, int]:
        """Return mapping of ``error_type`` to ``cluster_id``."""
        cur = self.conn.execute(
            "SELECT error_type, cluster_id FROM error_clusters"
        )
        return {row[0]: int(row[1]) for row in cur.fetchall()}

    def get_bot_error_types(self, bot_id: str) -> list[str]:
        """Return distinct error types recorded for ``bot_id``."""
        cur = self.conn.execute(
            "SELECT DISTINCT error_type FROM telemetry WHERE bot_id = ?",
            (bot_id,),
        )
        return [row[0] for row in cur.fetchall()]

    def get_error_types_for_clusters(self, cluster_ids: list[int]) -> list[str]:
        """Return error types that belong to any of ``cluster_ids``."""
        if not cluster_ids:
            return []
        placeholders = ",".join(["?"] * len(cluster_ids))
        cur = self.conn.execute(
            f"SELECT error_type FROM error_clusters WHERE cluster_id IN ({placeholders})",
            tuple(cluster_ids),
        )
        return [row[0] for row in cur.fetchall()]

    def add_test_result(self, passed: int, failed: int) -> None:
        """Record test suite execution results."""
        self.conn.execute(
            "INSERT INTO test_results(passed, failed, ts) VALUES(?,?,?)",
            (passed, failed, datetime.utcnow().isoformat()),
        )
        self.conn.commit()


class ErrorBot(AdminBotBase):
    """Detect anomalies, resolve known issues, and patch admin bots."""

    prediction_profile = {"scope": ["errors"], "risk": ["low"]}

    def __init__(
        self,
        db: ErrorDB | None = None,
        metrics_db: MetricsDB | None = None,
        *,
        prediction_manager: "PredictionManager" | None = None,
        data_bot: DataBot | None = None,
        menace_db: "MenaceDB" | None = None,
        bot_db: "BotDB" | None = None,
        enhancement_db: "EnhancementDB" | None = None,
        conversation_bot: "ConversationManagerBot" | None = None,
        db_router: "DatabaseRouter" | None = None,
        event_bus: Optional[EventBus] = None,
        memory_mgr: MenaceMemoryManager | None = None,
        graph: KnowledgeGraph | None = None,
        forecaster: ErrorForecaster | None = None,
        self_coding_engine: "SelfCodingEngine" | None = None,
        improvement_engine: "SelfImprovementEngine" | None = None,
    ) -> None:
        super().__init__(db_router=db_router)
        self.name = "ErrorBot"
        self.db = db or ErrorDB()
        self.graph = graph
        self.error_logger = ErrorLogger(self.db, knowledge_graph=self.graph)
        self.data_bot = data_bot
        if metrics_db:
            self.metrics_db = metrics_db
        elif data_bot is not None:
            self.metrics_db = data_bot.db
        else:
            self.metrics_db = MetricsDB()
        self.prediction_manager = prediction_manager
        if self.prediction_manager:
            self.prediction_ids = self.prediction_manager.assign_prediction_bots(self)
        else:
            self.prediction_ids = []
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ErrorBot")
        if Summary:
            self.summary = Summary(
                "error_resolution_seconds",
                "Time spent resolving errors",
                registry=None,
            )
        else:  # pragma: no cover - optional
            self.summary = None
        self.menace_db = menace_db
        self.bot_db = bot_db
        self.enhancement_db = enhancement_db
        self.conversation_bot = conversation_bot
        self.event_bus = event_bus
        self.memory_mgr = memory_mgr
        self.graph = graph or KnowledgeGraph()
        # ensure DB has access to the graph for persistence
        try:
            self.db.graph = self.graph
        except Exception:  # pragma: no cover - legacy DB without attribute
            pass
        # load persisted graph before processing new telemetry
        try:
            self.graph.load(self.graph.path)
        except Exception:  # pragma: no cover - best effort
            pass
        self.forecaster = forecaster or ErrorForecaster(
            self.metrics_db, graph=self.graph
        )
        self.self_coding_engine = self_coding_engine
        self.improvement_engine = improvement_engine
        self.last_forecast_chains: dict[str, list[str]] = {}
        self.generated_runbooks: dict[str, str] = {}
        self.last_error_event: object | None = None
        self.last_memory_event: MemoryEntry | None = None
        self._cluster_cache: dict[str, int] | None = None
        self._cluster_digest: str | None = None
        if self.event_bus:
            try:
                self.event_bus.subscribe("errors:new", self._on_error_event)
            except Exception as exc:
                self.logger.exception("event bus subscription failed: %s", exc)
                if error_bot_exceptions:
                    error_bot_exceptions.inc()
        if self.memory_mgr:
            try:
                self.memory_mgr.subscribe(self._on_memory_entry)
            except Exception as exc:
                self.logger.exception("memory subscription failed: %s", exc)
                if error_bot_exceptions:
                    error_bot_exceptions.inc()

    # ------------------------------------------------------------------
    # Safe mode management
    # ------------------------------------------------------------------

    def flag_module(self, module: str) -> None:
        self.db.set_safe_mode(module)
        if self.conversation_bot:
            try:
                self.conversation_bot.notify(f"{module} entered safe mode")
            except Exception as exc:
                self.logger.exception("notification failed: %s", exc)
                if error_bot_exceptions:
                    error_bot_exceptions.inc()

    def clear_module_flag(self, module: str) -> None:
        self.db.clear_safe_mode(module)
        if self.conversation_bot:
            try:
                self.conversation_bot.notify(f"{module} exited safe mode")
            except Exception as exc:
                self.logger.exception("notification failed: %s", exc)
                if error_bot_exceptions:
                    error_bot_exceptions.inc()

    def _start_prompt_rewriter(self, err_id: int) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.prompt_rewriter_daemon(err_id))
        except RuntimeError:
            threading.Thread(
                target=lambda: asyncio.run(self.prompt_rewriter_daemon(err_id)),
                daemon=True,
            ).start()

    def _on_error_event(self, topic: str, payload: object) -> None:
        self.last_error_event = payload
        try:
            msg = payload.get("message") if isinstance(payload, dict) else str(payload)
            self.logger.info("bus error event: %s", msg)
        except Exception as exc:
            self.logger.exception("failed handling bus event: %s", exc)
            if error_bot_exceptions:
                error_bot_exceptions.inc()

    def _on_memory_entry(self, entry: MemoryEntry) -> None:
        if "error" in (entry.tags or "").lower():
            self.last_memory_event = entry
            try:
                self.logger.info("memory error entry: %s", entry.data)
            except Exception as exc:
                self.logger.exception("memory logging failed: %s", exc)
                if error_bot_exceptions:
                    error_bot_exceptions.inc()

    async def prompt_rewriter_daemon(self, err_id: int) -> None:
        if not self.enhancement_db:
            return
        with sqlite3.connect(self.db.path) as conn:
            row = conn.execute(
                "SELECT message FROM errors WHERE id=?", (err_id,)
            ).fetchone()
        if not row:
            return
        fingerprint = row[0]
        strategy_path = Path("prompt_strategy.yaml")
        try:
            strategy = (
                yaml.safe_load(strategy_path.read_text()) if strategy_path.exists() else {}
            )
        except Exception:
            strategy = {}
        history = self.enhancement_db.fetch()
        hint = ""
        for h in history:
            if h.idea == fingerprint:
                hint = h.rationale
                break
        template = Template(
            "Resolve {{ fingerprint }}. Hint: {{ troubleshooting_hint }}. Use {{ max_tokens }} tokens."
        )
        prompt = template.render(
            fingerprint=fingerprint,
            troubleshooting_hint=hint,
            max_tokens=strategy.get("max_tokens", 256),
        )
        try:
            self.enhancement_db.add_prompt_history(
                fingerprint, prompt, fix="", success=False
            )
        except Exception as exc:
            self.logger.exception("failed to store prompt history: %s", exc)
            if error_bot_exceptions:
                error_bot_exceptions.inc()

    # ------------------------------------------------------------------
    # Runtime error logging helpers
    # ------------------------------------------------------------------

    def record_runtime_error(
        self,
        message: str,
        *,
        model_id: int | None = None,
        bot_ids: Iterable[str] | None = None,
        code_ids: Iterable[int] | None = None,
        info_ids: Iterable[int] | None = None,
        contrarian_id: int | None = None,
        bot_db: "BotDB" | None = None,
        menace_db: "MenaceDB" | None = None,
        contrarian_db: "ContrarianDB" | None = None,
        bot_failed: bool = False,
        resolved: bool = False,
        stack_trace: str | None = None,
    ) -> int:
        """Log a runtime error and update related tables.

        This helper deduplicates errors based on ``message`` and links any
        provided model, bot and code identifiers using the :class:`ErrorDB`.
        If ``info_ids`` are supplied along with a ``MenaceDB`` instance, the
        corresponding information entries will be marked as unvalidated.  When
        ``contrarian_id`` and ``ContrarianDB`` are supplied the error will also
        be linked to that experiment.
        When ``stack_trace`` is provided it is stored in the telemetry table
        using the :class:`ErrorLogger`.
        """

        self.query(message)

        err_id = self.db.add_error(
            message,
            type_="runtime",
            description=message,
            resolution="successful" if resolved else "fatal",
        )

        menace_err_id = None
        if menace_db:
            try:
                menace_err_id = menace_db.add_error(
                    message,
                    type_="runtime",
                    resolution="successful" if resolved else "fatal",
                )
            except Exception:
                menace_err_id = None

        if model_id is not None:
            try:
                self.db.link_model(err_id, model_id)
            except Exception as exc:
                self.logger.exception("link_model failed: %s", exc)
                if error_bot_exceptions:
                    error_bot_exceptions.inc()
            if menace_db and menace_err_id is not None:
                try:
                    menace_db.link_error_model(menace_err_id, model_id)
                    menace_db.flag_model_error(model_id, True)
                except Exception as exc:
                    self.logger.exception("link_error_model failed: %s", exc)
                    if error_bot_exceptions:
                        error_bot_exceptions.inc()

        for bid in bot_ids or []:
            try:
                self.db.link_bot(err_id, bid)
                if menace_db and menace_err_id is not None:
                    try:
                        menace_db.link_error_bot(menace_err_id, int(bid))
                    except Exception as exc:
                        self.logger.exception("link_error_bot failed: %s", exc)
                        if error_bot_exceptions:
                            error_bot_exceptions.inc()
                if bot_failed and bot_db:
                    bot_db.update_bot(bid, status="failed")
            except Exception as exc:
                self.logger.exception("link_bot failed: %s", exc)
                if error_bot_exceptions:
                    error_bot_exceptions.inc()

        for cid in code_ids or []:
            try:
                self.db.link_code(err_id, cid)
                if menace_db and menace_err_id is not None:
                    menace_db.link_error_code(menace_err_id, cid)
            except Exception as exc:
                self.logger.exception("link_code failed: %s", exc)
                if error_bot_exceptions:
                    error_bot_exceptions.inc()

        # update knowledge graph with error relationships
        if self.graph:
            self.graph.add_error(
                err_id,
                message,
                bots=list(bot_ids or []),
                models=[model_id] if model_id is not None else [],
                codes=list(code_ids or []),
            )

        if menace_db and info_ids:
            try:
                with menace_db.engine.begin() as conn:
                    for iid in info_ids:
                        conn.execute(
                            menace_db.information.update()
                            .where(menace_db.information.c.info_id == iid)
                            .values(validated=False)
                        )
            except Exception as exc:
                self.logger.exception("link_information failed: %s", exc)
                if error_bot_exceptions:
                    error_bot_exceptions.inc()

        if contrarian_db and contrarian_id is not None:
            try:
                link_id = menace_err_id if menace_err_id is not None else err_id
                contrarian_db.link_error(contrarian_id, link_id)
            except Exception as exc:
                self.logger.exception("link_error failed: %s", exc)
                if error_bot_exceptions:
                    error_bot_exceptions.inc()

        if stack_trace:
            try:
                self.db.add_telemetry(
                    TelemetryEvent(
                        task_id=None,
                        bot_id=";".join(bot_ids or []),
                        error_type=ErrorType.RUNTIME_FAULT,
                        stack_trace=stack_trace,
                        root_module=__name__,
                        timestamp=datetime.utcnow().isoformat(),
                        resolution_status="successful" if resolved else "fatal",
                    )
                )
                for b in bot_ids or []:
                    if self.graph:
                        self.graph.add_telemetry_event(b, "runtime", __name__)
                if self.graph:
                    try:
                        self.graph.update_error_stats(self.db)
                    except Exception as exc:
                        self.logger.exception("error stats update failed: %s", exc)
            except Exception as exc:
                self.logger.exception("telemetry logging failed: %s", exc)
                if error_bot_exceptions:
                    error_bot_exceptions.inc()

        self._start_prompt_rewriter(err_id)

        if not resolved and self.db.find_solution(message) is None:
            for mod in bot_ids or []:
                self.flag_module(mod)

        return err_id

    def handle_error(self, message: str) -> str:
        """Resolve error if known, otherwise log discrepancy."""
        sol = self.db.find_solution(message)
        if sol:
            self.logger.info("Resolved known issue: %s", message)
            return sol
        self.db.log_discrepancy(message)
        self.logger.warning("Unknown issue logged: %s", message)
        return "unresolved"

    def monitor(self) -> None:
        """Check metrics for anomalies and handle errors."""
        df = self.metrics_db.fetch(50)
        rows = DataBot.detect_anomalies(
            df, "errors", threshold=2.0, metrics_db=self.metrics_db
        )
        if not rows and not df.empty and df.iloc[-1]["errors"] > 0:
            rows = [len(df) - 1]
        for idx in rows:
            bot = df.iloc[idx]["bot"]
            msg = f"high error count for {bot}"
            if self.summary:
                with self.summary.time():  # pragma: no cover - optional
                    self.handle_error(msg)
            else:
                self.handle_error(msg)

    # ------------------------------------------------------------------
    # Prediction and discrepancy helpers
    # ------------------------------------------------------------------

    def cluster_error_modules(
        self, freq_threshold: int = 5, module_threshold: int = 10
    ) -> None:
        """Cluster high-frequency error types per module and trigger mitigation."""
        if not self.graph or not getattr(self.graph, "graph", None):
            return
        # compute digest of error_type->module edges to detect changes
        edges = [
            (u, v, d.get("weight", 0))
            for u, v, d in self.graph.graph.edges(data=True)
            if u.startswith("error_type:") and v.startswith("module:")
        ]
        digest = hashlib.md5(str(sorted(edges)).encode()).hexdigest()
        if digest == self._cluster_digest:
            return
        self._cluster_digest = digest
        clusters = self.graph.error_clusters(min_weight=freq_threshold)
        if not clusters:
            return
        self._cluster_cache = {k.split(":", 1)[1]: v for k, v in clusters.items()}
        try:
            self.db.set_error_clusters(self._cluster_cache)
        except Exception:
            pass
        mod_counts: dict[str, int] = {}
        g = self.graph.graph
        for enode in clusters:
            for _, mnode, d in g.out_edges(enode, data=True):
                if mnode.startswith("module:"):
                    mod_counts[mnode] = mod_counts.get(mnode, 0) + int(d.get("weight", 1))
        for mnode, count in mod_counts.items():
            if count >= module_threshold:
                module = mnode.split(":", 1)[1]
                self.flag_module(module)
                if self.self_coding_engine:
                    path = Path(f"{module}.py")
                    if path.exists():
                        try:
                            self.self_coding_engine.apply_patch(
                                path,
                                "cluster mitigation",
                                reason="cluster mitigation",
                                trigger="error_bot",
                            )
                        except Exception as exc:  # pragma: no cover - runtime issues
                            self.logger.exception("auto patch failed: %s", exc)
                            if error_bot_exceptions:
                                error_bot_exceptions.inc()

    def predict_errors(self) -> List[str]:
        """Return predicted upcoming errors from assigned prediction bots."""
        preds: List[str] = []
        self.cluster_error_modules()
        if not self.prediction_manager:
            return preds
        df = self.metrics_db.fetch(20)
        for pid in self.prediction_manager.get_prediction_bots_for("ErrorBot"):
            entry = self.prediction_manager.registry.get(pid)
            bot = getattr(entry, "bot", None)
            if not bot or not hasattr(bot, "predict"):
                continue
            try:
                res = bot.predict(df)
                if isinstance(res, str):
                    preds.append(res)
                elif isinstance(res, Iterable):
                    preds.extend(str(r) for r in res)
            except Exception as e:  # pragma: no cover - runtime issues
                self.logger.warning("prediction failed: %s", e)
        disc = self.db.discrepancies()
        if not disc.empty:
            preds.append(str(disc["message"].value_counts().idxmax()))
        bots = []
        try:
            if hasattr(df, "empty"):
                bots = list(dict.fromkeys(df["bot"].tolist()))
            elif isinstance(df, list):
                bots = list(dict.fromkeys(r.get("bot") for r in df))
        except Exception:
            bots = []
        for b in bots:
            try:
                probs = (
                    self.forecaster.predict_error_prob(b, steps=3)
                    if self.forecaster
                    else []
                )
                if any(p > 0.8 for p in probs):
                    self.flag_module(b)
                    preds.append(f"high_error_risk_{b}")
                    if self.self_coding_engine:
                        try:
                            path = Path(f"{b}.py")
                            if path.exists():
                                self.self_coding_engine.apply_patch(
                                    path,
                                    "error mitigation",
                                    reason="error mitigation",
                                    trigger="error_bot",
                                )
                        except Exception as exc:
                            self.logger.exception("auto patch failed: %s", exc)
                            if error_bot_exceptions:
                                error_bot_exceptions.inc()
                    modules_for_fix: list[str] = []
                    if self.forecaster and self.graph:
                        try:
                            chain_nodes = self.forecaster.predict_failure_chain(b, self.graph, steps=3)
                        except Exception as exc:  # pragma: no cover - runtime issues
                            self.logger.warning("failure chain prediction failed: %s", exc)
                            chain_nodes = []
                        if chain_nodes:
                            full_chain = [f"bot:{b}"] + chain_nodes
                            self.last_forecast_chains[b] = full_chain
                            preds.append(" -> ".join(full_chain))
                            modules_for_fix = [n.split(":", 1)[1] for n in chain_nodes if n.startswith("module:")]
                            if modules_for_fix:
                                runbook = {
                                    "bot": b,
                                    "failure_chain": full_chain,
                                    "affected_modules": modules_for_fix,
                                    "mitigation": "restart affected modules and review logs",
                                }
                                path = Path(
                                    f"runbook_{b}_{hashlib.md5(str(full_chain).encode()).hexdigest()}.json"
                                )
                                path.write_text(json.dumps(runbook, indent=2))
                                self.generated_runbooks[b] = str(path)
                                preds.append(str(path))
                    if modules_for_fix and self.improvement_engine:
                        try:
                            self.improvement_engine.enqueue_preventative_fixes(modules_for_fix)
                        except Exception as exc:  # pragma: no cover - best effort
                            self.logger.exception("queue preventative fixes failed: %s", exc)
            except Exception as e:  # pragma: no cover - runtime issues
                self.logger.warning("forecast failed: %s", e)
        return preds

    def summarize_telemetry(self, limit: int = 5) -> list[dict[str, float]]:
        """Return frequent telemetry error types with success rates."""
        cur = self.db.conn.execute(
            """
            SELECT error_type,
                   COUNT(*) as c,
                   AVG(CASE WHEN resolution_status='successful' THEN 1.0 ELSE 0.0 END) as rate
            FROM telemetry
            GROUP BY error_type
            ORDER BY c DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [
            {
                "error_type": r[0] if r[0] is not None else "",
                "count": float(r[1]),
                "success_rate": float(r[2] or 0.0),
            }
            for r in rows
        ]

    def auto_patch_recurrent_errors(self, threshold: int = 5) -> None:
        """Trigger self-coding for frequent error types."""
        if not self.self_coding_engine or not self.data_bot:
            return
        summary = self.summarize_telemetry(limit=10)
        for item in summary:
            count = int(item.get("count", 0.0))
            rate = float(item.get("success_rate", 1.0))
            if count >= threshold and rate < 0.5:
                bot = self.data_bot.worst_bot("errors")
                if not bot:
                    continue
                path = Path(f"{bot}.py")
                if path.exists():
                    desc = f"fix recurring {item.get('error_type', 'error')}"
                    try:
                        self.self_coding_engine.apply_patch(
                            path,
                            desc,
                            reason=desc,
                            trigger="error_bot",
                        )
                    except Exception as exc:
                        self.logger.exception("auto patch failed: %s", exc)
                        if error_bot_exceptions:
                            error_bot_exceptions.inc()

    def scan_roi_discrepancies(self, threshold: float = 10.0) -> None:
        """Log ROI mismatches and register them in the Menace database."""
        if not self.data_bot or not getattr(self.data_bot, "capital_bot", None):
            return
        df = self.metrics_db.fetch(50)
        if df.empty:
            return
        df["actual_roi"] = df["revenue"] - df["expense"]
        df["pred_roi"] = df["bot"].apply(self.data_bot.roi)
        mism = df[abs(df["actual_roi"] - df["pred_roi"]) > threshold]
        for _, row in mism.iterrows():
            desc = f"roi_mismatch_{row['bot']}"
            self.db.log_discrepancy(desc)
            if not self.menace_db or not self.bot_db:
                continue
            try:
                disc_id = self.menace_db.add_discrepancy(desc, "auto-detected")
            except Exception:
                continue
            bot_row = self.bot_db.find_by_name(row["bot"])
            if not bot_row:
                continue
            bot_id = bot_row.get("id")
            try:
                b_int = int(bot_id)
            except Exception:
                continue
            try:
                self.menace_db.link_discrepancy_bot(disc_id, b_int)
            except Exception as exc:
                self.logger.exception("link_discrepancy_bot failed: %s", exc)
                if error_bot_exceptions:
                    error_bot_exceptions.inc()
            mids = [r[0] for r in self.bot_db.conn.execute(
                "SELECT model_id FROM bot_model WHERE bot_id=?", (bot_id,)
            ).fetchall()]
            for mid in mids:
                try:
                    self.menace_db.link_discrepancy_model(disc_id, mid)
                except Exception as exc:
                    self.logger.exception("link_discrepancy_model failed: %s", exc)
                    if error_bot_exceptions:
                        error_bot_exceptions.inc()
            wids = [r[0] for r in self.bot_db.conn.execute(
                "SELECT workflow_id FROM bot_workflow WHERE bot_id=?", (bot_id,)
            ).fetchall()]
            for wid in wids:
                try:
                    self.menace_db.link_discrepancy_workflow(disc_id, wid)
                except Exception as exc:
                    self.logger.exception("link_discrepancy_workflow failed: %s", exc)
                    if error_bot_exceptions:
                        error_bot_exceptions.inc()
            enh_ids = [r[0] for r in self.bot_db.conn.execute(
                "SELECT enhancement_id FROM bot_enhancement WHERE bot_id=?",
                (bot_id,)
            ).fetchall()]
            for enh in enh_ids:
                try:
                    self.menace_db.link_discrepancy_enhancement(disc_id, enh)
                except Exception as exc:
                    self.logger.exception("link_discrepancy_enhancement failed: %s", exc)
                    if error_bot_exceptions:
                        error_bot_exceptions.inc()

    # ------------------------------------------------------------------
    # Code patching helpers
    # ------------------------------------------------------------------

    def fix_admin_bot_schema(self, path: Path, schema: Iterable[str]) -> None:
        """Ensure an administrative bot declares the expected DB schema."""
        try:
            text = path.read_text()
            marker = "# expected_schema:"
            schema_str = ",".join(sorted(schema))
            if marker not in text:
                path.write_text(f"{marker} {schema_str}\n" + text)
            self.logger.info("Patched %s with schema %s", path, schema_str)
            self.db.add_known(f"schema_fix_{path.name}", schema_str)
        except Exception as e:  # pragma: no cover - runtime issues
            self.handle_error(str(e))

    # ------------------------------------------------------------------
    # Graph analytics helpers
    # ------------------------------------------------------------------

    def root_causes(self, bot_name: str) -> list[str]:
        """Return potential root causes for ``bot_name`` from the graph."""

        return self.graph.root_causes(bot_name) if self.graph else []


__all__ = ["ErrorRecord", "ErrorDB", "ErrorBot"]
