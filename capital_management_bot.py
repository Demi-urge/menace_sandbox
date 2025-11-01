"""Capital management system with energy score engine."""

from __future__ import annotations

print(">>> [trace] Entered capital_management_bot.py")
print(">>> [trace] Successfully imported annotations from __future__")

print(">>> [trace] Importing BotRegistry from menace_sandbox.bot_registry...")
from .bot_registry import BotRegistry
print(">>> [trace] Successfully imported BotRegistry from menace_sandbox.bot_registry")

print(">>> [trace] Importing _DisabledSelfCodingManager, self_coding_managed from menace_sandbox.coding_bot_interface...")
from .coding_bot_interface import _DisabledSelfCodingManager, self_coding_managed
print(">>> [trace] Successfully imported _DisabledSelfCodingManager, self_coding_managed from menace_sandbox.coding_bot_interface")
print(">>> [trace] Importing SelfCodingManager, internalize_coding_bot from menace_sandbox.self_coding_manager...")
from .self_coding_manager import SelfCodingManager, internalize_coding_bot
print(">>> [trace] Successfully imported SelfCodingManager, internalize_coding_bot from menace_sandbox.self_coding_manager")
print(">>> [trace] Importing ensure_cooperative_init from menace_sandbox.shared.cooperative_init...")
from .shared.cooperative_init import ensure_cooperative_init
print(">>> [trace] Successfully imported ensure_cooperative_init from menace_sandbox.shared.cooperative_init")
print(">>> [trace] Importing SelfCodingEngine from menace_sandbox.self_coding_engine...")
from .self_coding_engine import SelfCodingEngine
print(">>> [trace] Successfully imported SelfCodingEngine from menace_sandbox.self_coding_engine")
print(">>> [trace] Importing CodeDB from menace_sandbox.code_database...")
from .code_database import CodeDB
print(">>> [trace] Successfully imported CodeDB from menace_sandbox.code_database")
print(">>> [trace] Importing GPTMemoryManager from menace_sandbox.gpt_memory...")
from .gpt_memory import GPTMemoryManager
print(">>> [trace] Successfully imported GPTMemoryManager from menace_sandbox.gpt_memory")
print(">>> [trace] Importing get_thresholds from menace_sandbox.self_coding_thresholds...")
from .self_coding_thresholds import get_thresholds
print(">>> [trace] Successfully imported get_thresholds from menace_sandbox.self_coding_thresholds")
print(">>> [trace] Importing get_orchestrator from menace_sandbox.orchestrator_loader...")
from .orchestrator_loader import get_orchestrator
print(">>> [trace] Successfully imported get_orchestrator from menace_sandbox.orchestrator_loader")
print(">>> [trace] Importing ThresholdService from menace_sandbox.threshold_service...")
from .threshold_service import ThresholdService
print(">>> [trace] Successfully imported ThresholdService from menace_sandbox.threshold_service")
print(">>> [trace] Importing CapitalMetrics, DataBotInterface from menace_sandbox.data_interfaces...")
from .data_interfaces import CapitalMetrics, DataBotInterface
print(">>> [trace] Successfully imported CapitalMetrics, DataBotInterface from menace_sandbox.data_interfaces")
print(">>> [trace] Importing get_data_bot as _load_shared_data_bot from menace_sandbox.shared.lazy_data_bot...")
from .shared.lazy_data_bot import get_data_bot as _load_shared_data_bot
print(">>> [trace] Successfully imported get_data_bot as _load_shared_data_bot from menace_sandbox.shared.lazy_data_bot")
print(">>> [trace] Importing create_context_builder from context_builder_util...")
from context_builder_util import create_context_builder
print(">>> [trace] Successfully imported create_context_builder from context_builder_util")
print(">>> [trace] Importing dataclass, field from dataclasses...")
from dataclasses import dataclass, field
print(">>> [trace] Successfully imported dataclass, field from dataclasses")
print(">>> [trace] Importing datetime, timedelta from datetime...")
from datetime import datetime, timedelta
print(">>> [trace] Successfully imported datetime, timedelta from datetime")
print(">>> [trace] Importing Path from pathlib...")
from pathlib import Path
print(">>> [trace] Successfully imported Path from pathlib")
print(">>> [trace] Importing Dict, Iterable, List, Mapping, Optional, Callable, Tuple, cast, Literal, TYPE_CHECKING from typing...")
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Callable,
    Tuple,
    cast,
    Literal,
    TYPE_CHECKING,
)
print(">>> [trace] Successfully imported Dict, Iterable, List, Mapping, Optional, Callable, Tuple, cast, Literal, TYPE_CHECKING from typing")
print(">>> [trace] Importing Enum from enum...")
from enum import Enum
print(">>> [trace] Successfully imported Enum from enum")
print(">>> [trace] Importing os...")
import os
print(">>> [trace] Successfully imported os")
print(">>> [trace] Importing threading...")
import threading
print(">>> [trace] Successfully imported threading")
print(">>> [trace] Importing time...")
import time
print(">>> [trace] Successfully imported time")
print(">>> [trace] Importing logging...")
import logging
print(">>> [trace] Successfully imported logging")
print(">>> [trace] Importing deque from collections...")
from collections import deque
print(">>> [trace] Successfully imported deque from collections")
print(">>> [trace] Importing asyncio...")
import asyncio
print(">>> [trace] Successfully imported asyncio")
print(">>> [trace] Importing statistics...")
import statistics
print(">>> [trace] Successfully imported statistics")
print(">>> [trace] Importing DBRouter, GLOBAL_ROUTER, init_db_router from db_router...")
from db_router import DBRouter, GLOBAL_ROUTER, init_db_router
print(">>> [trace] Successfully imported DBRouter, GLOBAL_ROUTER, init_db_router from db_router")
print(">>> [trace] Importing Scope, build_scope_clause, apply_scope from scope_utils...")
from scope_utils import Scope, build_scope_clause, apply_scope
print(">>> [trace] Successfully imported Scope, build_scope_clause, apply_scope from scope_utils")

try:  # pragma: no cover - optional dependency during bootstrap
    print(">>> [trace] Importing self_coding_import_depth from menace_sandbox.shared.self_coding_import_guard...")
    from .shared.self_coding_import_guard import self_coding_import_depth
    print(">>> [trace] Successfully imported self_coding_import_depth from menace_sandbox.shared.self_coding_import_guard")
except Exception:  # pragma: no cover - support flat execution
    print(">>> [trace] Failed to import menace_sandbox.shared.self_coding_import_guard, falling back to shared.self_coding_import_guard")
    from shared.self_coding_import_guard import self_coding_import_depth  # type: ignore
    print(">>> [trace] Successfully imported self_coding_import_depth from shared.self_coding_import_guard")

_registry_instance: BotRegistry | None = None
_data_bot_instance: DataBotInterface | None = None
_context_builder_instance: object | None = None
_engine_instance: SelfCodingEngine | None = None

# ``manager`` is injected later during module import once optional dependencies
# have been resolved.  Define it up-front so decorators referencing the symbol
# do not raise ``NameError`` during class creation when the bootstrap sequence
# has not yet initialised the self-coding manager.
manager: SelfCodingManager | None = None

try:  # pragma: no cover - optional dependency
    print(">>> [trace] Importing load_dotenv from dotenv...")
    from dotenv import load_dotenv
    print(">>> [trace] Successfully imported load_dotenv from dotenv")
except Exception:  # pragma: no cover - missing optional dependency
    print(">>> [trace] Failed to import load_dotenv from dotenv")
    load_dotenv = None
else:  # pragma: no cover - load env if library present
    load_dotenv()

print(">>> [trace] Importing send_discord_alert from menace_sandbox.alert_dispatcher...")
from .alert_dispatcher import send_discord_alert
print(">>> [trace] Successfully imported send_discord_alert from menace_sandbox.alert_dispatcher")

print(">>> [trace] Importing DB_PATH from menace_sandbox.database_manager...")
from .database_manager import DB_PATH
print(">>> [trace] Successfully imported DB_PATH from menace_sandbox.database_manager")
print(">>> [trace] Importing PathwayDB from menace_sandbox.neuroplasticity...")
from .neuroplasticity import PathwayDB
print(">>> [trace] Successfully imported PathwayDB from menace_sandbox.neuroplasticity")
try:  # pragma: no cover - optional dependency
    print(">>> [trace] Importing TrendPredictor from menace_sandbox.trend_predictor...")
    from .trend_predictor import TrendPredictor
    print(">>> [trace] Successfully imported TrendPredictor from menace_sandbox.trend_predictor")
except Exception:  # pragma: no cover - fallback when unavailable
    print(">>> [trace] Failed to import TrendPredictor from menace_sandbox.trend_predictor")
    TrendPredictor = None  # type: ignore
print(">>> [trace] Importing retry from menace_sandbox.retry_utils...")
from .retry_utils import retry
print(">>> [trace] Successfully imported retry from menace_sandbox.retry_utils")

if TYPE_CHECKING:  # pragma: no cover - typing only import avoids circular dependency
    print(">>> [trace] Importing DataBot for type checking from menace_sandbox.data_bot...")
    from .data_bot import DataBot
    print(">>> [trace] Successfully imported DataBot for type checking from menace_sandbox.data_bot")
    print(">>> [trace] Importing PredictionManager for type checking from menace_sandbox.prediction_manager_bot...")
    from .prediction_manager_bot import PredictionManager
    print(">>> [trace] Successfully imported PredictionManager for type checking from menace_sandbox.prediction_manager_bot")
    print(">>> [trace] Importing ModelAutomationPipeline for type checking from menace_sandbox.shared.pipeline_base...")
    from .shared.pipeline_base import ModelAutomationPipeline
    print(">>> [trace] Successfully imported ModelAutomationPipeline for type checking from menace_sandbox.shared.pipeline_base")
    print(">>> [trace] Importing SelfCodingThresholds for type checking from menace_sandbox.self_coding_thresholds...")
    from .self_coding_thresholds import SelfCodingThresholds
    print(">>> [trace] Successfully imported SelfCodingThresholds for type checking from menace_sandbox.self_coding_thresholds")
    print(">>> [trace] Importing ErrorBot for type checking from menace_sandbox.error_bot...")
    from .error_bot import ErrorBot
    print(">>> [trace] Successfully imported ErrorBot for type checking from menace_sandbox.error_bot")

logger = logging.getLogger(__name__)


def _get_pipeline_cls() -> "type[ModelAutomationPipeline] | None":
    """Load the :class:`ModelAutomationPipeline` implementation lazily."""

    try:
        from .entry_pipeline_loader import load_pipeline_class
    except Exception as exc:  # pragma: no cover - degraded bootstrap
        logger.warning(
            "ModelAutomationPipeline unavailable for CapitalManagementBot: %s",
            exc,
        )
        return None

    try:
        return load_pipeline_class()
    except Exception as exc:  # pragma: no cover - pipeline unavailable
        logger.warning(
            "ModelAutomationPipeline unavailable for CapitalManagementBot: %s",
            exc,
        )
        return None


def _get_router(router: DBRouter | None = None) -> DBRouter:
    """Return an initialised :class:`DBRouter` instance."""

    return router or GLOBAL_ROUTER or init_db_router("capital")


def _get_registry() -> BotRegistry:
    """Return a cached :class:`BotRegistry` instance."""

    global _registry_instance
    if _registry_instance is None:
        _registry_instance = BotRegistry()
    return _registry_instance


def _get_data_bot() -> DataBotInterface:
    """Return a cached :class:`DataBot` instance."""

    global _data_bot_instance
    if _data_bot_instance is None:
        _data_bot_instance = _load_shared_data_bot(logger)
    return _data_bot_instance


def _get_context_builder() -> object:
    """Return the shared context builder instance."""

    global _context_builder_instance
    if _context_builder_instance is None:
        _context_builder_instance = create_context_builder()
    return _context_builder_instance


def _get_engine() -> SelfCodingEngine:
    """Return a cached :class:`SelfCodingEngine` instance."""

    global _engine_instance
    if _engine_instance is None:
        _engine_instance = SelfCodingEngine(
            CodeDB(),
            GPTMemoryManager(),
            context_builder=_get_context_builder(),
        )
    return _engine_instance


# ---------------------------------------------------------------------------
# Configuration defaults can be overridden via environment variables. The
# values are loaded once at module import time so that classes relying on them
# have sensible defaults even when instantiated directly.
METRICS_DB_PATH = os.environ.get("CM_METRICS_DB", "metrics.db")
LEDGER_DB_PATH = os.environ.get("CM_LEDGER_PATH", "capital_ledger.db")
ALLOC_LEDGER_DB_PATH = os.environ.get(
    "CM_ALLOC_LEDGER_PATH", "allocation_ledger.db"
)
ROI_EVENTS_DB_PATH = os.environ.get("CM_ROI_DB_PATH", "roi_events.db")
PROFIT_HISTORY_DB_PATH = os.environ.get(
    "CM_PROFIT_HISTORY_DB", "profit_history.db"
)
SUMMARY_HISTORY_DB_PATH = os.environ.get("CM_SUMMARY_DB", "capital_summary.db")
ENERGY_HISTORY_LIMIT = int(os.environ.get("CM_ENERGY_HISTORY_LIMIT", "50"))
TREND_LOG_WINDOW = int(os.environ.get("CM_LOG_TREND_WINDOW", "100"))
ROI_FETCH_LIMIT = int(os.environ.get("CM_ROI_FETCH_LIMIT", "50"))
SUMMARY_FETCH_LIMIT = int(os.environ.get("CM_SUMMARY_FETCH_LIMIT", "50"))
PROFIT_HISTORY_LEN = int(os.environ.get("CM_PROFIT_HISTORY_LEN", "10"))
TRAIN_MIN_HISTORY = int(os.environ.get("CM_TRAIN_MIN_HISTORY", "20"))

# Energy thresholds for coarse energy level classification. These can be
# customised via environment variables so behaviour can be tuned without code
# changes.  Values must be in the ``0..1`` range.
ENERGY_THRESHOLDS = {
    "very_low": float(os.environ.get("CM_ENERGY_THRESH_VERY_LOW", "0.1")),
    "low": float(os.environ.get("CM_ENERGY_THRESH_LOW", "0.3")),
    "medium": float(os.environ.get("CM_ENERGY_THRESH_MEDIUM", "0.6")),
    "high": float(os.environ.get("CM_ENERGY_THRESH_HIGH", "0.8")),
}

# Default weights for ``dynamic_weighted_energy_score`` can be overridden via
# environment variables.  Values are normalised when used.
DYN_SCORE_WEIGHTS = {
    "capital": float(os.environ.get("CM_DYN_SCORE_WEIGHT_CAPITAL", "0.3")),
    "roi": float(os.environ.get("CM_DYN_SCORE_WEIGHT_ROI", "0.3")),
    "volatility": float(os.environ.get("CM_DYN_SCORE_WEIGHT_VOLATILITY", "-0.2")),
    "risk": float(os.environ.get("CM_DYN_SCORE_WEIGHT_RISK", "0.1")),
    "stddev": float(os.environ.get("CM_DYN_SCORE_WEIGHT_STDDEV", "-0.1")),
}

# Maximum Discord alert length to avoid API errors
ALERT_MAX_LEN = int(os.environ.get("CM_ALERT_MAX_LEN", "2000"))

# Additional tunables for database queries and scoring
ROI_UPDATE_FETCH_LIMIT = int(os.environ.get("CM_ROI_UPDATE_FETCH_LIMIT", "1000"))
ENGAGEMENT_FETCH_LIMIT = int(os.environ.get("CM_ENGAGEMENT_FETCH_LIMIT", "50"))
DYN_NORM_FACTOR = float(
    os.environ.get("CM_DYN_NORM_FACTOR", os.environ.get("CM_CAPITAL_NORM", "1000"))
)

# Path to a JSON file containing fallback metric values. If the file does not
# exist or cannot be parsed the defaults below are used instead.
METRIC_FALLBACK_PATH = Path(
    os.environ.get(
        "CM_METRIC_FALLBACK_PATH",
        "config/capital_metrics_fallbacks.json",
    )
)
try:
    if METRIC_FALLBACK_PATH.exists():
        print(">>> [trace] Importing json as _json for METRIC_FALLBACKS...")
        import json as _json

        METRIC_FALLBACKS = _json.loads(METRIC_FALLBACK_PATH.read_text())
    else:
        METRIC_FALLBACKS = {}
except Exception as exc:  # pragma: no cover - IO errors
    logger.exception("failed loading metric fallbacks: %s", exc)
    METRIC_FALLBACKS = {}

# Risk profile multipliers for :meth:`CapitalManagementConfig.apply_risk_profile`.
# ``CM_RISK_AGGR_MULT`` controls how much the conserve and aggressive
# thresholds are lowered when the "aggressive" profile is active, while
# ``CM_RISK_CONSERV_MULT`` adjusts them upward for the "conservative" profile.
# Values must be positive floats.


try:  # optional dependency
    print(">>> [trace] Importing pandas as pd...")
    import pandas as pd
    print(">>> [trace] Successfully imported pandas as pd")
except Exception:  # pragma: no cover - optional dependency
    print(">>> [trace] Failed to import pandas as pd")
    pd = None


@retry(Exception, attempts=3, delay=0.5)
def fetch_metric_from_db(
    metric_name: str,
    db_path: Path | str = METRICS_DB_PATH,
    *,
    default: float | None = None,
    error_bot: "ErrorBot" | None = None,
    webhook_url: str | None = None,
    router: DBRouter | None = None,
    scope: Literal["local", "global", "all"] = "local",
    source_menace_id: str | None = None,
) -> float:
    """Fetch the latest metric value from a SQLite database."""
    try:
        router = _get_router(router)
        menace_id = source_menace_id or router.menace_id
        clause, scope_params = build_scope_clause("metrics", Scope(scope), menace_id)
        sql = "SELECT value FROM metrics WHERE name=?"
        sql = apply_scope(sql, clause)
        sql += " ORDER BY ts DESC LIMIT 1"
        params = [metric_name]
        params.extend(scope_params)

        conn = router.get_connection("metrics")
        cur = conn.execute(sql, params)
        row = cur.fetchone()
        if row is None:
            if default is None:
                raise KeyError(f"metric {metric_name} not found")
            if error_bot:
                error_bot.flag_module(__name__)
            if webhook_url:
                try:
                    send_discord_alert(
                        f"Metric {metric_name} missing; using fallback",
                        webhook_url,
                    )
                except Exception:
                    logger.exception("discord alert failed")
            return float(default)
        return float(row[0])
    except Exception as exc:  # pragma: no cover - DB or parsing error
        logger.exception("metric fetch failed: %s", exc)
        if default is not None:
            if error_bot:
                error_bot.flag_module(__name__)
            if webhook_url:
                try:
                    send_discord_alert(
                        f"Metric {metric_name} fallback due to error: {exc}",
                        webhook_url,
                    )
                except Exception:
                    logger.exception("discord alert failed")
            return float(default)
        raise


def calculate_trend_from_logs(log_path: Path, window: int = TREND_LOG_WINDOW) -> float:
    """Calculate a simple trend value from numeric log entries."""
    try:
        from collections import deque as _deque

        vals: _deque[float] = _deque(maxlen=window)
        with open(log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    vals.append(float(line.strip()))
                except ValueError:
                    continue
        if len(vals) < 2:
            return 0.0
        diffs = [b - a for a, b in zip(list(vals)[:-1], list(vals)[1:])]
        return sum(diffs) / len(diffs)
    except Exception as exc:  # pragma: no cover - IO error
        logger.exception("trend calculation failed: %s", exc)
        return 0.0


def get_bot_performance_score(metrics: Iterable[float]) -> float:
    """Return an aggregate performance score from iterable metrics."""
    try:
        metrics = list(metrics)
        if not metrics:
            return 0.0
        return float(sum(metrics) / len(metrics))
    except Exception as exc:  # pragma: no cover - numeric issues
        logger.exception("performance score failed: %s", exc)
        return 0.0


def _normalise_capital_metrics(data: Mapping[str, float]) -> CapitalMetrics:
    """Convert raw mapping to :class:`CapitalMetrics`."""

    extras = {
        k: float(v)
        for k, v in data.items()
        if k not in CapitalMetrics._STANDARD_FIELDS
    }
    return CapitalMetrics(
        capital=float(data.get("capital", 0.0)),
        profit_trend=float(data.get("profit_trend", 0.0)),
        load=float(data.get("load", 0.0)),
        success_rate=float(data.get("success_rate", 0.0)),
        deploy_efficiency=float(data.get("deploy_efficiency", 0.0)),
        failure_rate=float(data.get("failure_rate", 0.0)),
        extras=extras,
    )


def get_capital_metrics(
    db_path: Path | str = METRICS_DB_PATH,
    *,
    cache: Optional[Dict[str, float]] = None,
    fallbacks: Optional[Dict[str, float]] = None,
    error_bot: "ErrorBot" | None = None,
    webhook_url: str | None = None,
) -> CapitalMetrics:
    """Return latest capital metrics from *db_path* with validation."""
    metrics = {}
    cache = cache or {}
    fallbacks = fallbacks or {}
    for name in [
        "capital",
        "profit_trend",
        "load",
        "success_rate",
        "deploy_efficiency",
        "failure_rate",
    ]:
        try:
            val = fetch_metric_from_db(
                name,
                db_path,
                default=cache.get(name, fallbacks.get(name)),
                error_bot=error_bot,
                webhook_url=webhook_url,
            )
            metrics[name] = float(val)
            cache[name] = metrics[name]
        except Exception:
            logger.exception("failed fetching metric %s", name)
            metrics[name] = cache.get(name, fallbacks.get(name, 0.0))
    return _normalise_capital_metrics(metrics)


async def fetch_capital_metrics_async(
    db_path: Path | str = METRICS_DB_PATH,
    metric_names: Optional[Iterable[str]] = None,
    *,
    cache: Optional[Dict[str, float]] = None,
    fallbacks: Optional[Dict[str, float]] = None,
    error_bot: "ErrorBot" | None = None,
    webhook_url: str | None = None,
    scope: Literal["local", "global", "all"] = "local",
    source_menace_id: str | None = None,
    ) -> CapitalMetrics:
    """Asynchronously fetch capital metrics from the database."""
    metric_names = list(
        metric_names
        or [
            "capital",
            "profit_trend",
            "load",
            "success_rate",
            "deploy_efficiency",
            "failure_rate",
        ]
    )
    cache = cache or {}
    fallbacks = fallbacks or {}

    async def _get(name: str) -> tuple[str, float]:
        val = await asyncio.to_thread(
            fetch_metric_from_db,
            name,
            db_path,
            default=cache.get(name, fallbacks.get(name)),
            error_bot=error_bot,
            webhook_url=webhook_url,
            scope=scope,
            source_menace_id=source_menace_id,
        )
        cache[name] = float(val)
        return name, float(val)

    tasks = [_get(n) for n in metric_names]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    data: Dict[str, float] = {}
    for res in results:
        if isinstance(res, Exception):
            logger.exception("async metric fetch failed: %s", res)
            continue
        key, val = cast(tuple[str, float], res)
        data[key] = val
    return _normalise_capital_metrics(data)


@dataclass
class LedgerEntry:
    """Entry recording capital inflow or expense."""

    entry_type: str  # inflow or expense or allocation
    amount: float
    description: str
    ts: str = datetime.utcnow().isoformat()


class CapitalLedger:
    """SQLite-backed ledger for capital movements."""

    def __init__(
        self, path: Path | str = LEDGER_DB_PATH, *, router: DBRouter | None = None
    ) -> None:
        # allow cross-thread usage since the bot may run async workers
        self.lock = threading.Lock()
        try:
            self.router = _get_router(router)
            self.conn = self.router.get_connection("ledger")
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ledger(
                    entry_type TEXT,
                    amount REAL,
                    description TEXT,
                    ts TEXT
                )
                """
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_ts ON ledger(ts)")
            self.conn.commit()
        except Exception as exc:
            logger.exception("ledger init failed: %s", exc)
            self.conn = None

    def log(self, entry: LedgerEntry) -> None:
        if self.conn is None:
            logger.warning("ledger DB unavailable; entry not logged")
            return
        try:
            with self.lock:
                self.conn.execute(
                    "INSERT INTO ledger(entry_type, amount, description, ts) VALUES(?,?,?,?)",
                    (entry.entry_type, entry.amount, entry.description, entry.ts),
                )
                self.conn.commit()
                logger.debug(
                    "ledger entry recorded type=%s amount=%s desc=%s",
                    entry.entry_type,
                    entry.amount,
                    entry.description,
                )
        except Exception as exc:
            logger.exception("ledger log failed: %s", exc)

    def fetch(self) -> List[Tuple[str, float, str, str]]:
        if self.conn is None:
            return []
        with self.lock:
            try:
                cur = self.conn.execute(
                    "SELECT entry_type, amount, description, ts FROM ledger"
                )
                return cur.fetchall()
            except Exception as exc:
                logger.exception("ledger fetch failed: %s", exc)
                return []

    def total(self) -> float:
        if self.conn is None:
            return 0.0
        with self.lock:
            try:
                cur = self.conn.execute("SELECT SUM(amount) FROM ledger")
                row = cur.fetchone()
                return float(row[0] or 0.0)
            except Exception as exc:
                logger.exception("ledger total failed: %s", exc)
                return 0.0


class CapitalAllocationLedger(CapitalLedger):
    """Ledger dedicated to capital deployment decisions."""

    def __init__(
        self, path: Path | str = ALLOC_LEDGER_DB_PATH, *, router: DBRouter | None = None
    ) -> None:
        super().__init__(path, router=router)


@dataclass
class ROIEvent:
    """ROI before/after an evolution cycle."""

    action: str
    roi_before: float
    roi_after: float
    ts: str = datetime.utcnow().isoformat()


class ROIEventDB:
    """SQLite-backed store for ROI change events."""

    def __init__(
        self, path: Path | str = ROI_EVENTS_DB_PATH, *, router: DBRouter | None = None
    ) -> None:
        try:
            self.router = _get_router(router)
            self.conn = self.router.get_connection("roi_events")
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS roi_events(
                    action TEXT,
                    roi_before REAL,
                    roi_after REAL,
                    ts TEXT
                )
                """
            )
            self.conn.commit()
        except Exception as exc:
            logger.exception("ROIEventDB init failed: %s", exc)
            self.conn = None

    def add(self, rec: ROIEvent) -> int:
        if self.conn is None:
            return 0
        try:
            cur = self.conn.execute(
                "INSERT INTO roi_events(action, roi_before, roi_after, ts) VALUES(?,?,?,?)",
                (rec.action, rec.roi_before, rec.roi_after, rec.ts),
            )
            self.conn.commit()
            rid = int(cur.lastrowid or 0)
            logger.debug(
                "ROI event recorded id=%s action=%s before=%s after=%s",
                rid,
                rec.action,
                rec.roi_before,
                rec.roi_after,
            )
            return rid
        except Exception as exc:
            logger.exception("failed to add ROI event: %s", exc)
            return 0

    def fetch(self, limit: int = ROI_FETCH_LIMIT) -> List[Tuple[str, float, float, str]]:
        if self.conn is None:
            return []
        try:
            cur = self.conn.execute(
                "SELECT action, roi_before, roi_after, ts FROM roi_events ORDER BY ts DESC LIMIT ?",
                (limit,),
            )
            return cur.fetchall()
        except Exception as exc:
            logger.exception("failed to fetch ROI events: %s", exc)
            return []


@dataclass
class ProfitRecord:
    """Recorded profit at a point in time."""

    profit: float
    ts: str = datetime.utcnow().isoformat()


class ProfitHistoryDB:
    """Simple SQLite store of profit over time."""

    def __init__(
        self, path: Path | str = PROFIT_HISTORY_DB_PATH, *, router: DBRouter | None = None
    ) -> None:
        try:
            self.router = _get_router(router)
            self.conn = self.router.get_connection("profit_history")
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profit_history(
                    profit REAL,
                    ts TEXT
                )
                """
            )
            self.conn.commit()
        except Exception as exc:
            logger.exception("ProfitHistoryDB init failed: %s", exc)
            self.conn = None

    def log_profit(self, rec: ProfitRecord) -> None:
        if self.conn is None:
            return
        try:
            self.conn.execute(
                "INSERT INTO profit_history(profit, ts) VALUES(?,?)",
                (rec.profit, rec.ts),
            )
            self.conn.commit()
            logger.debug("profit record logged value=%s ts=%s", rec.profit, rec.ts)
        except Exception as exc:
            logger.exception("failed to log profit: %s", exc)


@dataclass
class SummaryRecord:
    """Recorded summary of capital metrics."""

    run_id: str
    capital: float
    trend: float
    energy: float
    message: str
    ts: str = datetime.utcnow().isoformat()


class SummaryHistoryDB:
    """SQLite-backed store of capital summaries."""

    def __init__(
        self, path: Path | str = SUMMARY_HISTORY_DB_PATH, *, router: DBRouter | None = None
    ) -> None:
        try:
            self.router = _get_router(router)
            self.conn = self.router.get_connection("capital_summary")
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS capital_summary(
                    run_id TEXT,
                    capital REAL,
                    trend REAL,
                    energy REAL,
                    message TEXT,
                    ts TEXT
                )
                """,
            )
            self.conn.commit()
        except Exception as exc:
            logger.exception("SummaryHistoryDB init failed: %s", exc)
            self.conn = None

    def log_summary(self, rec: SummaryRecord) -> None:
        if self.conn is None:
            return
        try:
            self.conn.execute(
                (
                    "INSERT INTO capital_summary("  # noqa: E501
                    "run_id, capital, trend, energy, message, ts"
                    ") VALUES(?,?,?,?,?,?)"
                ),
                (rec.run_id, rec.capital, rec.trend, rec.energy, rec.message, rec.ts),
            )
            self.conn.commit()
            logger.debug(
                "summary logged run_id=%s capital=%s trend=%s energy=%s",
                rec.run_id,
                rec.capital,
                rec.trend,
                rec.energy,
            )
        except Exception as exc:
            logger.exception("failed to log summary: %s", exc)

    def fetch(
        self, limit: int = SUMMARY_FETCH_LIMIT
    ) -> List[Tuple[str, float, float, float, str, str]]:
        if self.conn is None:
            return []
        try:
            cur = self.conn.execute(
                (
                    "SELECT run_id, capital, trend, energy, message, ts "
                    "FROM capital_summary ORDER BY ts DESC LIMIT ?"
                ),
                (limit,),
            )
            return cur.fetchall()
        except Exception as exc:
            logger.exception("failed to fetch summaries: %s", exc)
            return []


@dataclass
class CapitalManagementConfig:
    """Configuration options for :class:`CapitalManagementBot`."""

    config_path: str | None = os.getenv("CM_CONFIG_FILE")
    decision_log_path: str = os.getenv("CM_DECISION_LOG", "capital_decisions.jsonl")
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "capital": float(os.getenv("CM_WEIGHT_CAPITAL", "0.3")),
            "profit_trend": float(os.getenv("CM_WEIGHT_PROFIT_TREND", "0.2")),
            "load": float(os.getenv("CM_WEIGHT_LOAD", "0.1")),
            "success": float(os.getenv("CM_WEIGHT_SUCCESS", "0.2")),
            "efficiency": float(os.getenv("CM_WEIGHT_EFFICIENCY", "0.1")),
            "failure": float(os.getenv("CM_WEIGHT_FAILURE", "0.1")),
            "myelination": float(os.getenv("CM_WEIGHT_MYELINATION", "0.0")),
            "market": float(os.getenv("CM_WEIGHT_MARKET", "0.0")),
            "engagement": float(os.getenv("CM_WEIGHT_ENGAGEMENT", "0.0")),
        }
    )
    dynamic_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "capital": float(os.getenv("CM_DYN_WEIGHT_CAPITAL", "0.3")),
            "roi": float(os.getenv("CM_DYN_WEIGHT_ROI", "0.3")),
            "volatility": float(os.getenv("CM_DYN_WEIGHT_VOLATILITY", "-0.2")),
            "risk": float(os.getenv("CM_DYN_WEIGHT_RISK", "0.1")),
            "stddev": float(os.getenv("CM_DYN_WEIGHT_STDDEV", "-0.1")),
        }
    )
    conserve_threshold: float = float(os.getenv("CM_THRESHOLD_CONSERVE", "0.3"))
    aggressive_threshold: float = float(os.getenv("CM_THRESHOLD_AGGRESSIVE", "0.7"))
    info_ratio_base: Dict[str, float] = field(
        default_factory=lambda: {
            "conserve": float(os.getenv("CM_INFO_RATIO_CONSERVE", "0.5")),
            "deploy": float(os.getenv("CM_INFO_RATIO_DEPLOY", "0.8")),
            "invest": float(os.getenv("CM_INFO_RATIO_INVEST", "1.0")),
            "aggressive": float(os.getenv("CM_INFO_RATIO_AGGRESSIVE", "1.5")),
        }
    )
    daily_budget: float | None = float(os.getenv("CM_DAILY_BUDGET", "0")) or None
    weekly_budget: float | None = float(os.getenv("CM_WEEKLY_BUDGET", "0")) or None
    profit_margin: float = float(os.getenv("CM_PROFIT_MARGIN", "0.1"))
    min_profit_margin: float = float(os.getenv("CM_MIN_PROFIT_MARGIN", "0.05"))
    capital_norm_factor: float = float(os.getenv("CM_CAPITAL_NORM", "1000"))
    risk_profile: str = os.getenv("CM_RISK_PROFILE", "balanced")
    trend_threshold: float = float(os.getenv("CM_TREND_THRESHOLD", "0.1"))
    volatility_threshold: float = float(os.getenv("CM_VOLATILITY_THRESHOLD", "0.05"))
    roi_interval: float = float(os.getenv("CM_ROI_INTERVAL", "60"))
    roi_db_path: str = os.getenv("CM_ROI_DB_PATH", "roi_events.db")
    ledger_path: str = os.getenv("CM_LEDGER_PATH", "capital_ledger.db")
    alloc_ledger_path: str = os.getenv("CM_ALLOC_LEDGER_PATH", "allocation_ledger.db")
    energy_state_path: str | None = os.getenv("CM_ENERGY_STATE_PATH")
    default_energy_threshold: float = field(
        default_factory=lambda: float(os.getenv("CM_DEFAULT_ENERGY_THRESHOLD", "0.7"))
    )
    binary_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "trend": float(os.getenv("CM_BINARY_WEIGHT_TREND", "0.4")),
            "capital": float(os.getenv("CM_BINARY_WEIGHT_CAPITAL", "0.3")),
            "volatility": float(os.getenv("CM_BINARY_WEIGHT_VOLATILITY", "0.3")),
        }
    )
    risk_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            "aggressive": float(os.getenv("CM_RISK_MULT_AGGRESSIVE", "1.2")),
            "balanced": float(os.getenv("CM_RISK_MULT_BALANCED", "1.0")),
            "conservative": float(os.getenv("CM_RISK_MULT_CONSERVATIVE", "0.8")),
        }
    )
    metrics_db_path: str = os.getenv("CM_METRICS_DB", METRICS_DB_PATH)
    metric_fallbacks_path: str | None = os.getenv("CM_METRIC_FALLBACK_PATH")
    metric_fallbacks: Dict[str, float] = field(
        default_factory=lambda: METRIC_FALLBACKS
        or {
            "capital": 0.0,
            "profit_trend": 0.0,
            "load": 0.5,
            "success_rate": 0.8,
            "deploy_efficiency": 0.7,
            "failure_rate": 0.2,
        }
    )
    energy_thresholds: Dict[str, float] = field(
        default_factory=lambda: dict(ENERGY_THRESHOLDS)
    )
    threshold_step: float = field(
        default_factory=lambda: float(os.getenv("CM_THRESHOLD_STEP", "0.05"))
    )

    def __post_init__(self) -> None:
        if self.config_path:
            import json

            try:
                with open(self.config_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as exc:  # pragma: no cover - IO errors
                logger.exception("failed to load CM config: %s", exc)
            else:
                for name, value in data.items():
                    if hasattr(self, name):
                        setattr(self, name, value)
                logger.debug("loaded config overrides from %s", self.config_path)
        # Clamp threshold_step to sensible bounds
        self.threshold_step = max(0.0, min(0.2, self.threshold_step))

        if self.metric_fallbacks_path:
            import json

            try:
                with open(self.metric_fallbacks_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as exc:  # pragma: no cover - IO errors
                logger.exception("failed to load metric fallbacks: %s", exc)
            else:
                if isinstance(data, dict):
                    self.metric_fallbacks.update({k: float(v) for k, v in data.items()})

    def normalize_weights(self) -> None:
        total = sum(abs(v) for v in self.weights.values()) or 1.0
        for k in self.weights:
            self.weights[k] = self.weights[k] / total
        d_total = sum(abs(v) for v in self.dynamic_weights.values()) or 1.0
        for k in self.dynamic_weights:
            self.dynamic_weights[k] = self.dynamic_weights[k] / d_total
        b_total = sum(abs(v) for v in self.binary_weights.values()) or 1.0
        for k in self.binary_weights:
            self.binary_weights[k] = self.binary_weights[k] / b_total

    def apply_risk_profile(self) -> None:
        aggr_mult = float(os.environ.get("CM_RISK_AGGR_MULT", "0.8"))
        conserv_mult = float(os.environ.get("CM_RISK_CONSERV_MULT", "1.2"))
        if self.risk_profile == "aggressive":
            self.conserve_threshold *= aggr_mult
            self.aggressive_threshold *= aggr_mult
        elif self.risk_profile == "conservative":
            self.conserve_threshold *= conserv_mult
            self.aggressive_threshold *= conserv_mult


class EnergyScoreEngine:
    """Compute a system-wide energy score from various metrics."""

    def __init__(
        self,
        lr: float | None = None,
        momentum: float | None = None,
        history_limit: int | None = None,
        *,
        weights=None,
        state_path=None,
        capital_norm_factor: float | None = None,
    ) -> None:
        import os as _os

        lr = lr if lr is not None else float(_os.getenv("CM_ENGINE_LR", "0.05"))
        momentum = momentum if momentum is not None else float(
            _os.getenv("CM_ENGINE_MOMENTUM", "0.7")
        )
        history_limit = history_limit if history_limit is not None else int(
            _os.getenv("CM_ENGINE_HISTORY_LIMIT", "100")
        )
        capital_norm_factor = capital_norm_factor if capital_norm_factor is not None else float(
            _os.getenv("CM_CAPITAL_NORM", "1000")
        )
        self.weights = weights or {
            "capital": 0.3,
            "profit_trend": 0.2,
            "load": 0.1,
            "success": 0.2,
            "efficiency": 0.1,
            "failure": 0.1,
            "myelination": 0.0,
            "market": 0.0,
            "engagement": 0.0,
        }
        self.lr = lr
        self.momentum = momentum
        self.prev_score = 0.5
        self.history_limit = history_limit
        self.feature_history: List[List[float]] = []
        self.reward_history: List[float] = []
        from pathlib import Path as _Path

        self.state_path = _Path(state_path) if state_path else None
        self.capital_norm_factor = capital_norm_factor
        try:
            from sklearn.linear_model import LinearRegression

            self.model: LinearRegression | None = LinearRegression()
        except Exception:
            self.model = None
        self._load_state()

    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        import json as _json
        import logging as _logging

        try:
            data = _json.loads(self.state_path.read_text())
            self.weights.update(data.get("weights", {}))
            self.prev_score = data.get("prev_score", self.prev_score)
            self.feature_history = data.get("feature_history", [])
            self.reward_history = data.get("reward_history", [])
            _logging.getLogger(__name__).debug("energy state loaded from %s", self.state_path)
        except Exception as exc:
            _logging.getLogger(__name__).exception(
                "failed to load energy state: %s", exc
            )

    def _save_state(self) -> None:
        if not self.state_path:
            return
        import json as _json
        import logging as _logging

        try:
            data = {
                "weights": self.weights,
                "prev_score": self.prev_score,
                "feature_history": self.feature_history,
                "reward_history": self.reward_history,
            }
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(_json.dumps(data))
            _logging.getLogger(__name__).debug("energy state saved to %s", self.state_path)
        except Exception as exc:
            _logging.getLogger(__name__).exception(
                "failed to save energy state: %s", exc
            )

    @staticmethod
    def _clamp(val: float) -> float:
        return max(0.0, min(1.0, val))

    @staticmethod
    def _logistic(val: float, k: float | None = None) -> float:
        import math
        import os as _os

        if k is None:
            k = float(_os.environ.get("CM_LOGISTIC_K", "5.0"))
        return 1.0 / (1.0 + math.exp(-k * (val - 0.5)))

    def _base_score(self, feats: dict[str, float]) -> float:
        score = sum(self.weights.get(k, 0.0) * v for k, v in feats.items())
        return self._clamp(score)

    def _update_weights(self, feats: dict[str, float], reward: float) -> None:
        pred = self._base_score(feats)
        error = reward - pred
        for k in self.weights:
            if k in feats:
                self.weights[k] += self.lr * error * feats[k]
        total = sum(abs(w) for w in self.weights.values()) or 1.0
        for k in self.weights:
            self.weights[k] = self.weights[k] / total

    def _train_model(self) -> None:
        if not self.model or len(self.feature_history) < TRAIN_MIN_HISTORY:
            return
        import numpy as np

        X = np.array(self.feature_history)
        y = np.array(self.reward_history)
        try:
            self.model.fit(X, y)
        except Exception as exc:
            logger.exception("model training failed: %s", exc)
            self.model = None

    def compute(
        self,
        capital: float,
        profit_trend: float,
        load: float,
        success_rate: float,
        deploy_eff: float,
        failure_rate: float,
        *,
        external_signals: dict[str, float] | None = None,
        reward: float | None = None,
    ) -> float:
        external_signals = external_signals or {}
        norm_cap = self._logistic(self._clamp(capital / self.capital_norm_factor))
        norm_trend = self._logistic(self._clamp((profit_trend + 1.0) / 2.0))
        norm_load = 1.0 - self._logistic(self._clamp(load))
        norm_success = self._logistic(self._clamp(success_rate))
        norm_eff = self._logistic(self._clamp(deploy_eff))
        norm_fail = 1.0 - self._logistic(self._clamp(failure_rate))
        feats = {
            "capital": norm_cap,
            "profit_trend": norm_trend,
            "load": norm_load,
            "success": norm_success,
            "efficiency": norm_eff,
            "failure": norm_fail,
            "myelination": self._clamp(external_signals.get("myelination", 0.0)),
            "market": self._clamp(external_signals.get("market", 0.0)),
            "engagement": self._clamp(external_signals.get("engagement", 0.0)),
        }

        score = self._base_score(feats)
        if self.model:
            try:
                pred = float(self.model.predict([list(feats.values())])[0])
                score = (score + pred) / 2.0
            except Exception as exc:
                import logging as _logging

                _logging.getLogger(__name__).exception("prediction failed: %s", exc)

        score = self.momentum * self.prev_score + (1.0 - self.momentum) * score
        self.prev_score = score

        if reward is not None:
            self._update_weights(feats, reward)
            self.feature_history.append(list(feats.values()))
            self.reward_history.append(reward)
            if len(self.feature_history) > self.history_limit:
                self.feature_history = self.feature_history[-self.history_limit:]
                self.reward_history = self.reward_history[-self.history_limit:]
            self._train_model()
            self._save_state()

        return self._clamp(score)


def dynamic_weighted_energy_score(
    capital: float,
    roi: float,
    volatility: float,
    risk_profile: str,
    performance_stddev: float,
    *,
    norm_factor: float = DYN_NORM_FACTOR,
    weights: Optional[Dict[str, float]] = None,
    log_func: Callable[[float], float] | None = None,
    clamp: Callable[[float], float] | None = None,
    risk_multipliers: Optional[Dict[str, float]] = None,
) -> float:
    """Return an adaptive energy level using multiple signals."""

    if log_func is None:
        import math

        def log_func(x: float, k: float | None = None) -> float:
            import os as _os
            if k is None:
                k = float(_os.environ.get("CM_LOGISTIC_K", "5.0"))
            return 1.0 / (1.0 + math.exp(-k * (x - 0.5)))

    if clamp is None:

        def clamp(v: float) -> float:
            return max(0.0, min(1.0, v))

    if weights is None:
        weights = dict(DYN_SCORE_WEIGHTS)

    risk_factor = risk_multiplier(volatility, risk_profile, risk_multipliers)

    features = {
        "capital": log_func(clamp(capital / norm_factor)),
        "roi": log_func(clamp((roi + 1.0) / 2.0)),
        "volatility": 1.0 - log_func(clamp(volatility)),
        "risk": risk_factor,
        "stddev": 1.0 - log_func(clamp(performance_stddev)),
    }

    score = sum(weights.get(k, 0.0) * features.get(k, 0.0) for k in weights)
    logger.debug(
        "dyn_energy weights=%s features=%s score=%s",
        weights,
        {k: round(v, 4) for k, v in features.items()},
        round(score, 4),
    )
    return clamp(score)


def risk_multiplier(
    volatility: float, risk_profile: str, multipliers: Dict[str, float] | None = None
) -> float:
    """Return a risk multiplier influenced by volatility and profile."""
    base_map = multipliers or {
        "aggressive": float(os.getenv("CM_RISK_MULT_AGGRESSIVE", "1.2")),
        "balanced": float(os.getenv("CM_RISK_MULT_BALANCED", "1.0")),
        "conservative": float(os.getenv("CM_RISK_MULT_CONSERVATIVE", "0.8")),
    }
    base = base_map.get(risk_profile, 1.0)
    return base * max(0.5, 1.0 - volatility)


def calculate_energy_level(
    trend: float,
    profit_growth: float,
    history: Iterable[float] | None = None,
    *,
    metrics: Optional[Iterable[float]] = None,
    risk_profile: str = "balanced",
) -> float:
    """Return an energy level using trend, profit growth and recent metrics."""

    history = list(history or [])
    metric_vals = list(metrics or [])
    avg_hist = statistics.mean(history) if history else 0.5
    volatility = statistics.pstdev(history) if len(history) > 1 else 0.0
    risk = risk_multiplier(volatility, risk_profile)
    metrics_avg = statistics.mean(metric_vals) if metric_vals else 0.5
    base = (trend * 0.5 + profit_growth * 0.3 + metrics_avg * 0.2) * risk
    base *= max(0.0, 1.0 - volatility)
    score = (base + avg_hist) / 2.0
    return max(0.0, min(1.0, score))


def calculate_allocation_score(
    energy: float,
    trend: float,
    *,
    urgency: float = 0.5,
    risk_profile: str = "balanced",
) -> float:
    """Return an adaptive allocation score based on energy and trend."""

    risk = risk_multiplier(0.0, risk_profile)
    score = (energy * 0.7 + trend * 0.3) * risk
    score *= 1.0 + min(max(urgency, 0.0), 1.0) * 0.1
    return max(0.0, min(1.0, score))


class StrategyTier(str, Enum):
    """Capital deployment strategies."""

    HIBERNATE = "hibernate"
    CONSERVE = "conserve"
    DEPLOY = "deploy"
    INVEST = "invest"
    AGGRESSIVE = "aggressive"


class _CapitalManagementBot:
    """Manage capital and decide reinvestment based on energy score."""

    prediction_profile = {"scope": ["energy"], "risk": ["medium"]}

    def __init__(
        self,
        ledger: CapitalLedger | None = None,
        allocation_ledger: CapitalAllocationLedger | None = None,
        engine: EnergyScoreEngine | None = None,
        data_bot: DataBotInterface | None = None,
        pathway_db: "PathwayDB" | None = None,
        prediction_manager: "PredictionManager" | None = None,
        trend_predictor: TrendPredictor | None = None,
        roi_db: ROIEventDB | None = None,
        profit_history_db: ProfitHistoryDB | None = None,
        summary_db: SummaryHistoryDB | None = None,
        *,
        daily_budget: float | None = None,
        weekly_budget: float | None = None,
        error_bot: "ErrorBot" | None = None,
        notifier: Optional[Callable[[str], None]] = None,
        webhook_url: str | None = None,
        config: CapitalManagementConfig | None = None,
        signal_adapters: Optional[Iterable[Callable[[], Dict[str, float]]]] = None,
    ) -> None:
        self.config = config or CapitalManagementConfig()
        self.config.normalize_weights()
        self.config.apply_risk_profile()
        self.run_id = os.getenv("CM_RUN_ID") or __import__("uuid").uuid4().hex
        self.ledger = ledger or CapitalLedger(self.config.ledger_path)
        self.alloc_ledger = allocation_ledger or CapitalAllocationLedger(
            self.config.alloc_ledger_path
        )
        self.engine = engine or EnergyScoreEngine(
            weights=dict(self.config.weights),
            state_path=self.config.energy_state_path,
            capital_norm_factor=self.config.capital_norm_factor,
        )
        self.state: StrategyTier = StrategyTier.CONSERVE
        self.last_profit = 0.0
        self.profit_history = deque(maxlen=PROFIT_HISTORY_LEN)
        self.data_bot = data_bot
        self.pathway_db = pathway_db
        self.prediction_manager = prediction_manager
        self.trend_predictor = trend_predictor
        self.roi_db = roi_db or ROIEventDB(self.config.roi_db_path)
        self.profit_history_db = profit_history_db
        self.summary_db = summary_db
        self.daily_budget = (
            daily_budget if daily_budget is not None else self.config.daily_budget
        )
        self.weekly_budget = (
            weekly_budget if weekly_budget is not None else self.config.weekly_budget
        )
        self.error_bot = error_bot
        self.notifier = notifier
        self.webhook_url = webhook_url or os.getenv("CM_DISCORD_WEBHOOK_URL")
        self.signal_adapters = list(signal_adapters or [])
        self.assigned_prediction_bots = []
        if self.prediction_manager:
            try:
                self.assigned_prediction_bots = self.prediction_manager.assign_prediction_bots(self)
            except Exception as exc:
                logger.exception("Failed to assign prediction bots: %s", exc)
        self.energy_history: deque[float] = deque(maxlen=ENERGY_HISTORY_LIMIT)
        self._roi_thread: threading.Thread | None = None
        self._roi_stop = False
        logger.info("CapitalManagementBot init run_id=%s", self.run_id)

    # ------------------------------------------------------------------
    def _apply_prediction_bots(self, base: float, feats: Iterable[float]) -> float:
        """Combine predictions from assigned bots."""
        if not self.prediction_manager:
            return base
        score = base
        count = 1
        vec = list(feats)
        for pid in self.assigned_prediction_bots:
            entry = self.prediction_manager.registry.get(pid)
            if not entry or not entry.bot:
                continue
            pred = getattr(entry.bot, "predict", None)
            if not callable(pred):
                continue
            try:
                val = pred(vec)
                if isinstance(val, (list, tuple)):
                    val = val[0]
                score += float(val)
                count += 1
            except Exception as exc:
                logger.exception(
                    "prediction bot %s failed: %s",
                    entry.bot.__class__.__name__ if entry else "unknown",
                    exc,
                )
                continue
        return float(score / count)

    def log_inflow(self, amount: float, source: str = "") -> None:
        self.ledger.log(LedgerEntry("inflow", amount, source))
        logger.debug("logged inflow amount=%s source=%s", amount, source)

    def log_expense(self, amount: float, reason: str = "") -> None:
        self.ledger.log(LedgerEntry("expense", -amount, reason))
        logger.debug("logged expense amount=%s reason=%s", amount, reason)
        self.check_budget()

    def record_profit(self, value: float) -> None:
        if not hasattr(self, "profit_history_db") or self.profit_history_db is None:
            return
        try:
            self.profit_history_db.log_profit(ProfitRecord(value))
            logger.debug("profit recorded value=%s", value)
        except Exception as exc:
            logger.exception("profit history logging failed: %s", exc)

    def profit(self) -> float:
        total = self.ledger.total()
        margin = max(self.config.profit_margin, self.config.min_profit_margin)
        return max(0.0, total * margin)

    def profit_trend(self) -> float:
        current = self.profit()
        self.record_profit(current)
        self.profit_history.append(current)
        slope = 0.0
        forecast = 0.0
        if self.trend_predictor is not None:
            try:
                pred = self.trend_predictor.predict_future_metrics(1)
                forecast = getattr(pred, "roi", 0.0) - current
            except Exception as exc:
                import logging as _logging

                _logging.getLogger(__name__).exception(
                    "trend prediction failed: %s", exc
                )
                forecast = 0.0
        if len(self.profit_history) > 1:
            try:
                import numpy as _np

                y = _np.array(self.profit_history, dtype=float)
                x = _np.arange(len(y))
                slope = float(_np.polyfit(x, y, 1)[0])
                if len(y) >= 3:
                    window = y[-3:]
                    avg = float(window.mean())
                    std = float(window.std()) or 1.0
                    norm = (y[-1] - avg) / std
                    slope += norm
            except Exception:
                try:
                    slope = (self.profit_history[-1] - self.profit_history[0]) / float(
                        len(self.profit_history) - 1
                    )
                except Exception as exc:
                    logger.exception("trend calc failed: %s", exc)
                    slope = 0.0
        self.last_profit = current
        return max(0.0, (slope + forecast) / 100.0)

    # ------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, float]:
        """Return the latest metrics for energy calculations.

        This method first attempts to read metrics from the configured metrics
        database. If anything fails it falls back to deriving values from the
        current ledger or uses sensible defaults. All failures are logged so the
        calling code can remain oblivious to transient issues.
        """

        metrics_obj: CapitalMetrics | None = None
        metrics: Dict[str, float] = {}
        try:
            metrics_obj = get_capital_metrics(
                self.config.metrics_db_path,
                fallbacks=self.config.metric_fallbacks,
                error_bot=self.error_bot,
                webhook_url=self.webhook_url,
            )
            metrics = metrics_obj.to_dict()
        except Exception as exc:  # pragma: no cover - runtime fetch issues
            logger.exception("get_metrics failed, falling back to simulation: %s", exc)

        if not metrics:
            logger.warning("Metrics DB unavailable, using simulated metrics")
            capital = self.profit()
            metrics = dict(self.config.metric_fallbacks)
            metrics["capital"] = capital
            metrics["profit_trend"] = self.profit_trend()
            if self.error_bot:
                self.error_bot.flag_module(self.__class__.__name__)
            if self.webhook_url:
                try:
                    self.send_alert("Metrics DB fallback triggered")
                except Exception:
                    logger.exception("alert failed")

        for key, val in list(metrics.items()):
            try:
                metrics[key] = float(val)
            except Exception as exc:  # pragma: no cover - invalid value
                logger.exception("corrupted metric %s=%r: %s", key, val, exc)
                metrics[key] = 0.0

        if metrics_obj is not None:
            extras = {k: v for k, v in metrics_obj.extras.items() if k not in metrics}
            metrics.update(extras)
        return metrics

    # ------------------------------------------------------------------
    def _spent_since(self, since: datetime) -> float:
        """Return total amount spent since *since* handling DB errors."""
        if not getattr(self.ledger, "conn", None):
            return 0.0
        try:
            cur = self.ledger.conn.execute(
                "SELECT amount FROM ledger WHERE ts >= ?",
                (since.isoformat(),),
            )
            rows = cur.fetchall()
            return float(-sum(r[0] for r in rows if r[0] < 0.0))
        except Exception as exc:
            logger.exception("spent_since query failed: %s", exc)
            return 0.0

    def spent_today(self) -> float:
        start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        return self._spent_since(start)

    def spent_week(self) -> float:
        start = datetime.utcnow() - timedelta(days=7)
        return self._spent_since(start)

    def check_budget(self) -> bool:
        breached = False
        if self.daily_budget is not None and self.spent_today() > self.daily_budget:
            breached = True
        if self.weekly_budget is not None and self.spent_week() > self.weekly_budget:
            breached = True
        if breached:
            if self.error_bot:
                self.error_bot.flag_module(self.__class__.__name__)
                if getattr(self.error_bot, "conversation_bot", None):
                    try:
                        self.error_bot.conversation_bot.notify(
                            "budget threshold exceeded"
                        )
                    except Exception as exc:
                        logger.exception("notify failed: %s", exc)
            if self.notifier:
                try:
                    self.notifier("capital budget threshold exceeded")
                except Exception as exc:
                    logger.exception("notifier failed: %s", exc)
        return not breached

    def spending_anomalies(self, limit: int = 100, threshold: float = 3.0) -> list[int]:
        rows = self.ledger.fetch()[-limit:]
        amounts = [-r[1] for r in rows if r[0] == "expense"]
        if not amounts:
            return []
        if pd is not None:
            df = pd.DataFrame({"amount": amounts})
        else:
            df = [{"amount": a} for a in amounts]
        try:
            detector = _get_data_bot()
            return cast(
                list[int], detector.detect_anomalies(df, "amount", threshold=threshold)
            )
        except Exception as exc:
            logger.exception("anomaly detection failed: %s", exc)
            return []

    def check_anomalies(self, *, limit: int = 100, threshold: float = 3.0) -> bool:
        anomalies = self.spending_anomalies(limit=limit, threshold=threshold)
        if anomalies and self.error_bot:
            self.error_bot.flag_module(self.__class__.__name__)
        return not bool(anomalies)

    def dynamic_energy_threshold(self) -> float:
        """Calculate an adaptive energy threshold based on recent history."""
        if not self.energy_history:
            return self.config.default_energy_threshold
        try:
            import numpy as _np

            arr = _np.array(self.energy_history, dtype=float)
            th = max(0.1, float(arr.mean() - arr.std()))
            return th
        except Exception as exc:  # pragma: no cover - numerical issues
            logger.exception("energy threshold calc failed: %s", exc)
            return self.config.default_energy_threshold

    def profit_trend_description(self) -> str:
        """Return a human-friendly description of the profit trend."""
        if len(self.profit_history) < 2:
            return "Insufficient data to determine capital trend."
        latest = self.profit_history[-1]
        prev = self.profit_history[-2]
        if prev == 0:
            change = 0.0
        else:
            change = (latest - prev) / abs(prev)
        if abs(change) < 0.01:
            return "Capital remains steady."
        if len(self.profit_history) >= 5:
            try:
                import numpy as _np

                arr = _np.array(self.profit_history, dtype=float)
                pct_std = float(_np.std(arr) / (_np.mean(arr) or 1.0))
                if pct_std > 0.1:
                    return "Capital trend fluctuating \u26a0\ufe0f"
            except Exception as exc:
                logger.exception("volatility calc failed: %s", exc)
        return (
            "Capital is increasing. \ud83d\udcc8"
            if change > 0
            else "Capital is decreasing. \ud83d\udcc9"
        )

    @retry(Exception, attempts=3, delay=1.0)
    def send_alert(self, message: str) -> bool:
        """Send a Discord alert if a webhook is configured."""
        if not self.webhook_url:
            logger.warning("Discord webhook not configured")
            return False
        if len(message) > ALERT_MAX_LEN:
            message = message[-ALERT_MAX_LEN:]
        logger.debug("sending alert len=%s", len(message))
        try:
            return send_discord_alert(message, self.webhook_url)
        except Exception as exc:  # pragma: no cover - network/other
            logger.exception("sending Discord alert failed: %s", exc)
            return False

    def auto_rollback(
        self,
        *,
        energy_threshold: float | None = None,
        roi_drop: float = -0.1,
        callback: Callable[[], None] | None = None,
    ) -> None:
        energy = 0.0
        try:
            energy = self.energy_score(
                load=0.0,
                success_rate=1.0,
                deploy_eff=1.0,
                failure_rate=0.0,
            )
        except Exception as exc:
            logger.exception("energy score failed: %s", exc)
        trend = self.profit_trend()
        if energy_threshold is None:
            energy_threshold = self.dynamic_energy_threshold()
        if energy < energy_threshold or trend <= roi_drop:
            if callback:
                try:
                    callback()
                except Exception as exc:
                    logger.exception("callback failed: %s", exc)
            if self.error_bot:
                self.error_bot.flag_module(self.__class__.__name__)
            self.send_alert(
                f"Energy drop detected: {energy:.2f}, trend: {trend:.2f}. Rolled back."
            )

    def log_evolution_event(self, action: str, before: float, after: float) -> None:
        """Record ROI change for an evolution cycle."""
        if not self.roi_db:
            return
        try:
            self.roi_db.add(ROIEvent(action, before, after))
        except Exception as exc:
            logger.exception("failed to log ROI event: %s", exc)

    def bot_roi(self, bot: str) -> float:
        if not self.data_bot:
            return 0.0
        try:
            df = self.data_bot.db.fetch(ROI_UPDATE_FETCH_LIMIT)
            df = df[df["bot"] == bot]
            if df.empty:
                return 0.0
            revenue = float(df["revenue"].sum())
            expense = float(df["expense"].sum())
            return max(0.0, revenue - expense)
        except Exception as exc:
            logger.exception("bot ROI fetch failed: %s", exc)
            return 0.0

    @retry(Exception, attempts=3, delay=1.0)
    def update_rois(self, models_db: Path = DB_PATH) -> None:
        if not self.data_bot:
            return
        try:
            df = self.data_bot.db.fetch(ROI_UPDATE_FETCH_LIMIT)
            if getattr(df, "empty", True):
                return
            if not all(k in df.columns for k in ("bot", "revenue", "expense")):
                return
            if hasattr(df, "groupby"):
                grp = df.groupby("bot")[["revenue", "expense"]].sum().reset_index()
                rows = grp.to_dict("records")
            else:
                rows = {}
                for r in df:
                    key = r.get("bot")
                    if not key:
                        continue
                    rows.setdefault(key, {"revenue": 0.0, "expense": 0.0})
                    rows[key]["revenue"] += float(r.get("revenue", 0.0))
                    rows[key]["expense"] += float(r.get("expense", 0.0))
                rows = [
                    {"bot": k, "revenue": v["revenue"], "expense": v["expense"]}
                    for k, v in rows.items()
                ]
            router = GLOBAL_ROUTER or init_db_router("default")
            conn = router.get_connection("models")
            for row in rows:
                roi = float(row.get("revenue", 0.0) - row.get("expense", 0.0))
                bot_name = row.get("bot", "")
                if not bot_name:
                    continue
                conn.execute(
                    "UPDATE models SET current_roi = ? WHERE name LIKE ?",
                    (roi, f"%{bot_name}%"),
                )
            conn.commit()
        except Exception as exc:
            logger.exception("failed to update ROIs: %s", exc)

    def _roi_loop(self, models_db: Path, interval: float) -> None:
        while not self._roi_stop:
            try:
                self.update_rois(models_db)
            except Exception as exc:
                logger.exception("ROI updater failed: %s", exc)
            time.sleep(interval)

    def start_roi_updater(
        self, models_db: Path = DB_PATH, interval: float | None = None
    ) -> None:
        if self._roi_thread and self._roi_thread.is_alive():
            return
        self._roi_stop = False
        logger.info(
            "starting ROI updater interval=%s models_db=%s",
            interval if interval is not None else self.config.roi_interval,
            models_db,
        )
        self._roi_thread = threading.Thread(
            target=self._roi_loop,
            args=(
                models_db,
                interval if interval is not None else self.config.roi_interval,
            ),
            daemon=True,
        )
        self._roi_thread.start()

    def stop_roi_updater(self) -> None:
        self._roi_stop = True
        if self._roi_thread:
            logger.info("stopping ROI updater")
            self._roi_thread.join(timeout=0)
            self._roi_thread = None

    def roi_updater_running(self) -> bool:
        """Return ``True`` when the ROI updater thread is alive."""
        return bool(self._roi_thread and self._roi_thread.is_alive())

    def energy_score(
        self,
        load: float,
        success_rate: float,
        deploy_eff: float,
        failure_rate: float,
        *,
        external: Optional[Dict[str, float]] = None,
        reward: Optional[float] = None,
    ) -> float:
        assert 0.0 <= load <= 1.0, "load must be between 0 and 1"
        assert 0.0 <= success_rate <= 1.0, "success_rate must be between 0 and 1"
        assert 0.0 <= deploy_eff <= 1.0, "deploy_eff must be between 0 and 1"
        assert 0.0 <= failure_rate <= 1.0, "failure_rate must be between 0 and 1"
        self.update_rois()
        capital = self.profit() or 0.0
        prev_profit = self.last_profit
        trend = self.profit_trend()
        current_profit = self.last_profit
        if external is None:
            external = {}
            if self.pathway_db:
                try:
                    top = self.pathway_db.top_pathways(1)
                    if top:
                        external["myelination"] = float(top[0][1]) / 10.0
                except Exception as exc:
                    logger.exception("failed reading pathways: %s", exc)
            if self.data_bot:
                try:
                    df = self.data_bot.db.fetch(ENGAGEMENT_FETCH_LIMIT)
                    if not getattr(df, "empty", True):
                        if hasattr(df, "sum"):
                            revenue = float(df["revenue"].sum())
                            expense = float(df["expense"].sum()) or 1.0
                        else:
                            revenue = sum(r.get("revenue", 0.0) for r in df)
                            expense = sum(r.get("expense", 0.0) for r in df) or 1.0
                        external["engagement"] = (revenue - expense) / 100.0
                except Exception as exc:
                    logger.exception("failed loading engagement data: %s", exc)
            if self.trend_predictor:
                try:
                    tp = self.trend_predictor.predict_future_metrics(1)
                    external["market"] = float(tp.roi) / max(
                        self.config.capital_norm_factor, 1.0
                    )
                    external.setdefault("volatility", abs(float(tp.errors)))
                except Exception as exc:
                    logger.exception("trend predictor signal failed: %s", exc)
            for adapter in self.signal_adapters:
                try:
                    external.update(adapter())
                except Exception as exc:
                    logger.exception("signal adapter failed: %s", exc)
        base = self.engine.compute(
            capital,
            trend,
            load,
            success_rate,
            deploy_eff,
            failure_rate,
            external_signals=external,
            reward=reward if reward is not None else (current_profit - prev_profit),
        )

        # simple binary signals for additional weighting
        trend_sig = 1.0 if trend >= self.config.trend_threshold else 0.0
        vol_val = external.get("volatility")
        if vol_val is None:
            logger.warning("volatility data missing for energy score")
            self.send_alert("volatility metric missing")
            vol_sig = 0.0
        else:
            vol_sig = 1.0 if vol_val <= self.config.volatility_threshold else 0.0
        capital_sig = 1.0 if capital >= self.config.capital_norm_factor else 0.0

        bw = self.config.binary_weights
        weighted_binary = (
            bw.get("trend", 0.4) * trend_sig
            + bw.get("capital", 0.3) * capital_sig
            + bw.get("volatility", 0.3) * vol_sig
        )

        dyn_score = dynamic_weighted_energy_score(
            capital,
            trend,
            external.get("volatility", 0.0),
            self.config.risk_profile,
            external.get("performance_stddev", 0.0),
            norm_factor=self.config.capital_norm_factor,
            weights=self.config.dynamic_weights,
            log_func=self.engine._logistic,
            clamp=self.engine._clamp,
            risk_multipliers=self.config.risk_multipliers,
        )

        base = (base + dyn_score + weighted_binary) / 3.0
        feats = [capital, trend, load, success_rate, deploy_eff, failure_rate]
        result = self._apply_prediction_bots(base, feats)
        result = self.engine._clamp(float(result))

        metric_values = [capital, load, success_rate, deploy_eff, 1.0 - failure_rate]
        hist_adjust = calculate_energy_level(
            trend,
            result,
            self.energy_history,
            metrics=metric_values,
            risk_profile=self.config.risk_profile,
        )
        result = self.engine._clamp((result + hist_adjust) / 2.0)
        logger.debug(
            (
                "energy_score run_id=%s capital=%s trend=%s load=%s "
                "succ=%s eff=%s fail=%s ext=%s result=%s"
            ),
            self.run_id,
            capital,
            trend,
            load,
            success_rate,
            deploy_eff,
            failure_rate,
            external,
            result,
        )
        self.energy_history.append(result)
        return result

    def decide_state(self, energy: float, trend: float) -> StrategyTier:
        """Return a capital strategy based on energy and trend levels."""
        th = self.config.energy_thresholds
        vlow = th.get("very_low", 0.1)
        low = th.get("low", 0.3)
        med = th.get("medium", 0.6)
        high = th.get("high", 0.8)
        if energy < vlow:
            return StrategyTier.HIBERNATE
        if energy < low:
            return StrategyTier.CONSERVE
        if energy < med:
            return StrategyTier.DEPLOY
        if energy < high:
            return (
                StrategyTier.INVEST
                if trend >= self.config.trend_threshold
                else StrategyTier.DEPLOY
            )
        return StrategyTier.AGGRESSIVE

    def evaluate(self, score: float, prediction: str = "") -> None:
        trend = self.profit_trend()
        self.state = self.decide_state(score, trend)
        alloc_score = calculate_allocation_score(
            score,
            trend,
            risk_profile=self.config.risk_profile,
        )
        self.alloc_ledger.log(
            LedgerEntry(
                entry_type=self.state.value,
                amount=self.profit() * alloc_score,
                description=prediction,
            )
        )
        self._log_decision(score, prediction)
        self._adapt_thresholds()

    def _log_decision(self, score: float, prediction: str) -> None:
        """Append decision details to the decision log file."""
        try:
            import json

            rec = {
                "ts": datetime.utcnow().isoformat(),
                "score": score,
                "state": self.state.value,
                "prediction": prediction,
            }
            path = Path(self.config.decision_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception as exc:  # pragma: no cover - IO issues
            logger.exception("decision log failed: %s", exc)

    def _adapt_thresholds(self) -> None:
        """Adjust thresholds based on recent energy history."""
        if not self.energy_history:
            return
        avg = sum(self.energy_history) / len(self.energy_history)
        diff = avg - 0.5
        step = max(0.0, min(0.2, self.config.threshold_step))
        self.config.conserve_threshold = max(
            0.1, min(0.9, self.config.conserve_threshold + diff * step)
        )
        self.config.aggressive_threshold = max(
            self.config.conserve_threshold + step,
            min(0.95, self.config.aggressive_threshold + diff * step),
        )

    def generate_summary(self, energy: float) -> str:
        """Return a human readable summary for the latest metrics."""
        trend_desc = self.profit_trend_description()
        summary = (
            f"Run {self.run_id} | Energy {energy:.3f}\n"
            f"Capital: {self.profit():.2f}\n"
            f"Trend: {self.profit_trend():.4f}\n"
            f"State: {self.state.value}\n"
            f"{trend_desc}"
        )
        return summary

    def send_summary(self, energy: float) -> bool:
        """Send current summary to Discord and log to the summary DB."""
        summary = self.generate_summary(energy)
        if hasattr(self, "summary_db") and self.summary_db is not None:
            try:
                self.summary_db.log_summary(
                    SummaryRecord(
                        run_id=self.run_id,
                        capital=self.profit(),
                        trend=self.profit_trend(),
                        energy=energy,
                        message=summary,
                    )
                )
            except Exception as exc:
                logger.exception("summary logging failed: %s", exc)
        return self.send_alert(summary)

    def allocation_curve(self, score: float) -> float:
        import math

        factor = risk_multiplier(
            0.0, self.config.risk_profile, self.config.risk_multipliers
        )
        import os as _os
        k = float(_os.environ.get("CM_ALLOC_CURVE_K", "5.0"))
        return 1.0 / (1.0 + math.exp(-k * (score - 0.5))) * factor

    def info_ratio(self, energy: float) -> float:
        base = self.config.info_ratio_base.get(self.state.value, 1.0)
        risk = risk_multiplier(
            0.0, self.config.risk_profile, self.config.risk_multipliers
        )
        ratio = base * (1.0 + energy * 0.1) * risk
        return max(0.1, min(ratio, 3.0))


_pipeline_instance: "ModelAutomationPipeline | None" = None
evolution_orchestrator: "EvolutionOrchestrator | None" = None
_manager_lock = threading.Lock()
_manager_retry_timer: threading.Timer | None = None
_manager_retry_count = 0
_MANAGER_MAX_RETRIES = int(os.environ.get("CM_MANAGER_MAX_RETRIES", "5"))
_MANAGER_RETRY_DELAY = float(os.environ.get("CM_MANAGER_RETRY_DELAY", "0.75"))


def _load_pipeline_instance() -> "ModelAutomationPipeline | None":
    """Instantiate :class:`ModelAutomationPipeline` lazily."""

    global _pipeline_instance
    if _pipeline_instance is not None:
        return _pipeline_instance
    pipeline_cls = _get_pipeline_cls()
    if pipeline_cls is None:
        return None
    try:
        _pipeline_instance = pipeline_cls(context_builder=_get_context_builder())
    except Exception as exc:  # pragma: no cover - degraded bootstrap
        logger.warning(
            "ModelAutomationPipeline initialisation failed for CapitalManagementBot: %s",
            exc,
        )
        _pipeline_instance = None
    return _pipeline_instance


def _load_evolution_orchestrator() -> "EvolutionOrchestrator | None":
    """Return a lazily constructed evolution orchestrator."""

    global evolution_orchestrator
    if evolution_orchestrator is not None:
        return evolution_orchestrator
    try:  # pragma: no cover - orchestrator optional
        evolution_orchestrator = get_orchestrator(
            "CapitalManagementBot", _get_data_bot(), _get_engine()
        )
    except Exception as exc:
        logger.warning(
            "EvolutionOrchestrator unavailable for CapitalManagementBot: %s",
            exc,
        )
        evolution_orchestrator = None
    return evolution_orchestrator


def _load_thresholds() -> "SelfCodingThresholds | None":
    """Fetch persisted thresholds for :class:`CapitalManagementBot`."""

    try:
        thresholds = get_thresholds("CapitalManagementBot")
    except Exception as exc:  # pragma: no cover - degraded bootstrap
        logger.warning("failed to load CapitalManagementBot thresholds: %s", exc)
        return None
    if thresholds is None:
        return None
    try:  # pragma: no cover - best effort persistence
        from .data_bot import persist_sc_thresholds as _persist_sc_thresholds

        _persist_sc_thresholds(
            "CapitalManagementBot",
            roi_drop=thresholds.roi_drop,
            error_increase=thresholds.error_increase,
            test_failure_increase=thresholds.test_failure_increase,
        )
    except Exception as exc:
        logger.warning(
            "failed to persist CapitalManagementBot thresholds: %s",
            exc,
        )
    return thresholds


def _initialise_self_coding_manager(*, retry: bool = True) -> None:
    """Bootstrap the self-coding manager once dependencies are available."""

    global manager, _manager_retry_timer, _manager_retry_count
    if manager is not None:
        return
    with _manager_lock:
        if manager is not None:
            return
        if self_coding_import_depth():
            logger.debug(
                "deferring CapitalManagementBot self-coding bootstrap; import depth=%s",
                self_coding_import_depth(),
            )
            if retry:
                _schedule_manager_retry()
            return
        pipeline = _load_pipeline_instance()
        thresholds = _load_thresholds()
        if pipeline is None or thresholds is None:
            if retry:
                _schedule_manager_retry()
            return
        orchestrator = _load_evolution_orchestrator()
        registry_obj = _get_registry()
        data_bot_obj = _get_data_bot()
        try:
            manager_local = internalize_coding_bot(
                "CapitalManagementBot",
                _get_engine(),
                pipeline,
                data_bot=data_bot_obj,
                bot_registry=registry_obj,
                evolution_orchestrator=orchestrator,
                threshold_service=ThresholdService(),
                roi_threshold=thresholds.roi_drop,
                error_threshold=thresholds.error_increase,
                test_failure_threshold=thresholds.test_failure_increase,
            )
        except Exception as exc:  # pragma: no cover - degraded bootstrap
            logger.warning(
                "failed to initialise self-coding manager for CapitalManagementBot: %s",
                exc,
            )
            if retry:
                _schedule_manager_retry()
            return
        manager = manager_local
        if _manager_retry_timer is not None:
            try:
                _manager_retry_timer.cancel()
            except Exception:  # pragma: no cover - best effort
                logger.debug(
                    "unable to cancel pending manager retry timer", exc_info=True
                )
            finally:
                _manager_retry_timer = None
        _manager_retry_count = 0
        try:
            capital_cls = _get_capital_management_bot_class()
            setattr(capital_cls, "manager", manager_local)
        except Exception:  # pragma: no cover - defensive
            logger.debug(
                "unable to bind self-coding manager to CapitalManagementBot", exc_info=True
            )
        try:
            module_path: str | Path
            try:
                module_path = Path(__file__).resolve()
            except Exception:  # pragma: no cover - filesystem edge cases
                module_path = __file__
            registry_obj.register_bot(
                "CapitalManagementBot",
                roi_threshold=getattr(thresholds, "roi_drop", None),
                error_threshold=getattr(thresholds, "error_increase", None),
                test_failure_threshold=getattr(
                    thresholds, "test_failure_increase", None
                ),
                manager=manager_local,
                data_bot=data_bot_obj,
                module_path=module_path,
                is_coding_bot=True,
            )
        except Exception:  # pragma: no cover - registry refresh best effort
            logger.warning(
                "CapitalManagementBot registry refresh failed during self-coding bootstrap",
                exc_info=True,
            )


def _schedule_manager_retry(delay: float | None = None) -> None:
    """Schedule a retry for :func:`_initialise_self_coding_manager`."""

    global _manager_retry_timer, _manager_retry_count
    if manager is not None:
        return
    if _manager_retry_count >= _MANAGER_MAX_RETRIES:
        return
    if _manager_retry_timer is not None and _manager_retry_timer.is_alive():
        return
    _manager_retry_count += 1
    retry_delay = _MANAGER_RETRY_DELAY if delay is None else delay

    def _retry() -> None:
        _initialise_self_coding_manager(retry=True)

    timer = threading.Timer(retry_delay, _retry)
    timer.daemon = True
    _manager_retry_timer = timer
    timer.start()


class _TruthyManagerProxy:
    """Wrapper that keeps fallback managers truthy for registry checks."""

    __slots__ = ("_manager",)

    def __init__(self, manager_obj: object) -> None:
        self._manager = manager_obj

    def __getattr__(self, name: str) -> object:
        return getattr(self._manager, name)

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return True


def _resolve_decorator_manager() -> object | None:
    """Return a manager placeholder so registration avoids retry loops."""

    if manager is not None:
        return manager
    try:
        existing = _get_registry().graph.nodes.get("CapitalManagementBot", {})
        helper = existing.get("selfcoding_manager") or existing.get("manager")
        if helper is not None:
            return helper
    except Exception:  # pragma: no cover - best effort lookup
        logger.debug("CapitalManagementBot manager lookup failed", exc_info=True)

    try:
        disabled = _DisabledSelfCodingManager(
            bot_registry=_get_registry(), data_bot=_get_data_bot()
        )
    except Exception:  # pragma: no cover - defensive
        logger.debug(
            "CapitalManagementBot fallback manager initialisation failed",
            exc_info=True,
        )
        return None
    return _TruthyManagerProxy(disabled)


def bootstrap_capital_management_self_coding() -> "SelfCodingManager | None":
    """Public entry point to trigger self-coding bootstrap."""

    _initialise_self_coding_manager(retry=False)
    return manager


_CapitalManagementBot = cast(
    type[_CapitalManagementBot],
    ensure_cooperative_init(cast(type, _CapitalManagementBot), logger=logger),
)

_capital_bot_class: type[_CapitalManagementBot] | None = None


def _get_capital_management_bot_class() -> type[_CapitalManagementBot]:
    """Return the decorated :class:`CapitalManagementBot` implementation."""

    global _capital_bot_class
    if _capital_bot_class is None:
        decorator_manager = _resolve_decorator_manager()
        decorated_cls = self_coding_managed(
            bot_registry=_get_registry(),
            data_bot=_get_data_bot(),
            manager=decorator_manager,
        )(_CapitalManagementBot)
        cooperative_cls = ensure_cooperative_init(cast(type, decorated_cls))
        guard_state = getattr(cooperative_cls, "__cooperative_guard__", False)
        if guard_state:
            logger.debug(
                "[init-guard] Cooperative init guard active for %s", cooperative_cls.__name__
            )
        else:
            logger.warning(
                "[init-guard] Cooperative init guard missing for %s", cooperative_cls.__name__
            )
        _capital_bot_class = cast(type[_CapitalManagementBot], cooperative_cls)
    return _capital_bot_class


def __getattr__(name: str) -> object:
    if name == "CapitalManagementBot":
        cls = _get_capital_management_bot_class()
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - trivial
    return sorted(set(globals()) | {"CapitalManagementBot"})


if TYPE_CHECKING:
    CapitalManagementBot = _CapitalManagementBot


_schedule_manager_retry()


__all__ = [
    "LedgerEntry",
    "CapitalLedger",
    "CapitalAllocationLedger",
    "CapitalMetrics",
    "ROIEvent",
    "ROIEventDB",
    "ProfitRecord",
    "ProfitHistoryDB",
    "SummaryRecord",
    "SummaryHistoryDB",
    "EnergyScoreEngine",
    "dynamic_weighted_energy_score",
    "calculate_energy_level",
    "calculate_allocation_score",
    "risk_multiplier",
    "StrategyTier",
    "fetch_capital_metrics_async",
    "fetch_metric_from_db",
    "calculate_trend_from_logs",
    "get_bot_performance_score",
    "get_capital_metrics",
    "CapitalManagementConfig",
    "CapitalManagementBot",
    "bootstrap_capital_management_self_coding",
]
