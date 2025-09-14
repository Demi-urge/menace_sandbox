# flake8: noqa
"""Revenue Signal Amplifier and tracking components."""

from __future__ import annotations

import threading
import logging
from dataclasses import dataclass
import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple, Optional

from .unified_event_bus import UnifiedEventBus
from .retry_utils import publish_with_retry
from .db_router import GLOBAL_ROUTER, init_db_router
from .scope_utils import Scope, build_scope_clause
from .coding_bot_interface import self_coding_managed

router = GLOBAL_ROUTER or init_db_router("revenue_amplifier")

logger = logging.getLogger(__name__)


class _StubRegistry:
    def register_bot(self, *args, **kwargs) -> None:  # pragma: no cover - stub
        return None

    def update_bot(self, *args, **kwargs) -> None:  # pragma: no cover - stub
        return None


class _StubDataBot:
    def reload_thresholds(self, _name: str):  # pragma: no cover - stub
        return type("_T", (), {})()


_REGISTRY_STUB = _StubRegistry()
_DATA_BOT_STUB = _StubDataBot()


@dataclass
class RevenueEvent:
    model_id: str
    amount: float
    platform: str
    segment: str
    ts: str = datetime.utcnow().isoformat()


@dataclass
class SubscriptionRecord:
    model_id: str
    user_id: str
    status: str  # active or cancelled
    ts: str = datetime.utcnow().isoformat()


@dataclass
class ChurnEvent:
    model_id: str
    user_id: str
    reason: str
    ts: str = datetime.utcnow().isoformat()


@dataclass
class LeadRecord:
    model_id: str
    lead_id: str
    converted: bool
    ts: str = datetime.utcnow().isoformat()

class RevenueEventsDB:
    """SQLite store for revenue events."""

    def __init__(self, path: Path | str = "revenue_events.db", *, event_bus: Optional[UnifiedEventBus] = None) -> None:
        # Allow usage across threads since bots may share this DB connection
        self.conn = router.get_connection("revenue")
        self.event_bus = event_bus
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS revenue(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                amount REAL,
                platform TEXT,
                segment TEXT,
                ts TEXT
            )
            """
        )
        self.conn.commit()

    def add(self, ev: RevenueEvent) -> int:
        cur = self.conn.execute(
            "INSERT INTO revenue(model_id, amount, platform, segment, ts) VALUES(?,?,?,?,?)",
            (ev.model_id, ev.amount, ev.platform, ev.segment, ev.ts),
        )
        self.conn.commit()
        rid = int(cur.lastrowid)
        if self.event_bus:
            if not publish_with_retry(self.event_bus, "revenue:new", dataclasses.asdict(ev) | {"id": rid}):
                logger.exception("failed to publish revenue:new event")
        return rid

    def fetch(self, model_id: str, limit: int = 50) -> List[Tuple[str, float, str, str, str]]:
        cur = self.conn.execute(
            "SELECT model_id, amount, platform, segment, ts FROM revenue WHERE model_id=? ORDER BY id DESC LIMIT ?",
            (model_id, limit),
        )
        return cur.fetchall()

class SubscriptionDB:
    """SQLite store for subscription history."""

    def __init__(self, path: Path | str = "subscriptions.db", *, event_bus: Optional[UnifiedEventBus] = None) -> None:
        # DB may be accessed from different threads
        self.conn = router.get_connection("subs")
        self.event_bus = event_bus
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS subs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                user_id TEXT,
                status TEXT,
                ts TEXT
            )
            """
        )
        self.conn.commit()

    def add(self, rec: SubscriptionRecord) -> int:
        cur = self.conn.execute(
            "INSERT INTO subs(model_id, user_id, status, ts) VALUES(?,?,?,?)",
            (rec.model_id, rec.user_id, rec.status, rec.ts),
        )
        self.conn.commit()
        sid = int(cur.lastrowid)
        if self.event_bus:
            if not publish_with_retry(self.event_bus, "subs:new", dataclasses.asdict(rec) | {"id": sid}):
                logger.exception("failed to publish subs:new event")
        return sid

    def active_count(self, model_id: str) -> int:
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM subs WHERE model_id=? AND status='active'",
            (model_id,),
        )
        row = cur.fetchone()
        return int(row[0] or 0)

class ChurnDB:
    """SQLite store for churn events."""

    def __init__(self, path: Path | str = "churn.db", *, event_bus: Optional[UnifiedEventBus] = None) -> None:
        # Connection shared across threads
        self.conn = router.get_connection("churn")
        self.event_bus = event_bus
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS churn(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_menace_id TEXT NOT NULL,
                model_id TEXT,
                user_id TEXT,
                reason TEXT,
                ts TEXT
            )
            """
        )
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(churn)")]
        if "source_menace_id" not in cols:
            self.conn.execute(
                "ALTER TABLE churn ADD COLUMN source_menace_id TEXT NOT NULL DEFAULT ''"
            )
            self.conn.execute(
                "UPDATE churn SET source_menace_id=? WHERE source_menace_id=''",
                (router.menace_id,),
            )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_churn_source_menace_id ON churn(source_menace_id)"
        )
        self.conn.commit()

    def add(self, ev: ChurnEvent, *, source_menace_id: str | None = None) -> int:
        menace_id = source_menace_id or router.menace_id
        cur = self.conn.execute(
            "INSERT INTO churn(source_menace_id, model_id, user_id, reason, ts) VALUES(?,?,?,?,?)",
            (menace_id, ev.model_id, ev.user_id, ev.reason, ev.ts),
        )
        self.conn.commit()
        cid = int(cur.lastrowid)
        if self.event_bus:
            payload = dataclasses.asdict(ev) | {"id": cid, "source_menace_id": menace_id}
            if not publish_with_retry(self.event_bus, "churn:new", payload):
                logger.exception("failed to publish churn:new event")
        return cid

    def fetch_recent(
        self,
        model_id: str,
        limit: int = 20,
        *,
        scope: Scope | str = Scope.ALL,
        source_menace_id: Any | None = None,
    ) -> List[Tuple[str, str, str, str]]:
        menace_id = source_menace_id or router.menace_id
        clause, params = build_scope_clause("churn", scope, menace_id)
        sql = "SELECT model_id, user_id, reason, ts FROM churn WHERE model_id=?"
        query_params: list[Any] = [model_id]
        if clause:
            sql += f" AND {clause}"
            query_params.extend(params)
        sql += " ORDER BY id DESC LIMIT ?"
        query_params.append(limit)
        cur = self.conn.execute(sql, query_params)
        return cur.fetchall()

class LeadDB:
    """SQLite store for lead conversion events."""

    def __init__(self, path: Path | str = "leads.db", *, event_bus: Optional[UnifiedEventBus] = None) -> None:
        # Use a connection that can be shared across threads
        self.conn = router.get_connection("leads")
        self.event_bus = event_bus
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS leads(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                lead_id TEXT,
                converted INTEGER,
                ts TEXT
            )
            """
        )
        self.conn.commit()

    def add(self, rec: LeadRecord) -> int:
        cur = self.conn.execute(
            "INSERT INTO leads(model_id, lead_id, converted, ts) VALUES(?,?,?,?)",
            (rec.model_id, rec.lead_id, int(rec.converted), rec.ts),
        )
        self.conn.commit()
        lid = int(cur.lastrowid)
        if self.event_bus:
            if not publish_with_retry(self.event_bus, "leads:new", dataclasses.asdict(rec) | {"id": lid}):
                logger.exception("failed to publish leads:new event")
        return lid

    def conversion_rate(self, model_id: str) -> float:
        cur = self.conn.execute(
            "SELECT COUNT(*), SUM(converted) FROM leads WHERE model_id=?",
            (model_id,),
        )
        total, conv = cur.fetchone()
        if not total:
            return 0.0
        return float(conv or 0) / float(total)

class ProfitabilityDB:
    """Track profitability over time."""

    def __init__(self, path: Path | str = "profitability.db", *, event_bus: Optional[UnifiedEventBus] = None) -> None:
        # Share connection across threads safely
        self.conn = router.get_connection("profit")
        self.lock = threading.Lock()
        self.event_bus = event_bus
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profit(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                revenue REAL,
                ts TEXT
            )
            """
        )
        self.conn.commit()

    def add(self, model_id: str, revenue: float, ts: str | None = None) -> int:
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO profit(model_id, revenue, ts) VALUES(?,?,?)",
                (model_id, revenue, ts or datetime.utcnow().isoformat()),
            )
            self.conn.commit()
        pid = int(cur.lastrowid)
        if self.event_bus:
            payload = {
                "id": pid,
                "model_id": model_id,
                "revenue": revenue,
                "ts": ts or datetime.utcnow().isoformat(),
            }
            if not publish_with_retry(self.event_bus, "profit:new", payload):
                logger.exception("failed to publish profit:new event")
        return pid

class SalesSpikeMonitor:
    """Log sales from bots or webhooks."""

    def __init__(self, db: RevenueEventsDB | None = None) -> None:
        self.db = db or RevenueEventsDB()

    def record_sale(self, model_id: str, amount: float, platform: str, segment: str) -> None:
        self.db.add(RevenueEvent(model_id, amount, platform, segment))


class SubscriptionHealthTracker:
    """Track subscriptions and churn velocity."""

    def __init__(self, db: SubscriptionDB | None = None) -> None:
        self.db = db or SubscriptionDB()

    def update_status(self, model_id: str, user_id: str, status: str) -> None:
        self.db.add(SubscriptionRecord(model_id, user_id, status))

    def active_subscriptions(self, model_id: str) -> int:
        return self.db.active_count(model_id)


class ChurnDetector:
    """Monitor churn events."""

    def __init__(self, db: ChurnDB | None = None) -> None:
        self.db = db or ChurnDB()

    def log_churn(self, model_id: str, user_id: str, reason: str) -> None:
        self.db.add(ChurnEvent(model_id, user_id, reason))

    def recent(self, model_id: str, limit: int = 20) -> List[Tuple[str, str, str, str]]:
        return self.db.fetch_recent(model_id, limit)


class LeadPerformanceMonitor:
    """Observe lead conversion speed."""

    def __init__(self, db: LeadDB | None = None) -> None:
        self.db = db or LeadDB()

    def record_lead(self, model_id: str, lead_id: str, converted: bool) -> None:
        self.db.add(LeadRecord(model_id, lead_id, converted))

    def conversion_rate(self, model_id: str) -> float:
        return self.db.conversion_rate(model_id)

@self_coding_managed(bot_registry=_REGISTRY_STUB, data_bot=_DATA_BOT_STUB)
class RevenueSpikeEvaluatorBot:
    """Detect revenue surges using exponential weighting."""

    def __init__(self, db: RevenueEventsDB, window: int = 20, threshold: float = 3.0) -> None:
        self.db = db
        self.window = window
        self.threshold = threshold

    def _stats(self, model_id: str) -> Tuple[float, float, float]:
        rows = self.db.fetch(model_id, self.window)
        if not rows:
            return 0.0, 0.0, 0.0
        amounts = [r[1] for r in reversed(rows)]
        weight = 0.9
        avg = 0.0
        for amt in amounts:
            avg = weight * avg + (1 - weight) * amt
        mean = sum(amounts) / len(amounts)
        var = sum((a - mean) ** 2 for a in amounts) / max(len(amounts) - 1, 1)
        std = var ** 0.5
        last = amounts[-1]
        return last, avg, std

    def detect_spike(self, model_id: str) -> bool:
        last, avg, std = self._stats(model_id)
        if std == 0.0:
            return False
        return last > avg + self.threshold * std

@self_coding_managed(bot_registry=_REGISTRY_STUB, data_bot=_DATA_BOT_STUB)
class CapitalAllocationBot:
    """Rebalance resources to favour surging models."""

    def __init__(self, profit_db: ProfitabilityDB | None = None, enh_db: ProfitabilityDB | None = None) -> None:
        self.profit_db = profit_db or ProfitabilityDB()
        self.enh_db = enh_db or ProfitabilityDB()

    def rebalance(self, model_id: str, revenue: float) -> None:
        self.profit_db.add(model_id, revenue)
        self.enh_db.add(model_id, revenue)

__all__ = [
    "RevenueEvent",
    "SubscriptionRecord",
    "ChurnEvent",
    "LeadRecord",
    "RevenueEventsDB",
    "SubscriptionDB",
    "ChurnDB",
    "LeadDB",
    "ProfitabilityDB",
    "SalesSpikeMonitor",
    "SubscriptionHealthTracker",
    "ChurnDetector",
    "LeadPerformanceMonitor",
    "RevenueSpikeEvaluatorBot",
    "CapitalAllocationBot",
]
