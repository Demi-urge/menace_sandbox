"""Autonomous Offer Testing Bot for dynamic monetization experiments."""

from __future__ import annotations

from .coding_bot_interface import self_coding_managed
import sqlite3
from dataclasses import dataclass
import dataclasses
from datetime import datetime
from typing import Any, List, Dict, Optional
from pathlib import Path
import logging

from db_router import GLOBAL_ROUTER

from .unified_event_bus import UnifiedEventBus
from .menace_memory_manager import MenaceMemoryManager, MemoryEntry

logger = logging.getLogger(__name__)


@dataclass
class OfferVariant:
    """Single offer variation deployed to a customer segment."""

    product: str
    price: float
    bundle: str
    upsell_chain: str
    landing_page: str
    active: bool = True
    id: int | None = None


@dataclass
class OfferInteraction:
    """Metrics captured for a user interaction with an offer."""

    variant_id: int
    converted: bool
    order_value: float
    retained: bool
    refund: bool
    latency: float
    ts: str = datetime.utcnow().isoformat()


class OfferDB:
    """SQLite-backed store for offers and interactions."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        event_bus: Optional[UnifiedEventBus] = None,
    ) -> None:
        # allow the connection to be used across threads since OfferTestingBot
        # may be accessed from a thread pool in ``ModelAutomationPipeline``
        if GLOBAL_ROUTER is None:
            raise RuntimeError("Database router is not initialised")
        with GLOBAL_ROUTER.get_connection("variants") as conn:
            self.conn = conn
        self.event_bus = event_bus
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS variants(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product TEXT,
                price REAL,
                bundle TEXT,
                upsell_chain TEXT,
                landing_page TEXT,
                active INTEGER
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                variant_id INTEGER,
                converted INTEGER,
                order_value REAL,
                retained INTEGER,
                refund INTEGER,
                latency REAL,
                ts TEXT,
                FOREIGN KEY(variant_id) REFERENCES variants(id)
            )
            """
        )
        self.conn.commit()

    def add_variant(self, var: OfferVariant) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO variants(product, price, bundle, upsell_chain, landing_page, active)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (var.product, var.price, var.bundle, var.upsell_chain, var.landing_page, 1 if var.active else 0),
        )
        self.conn.commit()
        vid = int(cur.lastrowid)
        if self.event_bus:
            try:
                payload = dataclasses.asdict(var)
                payload["id"] = vid
                self.event_bus.publish("variants:new", payload)
            except Exception as exc:
                logger.error("failed publishing variants:new event: %s", exc)
        return vid

    def update_variant(self, variant_id: int, **fields: Any) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        params = list(fields.values()) + [variant_id]
        self.conn.execute(f"UPDATE variants SET {sets} WHERE id=?", params)
        self.conn.commit()
        if self.event_bus:
            try:
                payload = {"variant_id": variant_id, **fields}
                self.event_bus.publish("variants:update", payload)
            except Exception as exc:
                logger.error("failed publishing variants:update event: %s", exc)

    def list_variants(self, active: Optional[bool] = None) -> List[OfferVariant]:
        cur = self.conn.execute(
            "SELECT id, product, price, bundle, upsell_chain, landing_page, active FROM variants"
            + (" WHERE active = ?" if active is not None else ""),
            (1 if active else 0,) if active is not None else (),
        )
        rows = cur.fetchall()
        return [
            OfferVariant(
                id=row[0],
                product=row[1],
                price=row[2],
                bundle=row[3],
                upsell_chain=row[4],
                landing_page=row[5],
                active=bool(row[6]),
            )
            for row in rows
        ]

    def log_interaction(self, inter: OfferInteraction) -> None:
        self.conn.execute(
            """
            INSERT INTO interactions(variant_id, converted, order_value, retained, refund, latency, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inter.variant_id,
                1 if inter.converted else 0,
                inter.order_value,
                1 if inter.retained else 0,
                1 if inter.refund else 0,
                inter.latency,
                inter.ts,
            ),
        )
        self.conn.commit()
        if self.event_bus:
            try:
                self.event_bus.publish(
                    "interactions:new", dataclasses.asdict(inter)
                )
            except Exception as exc:
                logger.error("failed publishing interactions:new event: %s", exc)

    def variant_stats(self, variant_id: int) -> Dict[str, float]:
        cur = self.conn.execute(
            """
            SELECT AVG(converted), AVG(order_value), AVG(retained), AVG(refund), AVG(latency)
            FROM interactions WHERE variant_id = ?
            """,
            (variant_id,),
        )
        row = cur.fetchone()
        if not row:
            return {}
        return {
            "conversion_rate": row[0] or 0.0,
            "avg_order_value": row[1] or 0.0,
            "retention_rate": row[2] or 0.0,
            "refund_rate": row[3] or 0.0,
            "avg_latency": row[4] or 0.0,
        }


@self_coding_managed
class OfferTestingBot:
    """Generate offer variations, deploy them and select winners."""

    def __init__(
        self,
        db: OfferDB | None = None,
        *,
        event_bus: Optional[UnifiedEventBus] = None,
        memory_mgr: MenaceMemoryManager | None = None,
    ) -> None:
        self.db = db or OfferDB(event_bus=event_bus)
        self.event_bus = event_bus
        self.memory_mgr = memory_mgr
        self.last_variant_event: object | None = None
        self.last_interaction_event: object | None = None
        if self.event_bus:
            try:
                self.event_bus.subscribe("variants:new", self._on_variant_event)
                self.event_bus.subscribe(
                    "interactions:new", self._on_interaction_event
                )
            except Exception as exc:
                logger.error(
                    "failed subscribing to offer events: %s",
                    exc,
                )
        if self.memory_mgr:
            try:
                self.memory_mgr.subscribe(self._on_memory_entry)
            except Exception as exc:
                logger.exception("memory manager subscribe failed: %s", exc)

    def _on_variant_event(self, topic: str, payload: object) -> None:
        self.last_variant_event = payload
        try:
            if self.memory_mgr:
                self.memory_mgr.store("offer_variant", payload, tags="offer")
        except Exception as exc:
            logger.exception("memory store failed: %s", exc)

    def _on_interaction_event(self, topic: str, payload: object) -> None:
        self.last_interaction_event = payload

    def _on_memory_entry(self, entry: MemoryEntry) -> None:
        if "offer" in (entry.tags or "").lower():
            self.last_interaction_event = entry

    # ------------------------------------------------------------------
    # Variation generation
    # ------------------------------------------------------------------
    def generate_variations(self, product: str, base_price: float) -> List[int]:
        """Create offer variants with different pricing and bundles."""
        variations = []
        price_points = [base_price * 0.9, base_price, base_price * 1.1]
        for price in price_points:
            psych = round(price) - 0.01
            for landing in ["A", "B"]:
                var = OfferVariant(
                    product=product,
                    price=psych,
                    bundle="standard",
                    upsell_chain="upsell1>upsell2",
                    landing_page=landing,
                )
                vid = self.db.add_variant(var)
                variations.append(vid)
        return variations

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def best_variants(self) -> List[int]:
        """Return active variants sorted by conversion rate."""
        vars = self.db.list_variants(active=True)
        scored = []
        for v in vars:
            stats = self.db.variant_stats(v.id or 0)
            if not stats:
                continue
            score = stats["conversion_rate"] + stats["retention_rate"] - stats["refund_rate"]
            scored.append((score, v.id))
        scored.sort(reverse=True)
        return [v for _, v in scored]

    def promote_winners(self, top_n: int = 1) -> None:
        winners = set(self.best_variants()[:top_n])
        for var in self.db.list_variants():
            active = 1 if var.id in winners else 0
            if var.id is not None:
                self.db.update_variant(var.id, active=active)


__all__ = ["OfferVariant", "OfferInteraction", "OfferDB", "OfferTestingBot"]
