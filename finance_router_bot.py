"""Finance Router Bot handling payouts via Stripe."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import asyncio
import logging

from dotenv import load_dotenv

from dynamic_path_router import resolve_path

from . import stripe_billing_router

from .capital_management_bot import CapitalManagementBot
from .unified_event_bus import UnifiedEventBus
from .menace_memory_manager import MenaceMemoryManager, MemoryEntry

logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    """Record of a payout transaction."""

    model_id: str
    amount: float
    result: str
    ts: str = datetime.utcnow().isoformat()


class FinanceRouterBot:
    """Route payments and log payouts for Menace."""

    def __init__(
        self,
        payout_log_path: Path | str | None = None,
        capital_manager: Optional[CapitalManagementBot] = None,
        *,
        event_bus: Optional[UnifiedEventBus] = None,
        memory_mgr: MenaceMemoryManager | None = None,
    ) -> None:
        load_dotenv()
        self.capital_manager = capital_manager
        raw_log_path = str(
            payout_log_path
            or os.getenv("PAYOUT_LOG_PATH", "finance_logs/payout_log.json")
        )
        try:
            self.payout_log_path = Path(resolve_path(raw_log_path))
        except FileNotFoundError:
            parent = Path(raw_log_path).parent
            if str(parent) not in {"", "."}:
                try:
                    resolved_parent = resolve_path(str(parent))
                    self.payout_log_path = Path(resolved_parent) / Path(raw_log_path).name
                except FileNotFoundError:
                    self.payout_log_path = Path(raw_log_path)
            else:
                self.payout_log_path = Path(raw_log_path)
        self.payout_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.payout_log_path.exists():
            self.payout_log_path.write_text("[]")
        logger.debug("FinanceRouterBot initialized")
        self.event_bus = event_bus
        self.memory_mgr = memory_mgr
        self.last_transaction_event: object | None = None
        self.last_finance_memory: MemoryEntry | None = None
        if self.event_bus:
            try:
                self.event_bus.subscribe(
                    "transactions:new", self._on_transaction_event
                )
            except Exception as exc:
                logger.error("failed subscribing to transactions:new: %s", exc)
        if self.memory_mgr:
            try:
                self.memory_mgr.subscribe(self._on_memory_entry)
            except Exception as exc:
                logger.error("memory subscription failed: %s", exc)

    def _save(self, records: List[Transaction]) -> None:
        with self.payout_log_path.open("w", encoding="utf-8") as f:
            json.dump([rec.__dict__ for rec in records], f)

    def _load(self) -> List[Transaction]:
        try:
            data = json.loads(self.payout_log_path.read_text())
        except Exception:  # pragma: no cover - malformed
            data = []
        return [Transaction(**d) for d in data]

    def log_transaction(self, model_id: str, amount: float, result: str) -> None:
        records = self._load()
        records.append(Transaction(model_id, amount, result))
        self._save(records)
        if self.event_bus:
            try:
                self.event_bus.publish(
                    "transactions:new",
                    {"model_id": model_id, "amount": amount, "result": result},
                )
            except Exception as exc:
                logger.error("failed publishing transaction event: %s", exc)

    def _on_transaction_event(self, topic: str, payload: object) -> None:
        self.last_transaction_event = payload

    def _on_memory_entry(self, entry: MemoryEntry) -> None:
        if "finance" in (entry.tags or "").lower():
            self.last_finance_memory = entry

    def route_payment(self, amount: float, model_id: str) -> str:
        """Charge via Stripe and log the result."""
        try:
            resp = stripe_billing_router.init_charge(
                model_id,
                amount,
                description=model_id,
            )
            status = resp.get("status")
            result = "success" if status == "succeeded" else f"error:{status}"
        except Exception as exc:  # pragma: no cover - network/API issues
            logger.exception("Stripe charge failed: %s", exc)
            result = f"error:{exc}"
        self.log_transaction(model_id, amount, result)
        if self.capital_manager and result == "success":
            self.capital_manager.log_inflow(amount, model_id)
        return result

    async def route_payment_async(self, amount: float, model_id: str) -> str:
        """Async wrapper for ``route_payment``."""
        return await asyncio.to_thread(self.route_payment, amount, model_id)

    def report_earnings_summary(self) -> Dict[str, float]:
        records = self._load()
        summary: Dict[str, float] = {}
        for rec in records:
            summary[rec.model_id] = summary.get(rec.model_id, 0.0) + rec.amount
        return summary


__all__ = ["Transaction", "FinanceRouterBot"]
