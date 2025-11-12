"""Investment engine for automated reinvestment via Stripe."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

from db_router import DBRouter, GLOBAL_ROUTER, LOCAL_TABLES, init_db_router
from dynamic_path_router import resolve_path

from . import stripe_billing_router
import logging
from .coding_bot_interface import self_coding_managed

logger = logging.getLogger(__name__)

DEFAULT_BOT_ID = "finance:finance_router_bot"


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
class InvestmentRecord:
    """Record of a reinvestment transaction."""

    amount: float
    predicted_roi: float
    target: str
    cap_used: float
    ts: str = datetime.utcnow().isoformat()


class InvestmentDB:
    """SQLite-backed store of reinvestment history."""

    def __init__(
        self,
        path: Path | str = resolve_path("investment_log.db"),
        router: DBRouter | None = None,
    ) -> None:
        # Allow the database connection to be used across threads. The
        # reinvestment bot may execute database operations from background
        # workers, so SQLite's default same-thread restriction would raise
        # `sqlite3.ProgrammingError` when accessed from a different thread.
        LOCAL_TABLES.add("investments")
        self.router = router or GLOBAL_ROUTER or init_db_router(
            "investment_engine", local_db_path=str(path), shared_db_path=str(path)
        )
        self.conn = self.router.get_connection("investments")
        self.lock = threading.Lock()
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS investments(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                amount REAL,
                predicted_roi REAL,
                target TEXT,
                cap_used REAL,
                ts TEXT
            )
            """,
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_investments_ts ON investments(ts)"
        )
        self.conn.commit()

    def add(self, rec: InvestmentRecord) -> int:
        with self.lock:
            cur = self.conn.execute(
                (
                    "INSERT INTO investments(" "amount, predicted_roi, target, cap_used, ts"
                    ") VALUES(?,?,?,?,?)"
                ),
                (rec.amount, rec.predicted_roi, rec.target, rec.cap_used, rec.ts),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def fetch(self, limit: int = 50) -> List[Tuple[float, float, str, float, str]]:
        with self.lock:
            cur = self.conn.execute(
                (
                    "SELECT amount, predicted_roi, target, cap_used, ts "
                    "FROM investments ORDER BY id DESC LIMIT ?"
                ),
                (limit,),
            )
            rows = cur.fetchall()
            return [
                (float(r[0]), float(r[1]), r[2], float(r[3]), r[4])
                for r in rows
            ]


class PredictiveSpendEngine:
    """Simple spend predictor using historical ROI with linear regression."""

    def __init__(self, history_db: InvestmentDB | None = None) -> None:
        self.history = history_db or InvestmentDB(":memory:")

    def predict(self, balance: float, cap: float) -> Tuple[float, float]:
        """Return predicted optimal spend and ROI."""
        hist = self.history.fetch(20)
        rois = [r[1] for r in hist]

        try:
            import numpy as np  # type: ignore
        except Exception:  # pragma: no cover - optional
            np = None  # type: ignore

        try:
            import pandas as pd  # type: ignore
            from sklearn.ensemble import RandomForestRegressor  # type: ignore
        except Exception:  # pragma: no cover - optional
            pd = None  # type: ignore
            RandomForestRegressor = None  # type: ignore

        predicted_roi: float
        if pd is not None and RandomForestRegressor is not None and len(rois) >= 5:
            try:
                df = pd.DataFrame({"roi": rois})
                X = df.index.values.reshape(-1, 1)
                y = df["roi"].values
                model = RandomForestRegressor(n_estimators=100)
                model.fit(X, y)
                predicted_roi = float(model.predict([[len(rois) + 1]])[0])
            except Exception:
                predicted_roi = sum(rois) / len(rois) if rois else 0.1
        elif np is not None and len(rois) >= 3:
            x = np.arange(len(rois))
            y = np.array(rois)
            slope, intercept = np.polyfit(x, y, 1)
            predicted_roi = float(slope * (len(rois) + 1) + intercept)
        else:
            predicted_roi = sum(rois) / len(rois) if rois else 0.1

        amount = balance * min(cap, 0.1 + predicted_roi)
        return amount, predicted_roi


@self_coding_managed(bot_registry=_REGISTRY_STUB, data_bot=_DATA_BOT_STUB)
class AutoReinvestmentBot:
    """Automate reinvestment decisions and spending."""

    def __init__(
        self,
        cap_percentage: float = 0.5,
        safety_reserve: float = 0.0,
        minimum_threshold: float = 10.0,
        predictor: PredictiveSpendEngine | None = None,
        db: InvestmentDB | None = None,
        bot_id: str = DEFAULT_BOT_ID,
        *,
        manager: "SelfCodingManager | None" = None,
    ) -> None:
        self.cap_percentage = cap_percentage
        self.safety_reserve = safety_reserve
        self.minimum_threshold = minimum_threshold
        self.predictor = predictor or PredictiveSpendEngine()
        self.db = db or InvestmentDB()
        self.bot_id = bot_id
        logger.debug("AutoReinvestmentBot initialized")

    # balance helpers -----------------------------------------------------
    def _current_balance(self) -> float:
        return stripe_billing_router.get_balance(self.bot_id)

    def _execute_spending(self, amount: float) -> str:
        try:
            resp = stripe_billing_router.charge(
                self.bot_id, amount, description="reinvestment"
            )
            status = resp.get("status")
            return "success" if status == "succeeded" else f"error:{status}"
        except Exception as exc:  # pragma: no cover - network/API issues
            logger.exception("Stripe charge failed: %s", exc)
            return f"error:{exc}"

    # core ----------------------------------------------------------------
    def reinvest(self, target: str = "infrastructure") -> float:
        try:
            balance = self._current_balance()
        except RuntimeError as exc:
            logger.exception("Stripe balance retrieval failed: %s", exc)
            raise
        if balance <= 0:
            logger.info("Stripe ROI check skipped: No funds available")
            return 0.0
        reinvestable = balance * self.cap_percentage
        predicted, predicted_roi = self.predictor.predict(balance, self.cap_percentage)
        amount = min(predicted, reinvestable)
        if amount < self.minimum_threshold or balance - amount < self.safety_reserve:
            return 0.0
        result = self._execute_spending(amount)
        if result == "success":
            self.db.add(
                InvestmentRecord(
                    amount=amount,
                    predicted_roi=predicted_roi,
                    target=target,
                    cap_used=self.cap_percentage,
                )
            )
        return amount if result == "success" else 0.0


__all__ = ["InvestmentRecord", "InvestmentDB", "PredictiveSpendEngine", "AutoReinvestmentBot"]
if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .self_coding_manager import SelfCodingManager
else:  # pragma: no cover - runtime fallback when manager is unused
    SelfCodingManager = object  # type: ignore[assignment]
