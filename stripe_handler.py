"""Unified Stripe API helper with optional mocking."""

from __future__ import annotations

import logging
import os
from typing import Any

from .config_loader import get_config_value

try:
    import stripe  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    stripe = None  # type: ignore
    logging.getLogger(__name__).warning("stripe library unavailable: %s", exc)

logger = logging.getLogger(__name__)


class _MockCharge:
    @staticmethod
    def create(**kwargs: Any) -> dict[str, Any]:
        logger.info("Mock Stripe charge: %s", kwargs)
        return {"id": "mock_charge", "status": "succeeded"}


class _MockBalance:
    @staticmethod
    def retrieve() -> dict[str, Any]:
        logger.info("Mock Stripe balance retrieval")
        return {"available": [{"amount": 0}]}


class _MockStripe:
    api_key: str = ""
    Charge = _MockCharge
    Balance = _MockBalance


def _bool(val: Any) -> bool:
    if isinstance(val, str):
        return val.lower() in {"1", "true", "yes", "on"}
    return bool(val)


def is_enabled() -> bool:
    """Return True if Stripe usage is enabled via config or environment."""
    env = os.getenv("STRIPE_ENABLED")
    try:
        cfg = get_config_value("stripe_enabled", env if env is not None else False)
    except Exception:
        cfg = env if env is not None else False
    return _bool(cfg)


def _use_mock(test_mode: bool) -> bool:
    return test_mode or os.getenv("MENACE_MODE", "test").lower() != "production"


def _client(api_key: str, test_mode: bool):
    if not is_enabled() or not api_key:
        return None
    if stripe is None or _use_mock(test_mode):
        return _MockStripe()
    stripe.api_key = api_key
    return stripe


def get_balance(api_key: str, *, test_mode: bool = False) -> float:
    client = _client(api_key, test_mode)
    if client is None:
        logger.info("Stripe ROI check skipped: No API key or disabled")
        return 0.0
    try:
        bal = client.Balance.retrieve()
        amount = bal["available"][0]["amount"] / 100.0
        if amount == 0:
            logger.info("Stripe ROI check skipped: No funds available")
        return float(amount)
    except Exception as exc:  # pragma: no cover - network/API issues
        logger.exception("Stripe balance retrieval failed: %s", exc)
        return 0.0


def charge(amount: float, api_key: str, *, description: str = "", test_mode: bool = False) -> str:
    client = _client(api_key, test_mode)
    if client is None:
        logger.info("Stripe charge skipped: disabled or missing API key")
        return "stripe_unavailable"
    params = {"amount": int(amount * 100), "currency": "usd", "description": description}
    if _use_mock(test_mode):
        params["source"] = "tok_visa"
    try:
        client.Charge.create(**params)
        return "success"
    except Exception as exc:  # pragma: no cover - network/API issues
        logger.exception("Stripe charge failed: %s", exc)
        return f"error:{exc}"


__all__ = ["is_enabled", "get_balance", "charge"]
