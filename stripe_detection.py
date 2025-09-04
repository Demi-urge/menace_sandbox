"""Shared Stripe detection rules and helpers."""
from __future__ import annotations

from typing import Iterable

PAYMENT_KEYWORDS = {
    "payment",
    "checkout",
    "billing",
    "stripe",
    "invoice",
    "subscription",
    "payout",
    "charge",
}

HTTP_LIBRARIES = {"requests", "httpx"}


def contains_payment_keyword(name: str, keywords: Iterable[str] = PAYMENT_KEYWORDS) -> bool:
    """Return ``True`` if ``name`` includes any payment-related keyword."""
    parts = name.lower().split("_")
    return any(part in keywords for part in parts)
