"""Shared Stripe detection rules and helpers."""
from __future__ import annotations

import re
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

# Common HTTP client libraries that can issue requests to Stripe's API.
# Detected usages of these libraries contacting the Stripe service must go
# through ``stripe_billing_router``.
HTTP_LIBRARIES = {"requests", "httpx", "aiohttp", "urllib", "urllib3"}


def contains_payment_keyword(name: str, keywords: Iterable[str] = PAYMENT_KEYWORDS) -> bool:
    """Return ``True`` if ``name`` includes any payment-related keyword."""
    normalized = name.replace("-", "_")
    normalized = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", normalized)
    parts = normalized.lower().split("_")
    return any(part in keywords for part in parts if part)
