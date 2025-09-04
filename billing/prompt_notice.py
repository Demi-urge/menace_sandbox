"""Centralized payment router notice for LLM prompts."""

from stripe_policy import PAYMENT_ROUTER_NOTICE
import stripe_billing_router  # noqa: F401

__all__ = ["PAYMENT_ROUTER_NOTICE"]
