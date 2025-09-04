"""Stripe policy constants used across the codebase."""

PAYMENT_ROUTER_NOTICE = (
    "All payment logic must import and use stripe_billing_router. "
    "Direct Stripe SDK calls or raw Stripe keys are forbidden. "
    "Missing the import must raise a critical generation failure."
)
