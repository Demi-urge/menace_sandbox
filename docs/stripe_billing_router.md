# Stripe Billing Router

`stripe_billing_router` maps bots to Stripe products, prices and customers.
It reads API keys from the `STRIPE_SECRET_KEY` and `STRIPE_PUBLIC_KEY`
environment variables at import time and raises a `RuntimeError` if they are
missing.

## Usage

Bots are identified by a `"domain:name:category"` string. The router looks up
routing details for that bot and adds the Stripe keys. For example:

```python
from stripe_billing_router import charge, get_balance

charge("finance:finance_router_bot:monetization", amount=10.0)
bal = get_balance("finance:finance_router_bot:monetization")
```

## Extending to New Bots

New bots can be supported by adding an entry to `BILLING_RULES` at start up:

```python
from stripe_billing_router import BILLING_RULES

BILLING_RULES[("analytics", "new_bot", "monetization")] = {
    "product_id": "prod_new_bot",
    "price_id": "price_new_bot_standard",
    "customer_id": "cus_new_bot_default",
}
```

## Regional Overrides

Per-region adjustments use the `register_override` hook. A qualifier such as a
region is supplied at call time via `overrides`:

```python
from stripe_billing_router import charge, register_override

register_override(
    "finance",
    "finance_router_bot",
    "monetization",
    key="region",
    value="eu",
    route={"price_id": "price_finance_eu"},
)

charge(
    "finance:finance_router_bot:monetization",
    amount=10.0,
    overrides={"region": "eu"},
)
```

This updates the price used for European customers while leaving the base rule
unchanged.
