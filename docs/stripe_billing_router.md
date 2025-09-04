# Stripe Billing Router

`stripe_billing_router` maps bots to Stripe products, prices and customers.
API keys are pulled from a secure vault provider or fall back to baked‑in
production values. A `RuntimeError` is raised if the keys are missing or empty.

## Bot Invocation and Routing Rules

Bots either import `stripe_billing_router` directly or use the compatibility
wrapper `stripe_handler`. Each bot supplies a `"domain:name:category"` string
which the router uses to look up billing information and attach the appropriate
Stripe keys.

Routing rules live in the global `BILLING_RULES` mapping inside
`stripe_billing_router.py`. Edit this mapping at start‑up or use
`register_override` for dynamic adjustments.

## Usage

Bots are identified by a `"domain:name:category"` string. The router looks up
routing details for that bot and adds the Stripe keys. For example:

```python
from stripe_billing_router import init_charge, get_balance

init_charge("finance:finance_router_bot:monetization", amount=10.0)
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
from stripe_billing_router import init_charge, register_override

register_override(
    "finance",
    "finance_router_bot",
    "monetization",
    key="region",
    value="eu",
    route={"price_id": "price_finance_eu"},
)

init_charge(
    "finance:finance_router_bot:monetization",
    amount=10.0,
    overrides={"region": "eu"},
)
```

This updates the price used for European customers while leaving the base rule
unchanged.
