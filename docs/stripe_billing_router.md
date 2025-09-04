# Stripe Billing Router

`stripe_billing_router` maps bots to Stripe products, prices and customers.
API keys are pulled from a secure vault provider or fall back to baked‑in
production values. A `RuntimeError` is raised if the keys are missing, empty or
test mode keys. Routing attempts for unsupported domains or missing rules also
raise `RuntimeError`.

## Bot Invocation and Routing Rules

Bots import `stripe_billing_router` directly. Each bot supplies a
`"domain:name:category"` string which the router uses to look up billing
information and attach the appropriate Stripe keys.

Routing rules live in the hierarchical `ROUTING_MAP` inside
`stripe_billing_router.py`. The mapping is organised by
`region -> domain -> bot -> category`:

```python
ROUTING_MAP = {
    "default": {
        "finance": {
            "finance_router_bot": {
                "monetization": {
                    "product_id": "prod_finance_router",
                    "price_id": "price_finance_standard",
                    "customer_id": "cus_finance_default",
                }
            }
        }
    }
}
```

Modify this structure via `register_route` or by editing `ROUTING_MAP` at
start‑up. Use `register_override` for dynamic adjustments.

## Usage

Bots are identified by a `"domain:name:category"` string. The router looks up
routing details for that bot and adds the Stripe keys. For example:

```python
from stripe_billing_router import init_charge, get_balance

init_charge("finance:finance_router_bot:monetization", amount=10.0)
bal = get_balance("finance:finance_router_bot:monetization")
```

## Extending to New Bots

New bots can be supported by registering a route:

```python
from stripe_billing_router import register_route

register_route(
    "analytics",
    "new_bot",
    "monetization",
    {
        "product_id": "prod_new_bot",
        "price_id": "price_new_bot_standard",
        "customer_id": "cus_new_bot_default",
    },
)
```

## Regional Overrides
Region‑specific pricing is handled by registering a route for that region and
selecting it at call time:

```python
from stripe_billing_router import init_charge, register_route

register_route(
    "finance",
    "finance_router_bot",
    "monetization",
    {
        "product_id": "prod_finance_router",
        "price_id": "price_finance_eu",
        "customer_id": "cus_finance_default",
    },
    region="eu",
)

init_charge(
    "finance:finance_router_bot:monetization",
    amount=10.0,
    overrides={"region": "eu"},
)
```

### Business Overrides

Overrides can also target specific business segments by using a different
qualifier:

```python
register_override(
    "finance",
    "finance_router_bot",
    "monetization",
    key="business",
    value="enterprise",
    route={"price_id": "price_finance_enterprise"},
)

init_charge(
    "finance:finance_router_bot:monetization",
    amount=10.0,
    overrides={"business": "enterprise"},
)
```

These examples update the price while leaving the base rule unchanged.
