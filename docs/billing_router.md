## Stripe billing router

`stripe_billing_router` centralises **all** Stripe usage and is the sole payment
interface for bots.  The module owns the Stripe API keys, resolves the correct
product, price, customer and currency identifiers for a bot via `_resolve_route`
and exposes helpers such as `charge` and `create_customer`.  Routes may **not**
include `secret_key` or `public_key` fields—the router injects centrally
managed keys and prevents per‑route overrides.  Keys must not be duplicated or
reimplemented in other modules.

```python
from stripe_billing_router import charge, create_customer

# ``bot_id`` may be "business_category:bot_name" or include a domain prefix.
charge("finance:finance_router_bot", 12.5)
create_customer("finance:finance_router_bot", {"email": "bot@example.com"})
```

### Startup verification

`startup_checks.verify_stripe_router` validates the router configuration during
application startup.  Pass a list of bot identifiers that must resolve to a
billing route; the check raises a `RuntimeError` when any required bot lacks a
valid route.

```python
from startup_checks import verify_stripe_router

verify_stripe_router(["finance:finance_router_bot"])
```

### Extending routing

New bots register billing information with `register_route` or apply conditional
adjustments via `register_override`.  More complex policies can subclass
`RouteStrategy` and register the instance with `register_strategy`.

```python
from stripe_billing_router import register_route

# Default USD billing
register_route(
    "finance",
    "finance_router_bot",
    {
        "product_id": "prod_finance_router",
        "price_id": "price_finance_standard",
        "currency": "usd",
    },
)

# EU region uses EUR for the same bot
register_route(
    "finance",
    "finance_router_bot",
    {
        "product_id": "prod_finance_router_eu",
        "price_id": "price_finance_eu",
        "currency": "eur",
    },
    region="eu",
)
```

### Avoid direct Stripe usage

The Stripe SDK should **never** be accessed directly and Stripe keys must not be
stored outside this module.  Bypassing the router skips key management and audit
checks and risks charging the wrong customer.  Always call the router helpers
and let them obtain a configured Stripe client.

