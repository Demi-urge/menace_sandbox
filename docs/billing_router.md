## Stripe billing router

`stripe_billing_router` centralises **all** Stripe usage and is the sole payment
interface for bots.  The module owns the API keys, resolves the correct product,
price and customer identifiers for a bot via `_resolve_route` and exposes
helpers such as `charge` and `create_customer`.  Keys must not be duplicated or
reimplemented in other modules.

```python
from stripe_billing_router import charge, create_customer

# ``bot_id`` must be in "business_category:bot_name" format.
charge("finance:finance_router_bot", 12.5)
create_customer("finance:finance_router_bot", {"email": "bot@example.com"})
```

### Extending routing

New bots register billing information with `register_route` or apply conditional
adjustments via `register_override`.  More complex policies can subclass
`RouteStrategy` and register the instance with `register_strategy`.

```python
from stripe_billing_router import register_route

register_route(
    "finance",
    "finance_router_bot",
    "monetization",
    {"product_id": "prod_finance_router", "price_id": "price_finance_standard"},
)
```

### Avoid direct Stripe usage

The Stripe SDK should **never** be accessed directly and Stripe keys must not be
stored outside this module.  Bypassing the router skips key management and audit
checks and risks charging the wrong customer.  Always call the router helpers
and let them obtain a configured Stripe client.

