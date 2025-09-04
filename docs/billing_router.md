## Stripe billing router

`stripe_billing_router` centralises all interactions with Stripe.  Modules
resolve the correct product, price and customer identifiers for a bot via
`_resolve_route` and initiate payments through `initiate_charge`.  The router
loads API keys from `VaultSecretProvider`, applies region or tier overrides and
exposes hooks for custom `RouteStrategy` implementations.

```python
from stripe_billing_router import initiate_charge

# All callers must supply a ``bot_id`` in "domain:name:category" format.
initiate_charge("finance:finance_router_bot:monetization", 12.5)
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

The Stripe SDK should **never** be accessed directly.  Bypassing the router
skips key management and audit checks and risks charging the wrong customer.
Always call the router helpers and let them obtain a configured Stripe client.

