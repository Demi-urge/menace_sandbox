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

The repository includes `scripts/check_raw_stripe_usage.py` which scans all
tracked text files for raw Stripe keys or payment related keywords such as
"payment", "checkout" or "billing".  Any file mentioning these keywords must
also reference ``stripe_billing_router`` or the check fails.

### Ledger schema and automatic logging

Every billing operation recorded through ``stripe_billing_router`` is persisted
via ``billing_logger.log_event``.  Records are written to the ``stripe_ledger``
table—falling back to ``finance_logs/stripe_ledger.jsonl`` when the database is
unavailable—and contain the following columns:

- ``id``
- ``action_type``
- ``amount``
- ``currency``
- ``timestamp_ms``
- ``user_email``
- ``bot_id``
- ``destination_account``
- ``raw_event_json``
- ``error``

### Master account and rollback alerts

Set ``STRIPE_MASTER_ACCOUNT_ID`` to the platform's Stripe master account
identifier.  The router verifies that each Stripe response references this
account.  If a different account is detected the event is logged with
``error=1``, ``alert_dispatcher`` sends a ``critical_discrepancy`` alert and
``rollback_manager.RollbackManager.auto_rollback`` attempts to revert the
action.

### Example flows with logging

```python
from stripe_billing_router import (
    charge,
    refund,
    create_subscription,
    create_checkout_session,
)

# Each call below automatically appends a row to ``stripe_ledger``
charge("finance:finance_router_bot", amount=10.0)
refund("finance:finance_router_bot", "pi_123", amount=5.0)
create_subscription("finance:finance_router_bot")
create_checkout_session(
    "finance:finance_router_bot",
    {"success_url": "https://example.com/s", "cancel_url": "https://example.com/c", "mode": "payment"},
)
```

