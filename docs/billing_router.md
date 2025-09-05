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
unavailable—and **must** contain the following fields:

- ``id`` – Stripe object identifier or generated UUID.
- ``action_type`` – ``charge``, ``refund``, ``subscription`` or
  ``checkout_session``.
- ``amount`` – Numeric amount of the transaction.
- ``currency`` – Lower‑case ISO‑4217 currency code.
- ``timestamp_ms`` – Milliseconds since the Unix epoch.
- ``user_email`` – Email address tied to the payment, if any.
- ``bot_id`` – ``business_category:bot_name`` that initiated the event.
- ``destination_account`` – Connected or on‑behalf‑of account ID.
- ``raw_event_json`` – Raw JSON payload returned by Stripe.
- ``error`` – ``1`` when a critical discrepancy occurs, otherwise ``0``.

### Master account, allowed keys and rollback alerts

The platform's Stripe master account identifier is hard coded as
``stripe_billing_router.STRIPE_MASTER_ACCOUNT_ID``.  Secret keys that may be
used on behalf of the platform are enumerated via ``STRIPE_ALLOWED_SECRET_KEYS``
(comma separated) or the ``allowed_secret_keys`` list in the routing
configuration.

```bash
export STRIPE_ALLOWED_SECRET_KEYS=sk_prod_main,sk_prod_backup
```

If an unknown key is supplied or a route's ``account_id`` differs from the
master account, the router records the discrepancy in ``DiscrepancyDB``, sends a
``critical_discrepancy`` alert and
``AutomatedRollbackManager.auto_rollback`` reverts the most recent sandbox
changes for the offending bot.

### Responding to critical discrepancy alerts

A ``critical_discrepancy`` alert indicates a misconfigured key or account
mismatch.  Review the corresponding entry in ``stripe_ledger`` or
``finance_logs/stripe_ledger.jsonl`` and the details recorded in
``DiscrepancyDB``.  Once the configuration is corrected, rerun the operation; the
sandbox rollback occurs automatically.

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
refund("finance:finance_router_bot", "ch_123", amount=5.0)
create_subscription("finance:finance_router_bot")
create_checkout_session(
    "finance:finance_router_bot",
    [{"price": "price_finance_standard", "quantity": 1}],
    success_url="https://example.com/s",
    cancel_url="https://example.com/c",
    mode="payment",
)
```

