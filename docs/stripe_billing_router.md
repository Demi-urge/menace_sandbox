# Stripe Billing Router

`stripe_billing_router` maps bots to Stripe products, prices and customers and
acts as the **sole payment interface**.  API keys are pulled from a secure vault
provider or fall back to baked‑in production values.  Charges are created via
Stripe's modern PaymentIntent or Invoice APIs—`Charge.create` is no longer
used. A `RuntimeError` is raised if keys are missing, empty or test‑mode keys.
Routing attempts for unsupported domains, business categories or missing rules
also raise `RuntimeError`.  Routes must never supply `secret_key` or
`public_key` values—the router injects centrally managed keys and prevents
per‑route overrides.  Stripe keys and billing logic must not be duplicated
outside this module.

## Bot Invocation and Routing Rules

Bots import `stripe_billing_router` directly. Each bot supplies a
`"business_category:bot_name"` string and may optionally prefix the domain,
resulting in ``"domain:business_category:bot_name"``.  When omitted, the domain
defaults to ``stripe``.  The router uses this identifier to look up billing
information and attach the appropriate Stripe keys.

Routing rules are loaded from ``config/stripe_billing_router.yaml`` (override
via the ``STRIPE_ROUTING_CONFIG`` environment variable) and stored in
``ROUTING_TABLE``. The configuration is a nested mapping of
``domain -> region -> business_category -> bot_name``.  A minimal example:

```yaml
stripe:
  default:
    finance:
      finance_router_bot:
        product_id: prod_finance_router
        price_id: price_finance_standard
        customer_id: cus_finance_default
```

Modify the configuration file or call `register_route` at start‑up to add or
change routes. Use `register_override` for dynamic adjustments.

## Usage

Bots are identified by a `"business_category:bot_name"` string or
``"domain:business_category:bot_name"``.  The router looks up routing details
for that bot and adds the Stripe keys.  Bots request charges, subscriptions or
create customers by calling router helpers:

```python
from stripe_billing_router import (
    charge,
    create_customer,
    create_subscription,
    refund,
    create_checkout_session,
    get_balance,
)

# Each helper automatically logs to the ``stripe_ledger`` table
# via ``billing_logger.log_event``.

# One‑off payment via PaymentIntent
charge("finance:finance_router_bot", amount=10.0)

# Price based invoice using the route's price_id
charge("finance:finance_router_bot", price_id="price_finance_standard")

# Create a recurring subscription
create_subscription("finance:finance_router_bot")

# Issue a refund for a previous payment
refund("finance:finance_router_bot", "pi_test")

# Create a Checkout session
create_checkout_session(
    "finance:finance_router_bot",
    {"success_url": "https://example.com/s", "cancel_url": "https://example.com/c"},
)

create_customer("finance:finance_router_bot", {"email": "bot@example.com"})
bal = get_balance("finance:finance_router_bot")
```

## Extending to New Bots

New bots can be supported by registering a route:

```python
from stripe_billing_router import register_route

register_route(
    "analytics",
    "new_bot",
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
from stripe_billing_router import charge, register_route

register_route(
    "finance",
    "finance_router_bot",
    {
        "product_id": "prod_finance_router",
        "price_id": "price_finance_eu",
        "customer_id": "cus_finance_default",
    },
    region="eu",
)

charge(
    "finance:finance_router_bot",
    overrides={"region": "eu"},
)
```

### Business Overrides

Overrides can also target specific business segments by using a different
qualifier:

```python
register_override(
    {
        "business_category": "finance",
        "bot_name": "finance_router_bot",
        "key": "business",
        "value": "enterprise",
        "route": {"price_id": "price_finance_enterprise"},
    }
)

charge(
    "finance:finance_router_bot",
    overrides={"business": "enterprise"},
)
```

These examples update the price while leaving the base rule unchanged.

## Ledger Schema and Automatic Logging

All billing helpers call ``billing_logger.log_event`` which persists a record to
the ``stripe_ledger`` table (falling back to ``finance_logs/stripe_ledger.jsonl``
if the database is unavailable).  The schema columns are:

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

### Master Account Verification and Rollback

The router expects a platform master account identifier supplied via the
``STRIPE_MASTER_ACCOUNT_ID`` environment variable (or secret vault).  Each
Stripe response must reference this account; if a mismatch is detected the
router logs a discrepancy with ``error=1``, dispatches a
``critical_discrepancy`` alert and triggers
``rollback_manager.RollbackManager.auto_rollback``.

```bash
export STRIPE_MASTER_ACCOUNT_ID=acct_master
```
