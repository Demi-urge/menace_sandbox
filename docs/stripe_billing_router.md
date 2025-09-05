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
        account_id: acct_master
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
# via the internal ``_log_payment`` helper backed by ``StripeLedger``.

# One‑off payment via PaymentIntent
charge("finance:finance_router_bot", amount=10.0)

# Price based invoice using the route's price_id
charge("finance:finance_router_bot", price_id="price_finance_standard")

# Create a recurring subscription
create_subscription("finance:finance_router_bot")

# Issue a refund for a previous payment
refund("finance:finance_router_bot", "ch_test")

# Create a Checkout session
create_checkout_session(
    "finance:finance_router_bot",
    [{"price": "price_finance_standard", "quantity": 1}],
    success_url="https://example.com/s",
    cancel_url="https://example.com/c",
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

All billing helpers use the private ``_log_payment`` helper which records
events via :class:`StripeLedger` to the ``stripe_ledger`` table (falling back to
``finance_logs/stripe_ledger.jsonl`` if the database is unavailable).  Each log
entry **must** populate the
following fields:

- ``id`` – Stripe object identifier or a generated UUID for the event.
- ``action_type`` – ``charge``, ``refund``, ``subscription`` or
  ``checkout_session``.
- ``amount`` – Numeric amount associated with the action.
- ``currency`` – ISO‑4217 currency code in lower case.
- ``timestamp_ms`` – Event time in milliseconds since the Unix epoch.
- ``user_email`` – Email address associated with the payment, when available.
- ``bot_id`` – ``business_category:bot_name`` that initiated the event.
- ``destination_account`` – Connected account or on‑behalf‑of account.
- ``raw_event_json`` – Full JSON response returned by Stripe.
- ``error`` – ``1`` when a critical discrepancy is detected, otherwise ``0``.

### Allowed Keys, Account Verification and Rollback

Allowed secret keys are provided via the ``STRIPE_ALLOWED_SECRET_KEYS``
environment variable (comma separated) or the ``allowed_secret_keys`` list in
``config/stripe_billing_router.yaml``. Routes may also include an
``account_id`` which is cross‑checked against the registered account.
Before any Stripe action is executed, the router ensures the resolved
``secret_key`` is in the allowed list and that the route's ``account_id``
matches the registered account. If either check fails the router:

1. Records the discrepancy in :class:`DiscrepancyDB`.
2. Dispatches a ``critical_discrepancy`` alert.
3. Calls ``AutomatedRollbackManager.auto_rollback`` for the bot.

Investigate these alerts by inspecting the discrepancy record; they typically
indicate a misconfigured key or account.

The platform's registered account identifier is hard coded as
``stripe_billing_router.STRIPE_MASTER_ACCOUNT_ID`` (also exported as
``STRIPE_REGISTERED_ACCOUNT_ID``). This value is immutable and must never be
overridden or sourced from environment variables or secret storage.

A ``critical_discrepancy`` alert signals the automatic rollback described
above; resolve the configuration issue before retrying the billing operation.

### Central Anomaly Routing

Any detected discrepancy or account mismatch **must** be routed through
``menace_sanity_layer.record_payment_anomaly`` with a severity of ``5.0``.
This central handler records the event in the shared ledger and ensures audit
consistency.  Bypassing the router or logging anomalies elsewhere is
prohibited; always forward full details so investigations have a single source
of truth.

## Webhook Account Validation

Stripe webhooks should also be verified to ensure events originate from the
registered platform account.  The helper
``stripe_billing_router.validate_webhook_account`` inspects an incoming webhook
payload and returns ``False`` when the embedded account identifier does not
match :data:`STRIPE_MASTER_ACCOUNT_ID`:

```python
from stripe_billing_router import validate_webhook_account

event = json.loads(request.data)
if not validate_webhook_account(event):
    # Pause or reject processing until the mismatch is reviewed
    return "account mismatch", 400
```

When a mismatch is detected the router dispatches the standard alert via
``_alert_mismatch`` allowing callers to pause the bot or trigger additional
review steps.

## Watchdog Webhook Configuration

The companion `stripe_watchdog` module verifies the set of webhook endpoints
registered in Stripe.  Authorized endpoints are listed in
`config/stripe_watchdog.yaml`:

```yaml
authorized_webhooks:
  - https://example.com/stripe/webhook
```

During a watchdog run, the output of `stripe.WebhookEndpoint.list()` is compared
against this list. Any unrecognized endpoint results in a
`stripe_unknown_endpoint` alert being logged.

### Training Data Export

`stripe_watchdog` can also export anomalies for model training.  Running the
watchdog with the `--export-training` flag appends normalized records to
`training_data/stripe_anomalies.jsonl`.  Each line is a JSON object matching the
[`codex_training_data`](codex_training_data.md) format, for example:

```json
{"source": "stripe_watchdog", "content": "{\"type\": \"unknown_webhook\"}", "timestamp": 1690000000}
```

These exports can be ingested by existing training data loaders alongside other
Codex samples.
