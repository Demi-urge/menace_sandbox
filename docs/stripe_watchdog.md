# Stripe Watchdog

The Stripe Watchdog cross-checks recent Stripe activity against the local
billing ledger and alerts on discrepancies.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Provide the Stripe secret key through environment configuration or a
   compatible `VaultSecretProvider` entry.
3. Ensure billing events are appended to `finance_logs/stripe_ledger.jsonl`.

## Environment

Set the following environment variables before execution:

- `STRIPE\_SECRET_KEY` – Stripe API key used to fetch events.
- `STRIPE_ALLOWED_WEBHOOKS` *(optional)* – comma-separated list of additional
  authorized webhook endpoints.
- `GPT_MEMORY_DB` *(optional)* – path to the SQLite file where feedback
  snippets are stored. Defaults to `gpt_memory.db`.

Anomaly summaries are appended to `finance_logs/stripe_watchdog.log`, while
structured audit records are written to
`finance_logs/stripe_watchdog_audit.jsonl`. When Sanity Layer feedback is
enabled, corrective guidance is logged to the GPT memory database referenced by
`GPT_MEMORY_DB`.

## Configuration

Authorized webhook endpoints are defined in
`config/stripe_watchdog.yaml`:

```yaml
authorized_webhooks:
  - https://example.com/stripe/webhook
```

Endpoints not listed here will trigger an alert.

To enable the Sanity Layer feedback loop, ensure `sanity_layer_feedback` is set
to `true` in this YAML file (the default). Setting it to `false` disables
recording anomalies to GPT memory and the corresponding audit trail.

## Systemd timer

The repository includes `systemd/stripe_watchdog.service` and `systemd/stripe_watchdog.timer` which execute the watchdog hourly. Install and enable the timer with:

```bash
sudo cp systemd/stripe_watchdog.* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now stripe_watchdog.timer
```

## Cron fallback

If systemd is unavailable, an example installer adds a comparable cron job:

```bash
bash scripts/install_stripe_watchdog_cron.sh
```

The script registers an hourly cron entry running `stripe_watchdog.py`.

## Sanity layer integration

Detected discrepancies are forwarded to the Menace Sanity Layer which records
them in SQLite and logs corrective guidance to GPT memory.  A minimal example
invocation looks like:

```python
from db_router import init_db_router
from stripe_watchdog import detect_missing_charges

init_db_router("ba", "local.db", "shared.db")
charges = [{"id": "ch_1", "amount": 5}]
detect_missing_charges(charges, [], write_codex=True, export_training=False)
```

Each anomaly is logged with the instruction:

```
Avoid generating bots that make Stripe charges without proper logging or central routing.
```

Event types follow a unified naming scheme. For example,
unauthorized Stripe activity is recorded as `unauthorized_charge`,
`unauthorized_refund`, or `unauthorized_failure` depending on the
operation. Missing ledger entries continue to use `missing_charge`,
`missing_refund`, and `missing_failure_log`.

Use the `write_codex` and `export_training` flags to control whether anomalies
are exported as Codex samples or appended to the training dataset.
