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

## Configuration

Authorized webhook endpoints are defined in
`config/stripe_watchdog.yaml`:

```yaml
authorized_webhooks:
  - https://example.com/stripe/webhook
```

Endpoints not listed here will trigger an alert.

## Cron scheduling

The watchdog can run continuously via APScheduler, or it may be scheduled
with cron. To run hourly using cron:

```
0 * * * * cd /path/to/repo && python stripe_watchdog.py >> stripe_watchdog.log 2>&1
```

This command invokes the watchdog every hour and logs output for later
inspection.
