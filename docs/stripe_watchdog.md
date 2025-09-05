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

- `STRIPE_SECRET_KEY` – Stripe API key used to fetch events.
- `STRIPE_ALLOWED_WEBHOOKS` *(optional)* – comma-separated list of additional
  authorized webhook endpoints.

Anomaly summaries and audit entries are written to
`finance_logs/stripe_watchdog.log`.

## Configuration

Authorized webhook endpoints are defined in
`config/stripe_watchdog.yaml`:

```yaml
authorized_webhooks:
  - https://example.com/stripe/webhook
```

Endpoints not listed here will trigger an alert.

## Systemd service

To run the watchdog as a background service create a unit file such as:

```
[Unit]
Description=Stripe Watchdog
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/repo
Environment=STRIPE_SECRET_KEY=sk_live_yourkey
ExecStart=/usr/bin/python stripe_watchdog.py
Restart=on-failure
StandardOutput=append:/path/to/repo/finance_logs/stripe_watchdog.log
StandardError=append:/path/to/repo/finance_logs/stripe_watchdog.log

[Install]
WantedBy=multi-user.target
```

## Cron scheduling

The watchdog can run continuously via APScheduler, or it may be scheduled
with cron. To run hourly using cron:

```
0 * * * * cd /path/to/repo && STRIPE_SECRET_KEY=sk_live_yourkey python stripe_watchdog.py >> finance_logs/stripe_watchdog.log 2>&1
```

This command invokes the watchdog every hour and logs output for later
inspection.
