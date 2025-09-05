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
