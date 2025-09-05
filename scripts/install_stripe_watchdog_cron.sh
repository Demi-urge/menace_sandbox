#!/bin/bash
set -e

# Install an hourly cron entry for stripe_watchdog.py when systemd isn't available.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

CRON_LINE="0 * * * * cd $REPO_ROOT && /usr/bin/python stripe_watchdog.py >> finance_logs/stripe_watchdog.log 2>&1"
( crontab -l 2>/dev/null | grep -v 'stripe_watchdog.py'; echo "$CRON_LINE" ) | crontab -

echo "Installed stripe_watchdog cron entry."
