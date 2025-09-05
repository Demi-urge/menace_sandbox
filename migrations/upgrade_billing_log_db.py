import sys
from pathlib import Path

from billing.billing_log_db import BillingLogDB


def upgrade(path: str | Path | None = None) -> None:
    """Ensure billing_logs table has the stripe_id column."""
    BillingLogDB(path or BillingLogDB().path)


if __name__ == "__main__":
    upgrade(sys.argv[1] if len(sys.argv) > 1 else None)
