from __future__ import annotations

"""Simple transaction helper for mirroring operations."""

from typing import Callable, Any
import logging
import time

class TransactionManager:
    """Execute mirrored DB operations atomically with retries."""

    def __init__(self, retries: int = 1, delay: float = 0.1) -> None:
        self.retries = retries
        self.delay = delay

    def run(self, operation: Callable[[], Any], rollback: Callable[[], None]) -> Any:
        """Run *operation* with optional retries. Roll back on failure."""
        attempt = 0
        while True:
            try:
                return operation()
            except Exception:
                if attempt >= self.retries:
                    try:
                        rollback()
                    except Exception:
                        logging.getLogger(__name__).exception(
                            "rollback callback failed"
                        )
                    raise
                attempt += 1
                time.sleep(self.delay)
