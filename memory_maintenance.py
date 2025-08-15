import os
import threading
from typing import Dict, Mapping

from gpt_memory import GPTMemoryManager
from logging_utils import get_logger

logger = get_logger(__name__)

# Interval in seconds between compaction cycles
DEFAULT_INTERVAL = float(os.getenv("GPT_MEMORY_COMPACT_INTERVAL", "3600"))


def _load_retention_rules() -> Dict[str, int]:
    """Parse per-tag retention limits from environment variables."""
    rules: Dict[str, int] = {}
    # Combined mapping: "tag1=10,tag2=5"
    env_map = os.getenv("GPT_MEMORY_RETENTION", "")
    for item in env_map.split(","):
        if not item.strip():
            continue
        tag, _, num = item.partition("=")
        try:
            rules[tag.strip()] = int(num.strip())
        except ValueError:
            continue
    # Prefix variables: GPT_MEMORY_RETENTION_<TAG>=N
    prefix = "GPT_MEMORY_RETENTION_"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            tag = key[len(prefix) :].lower()
            try:
                rules[tag] = int(value)
            except ValueError:
                continue
    return rules


class MemoryMaintenance:
    """Background thread that periodically compacts GPT memory."""

    def __init__(
        self,
        manager: GPTMemoryManager,
        *,
        interval: float | None = None,
        retention: Mapping[str, int] | None = None,
    ) -> None:
        self.manager = manager
        self.interval = float(interval or DEFAULT_INTERVAL)
        self.retention: Dict[str, int] = (
            dict(retention) if retention is not None else _load_retention_rules()
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        if not self.retention:
            logger.debug("memory maintenance disabled – no retention rules configured")
            return
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    # ------------------------------------------------------------------ internal
    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                removed = self.manager.compact(self.retention)
                logger.debug("memory maintenance removed %d rows", removed)
            except Exception:
                logger.exception("memory maintenance failed during compaction")
