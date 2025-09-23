"""Background maintenance utilities for :mod:`gpt_memory`."""

from __future__ import annotations

import os
import threading
import time
import argparse
import sys
from typing import Dict, Mapping, TYPE_CHECKING, Iterable

from menace_sandbox.gpt_memory import GPTMemoryManager

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from gpt_knowledge_service import GPTKnowledgeService
from logging_utils import get_logger

logger = get_logger(__name__)

# Interval in seconds between maintenance cycles
DEFAULT_INTERVAL = float(os.getenv("GPT_MEMORY_COMPACT_INTERVAL", "3600"))

# Interval in seconds between knowledge refresh cycles
# Falls back to ``DEFAULT_INTERVAL`` when not explicitly configured.
DEFAULT_INSIGHT_INTERVAL = float(os.getenv("GPT_INSIGHT_REFRESH_INTERVAL", "0"))


def _load_prune_limit() -> int | None:
    """Return the max rows per tag parsed from ``GPT_MEMORY_MAX_ROWS``."""

    try:
        limit = int(os.getenv("GPT_MEMORY_MAX_ROWS", "0"))
    except ValueError:
        return None
    return limit if limit > 0 else None


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


def prune_memory(manager: GPTMemoryManager, max_rows: int) -> int:
    """Convenience wrapper around :meth:`GPTMemoryManager.prune_old_entries`."""

    removed = manager.prune_old_entries(max_rows)
    logger.debug("memory maintenance pruned %d rows", removed)
    return removed


class MemoryMaintenance:
    """Background thread that periodically compacts GPT memory."""

    def __init__(
        self,
        manager: GPTMemoryManager,
        *,
        interval: float | None = None,
        retention: Mapping[str, int] | None = None,
        max_rows: int | None = None,
        knowledge_service: GPTKnowledgeService | None = None,
        insight_interval: float | None = None,
    ) -> None:
        self.manager = manager
        self.interval = float(interval or DEFAULT_INTERVAL)
        self.retention: Dict[str, int] = (
            dict(retention) if retention is not None else _load_retention_rules()
        )
        self.max_rows = max_rows if max_rows is not None else _load_prune_limit()
        self.knowledge_service = knowledge_service
        self.insight_interval = float(
            insight_interval or DEFAULT_INSIGHT_INTERVAL or self.interval
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._insight_thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.retention and not self.max_rows and self.knowledge_service is None:
            logger.debug(
                "memory maintenance disabled – no retention, pruning rules or knowledge service"
            )
            return
        if self.retention or self.max_rows:
            self._thread.start()
        if self.knowledge_service is not None:
            if self._insight_thread is None:
                self._insight_thread = threading.Thread(
                    target=self._insight_loop, daemon=True
                )
            self._insight_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._insight_thread and self._insight_thread.is_alive():
            self._insight_thread.join(timeout=1.0)

    # ------------------------------------------------------------------ internal
    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                if self.retention:
                    removed = self.manager.compact(self.retention)
                    logger.debug("memory maintenance removed %d rows", removed)
                if self.max_rows:
                    pruned = self.manager.prune_old_entries(self.max_rows)
                    logger.debug("memory maintenance pruned %d rows", pruned)
            except Exception:
                logger.exception("memory maintenance failed during maintenance")

            kg = getattr(self.manager, "graph", None)
            if kg is not None:
                try:
                    kg.ingest_gpt_memory(self.manager)
                except Exception:
                    logger.exception("knowledge graph update failed")

    def _insight_loop(self) -> None:
        while not self._stop.wait(self.insight_interval):
            try:
                assert self.knowledge_service is not None
                self.knowledge_service.update_insights()
            except Exception:
                logger.exception("knowledge service update failed")


# --------------------------------------------------------------------------- CLI
def cli(argv: Iterable[str] | None = None) -> int:
    """Entry point for command line execution."""

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="start maintenance loops")
    p_run.add_argument("--interval", type=float, default=None, help="memory compact interval")
    p_run.add_argument(
        "--insight-interval",
        type=float,
        default=None,
        help="insight refresh interval",
    )

    sub.add_parser("refresh", help="refresh insights once and exit")

    args = parser.parse_args(list(argv) if argv is not None else None)

    from shared_gpt_memory import GPT_MEMORY_MANAGER
    from gpt_knowledge_service import GPTKnowledgeService

    service = GPTKnowledgeService(GPT_MEMORY_MANAGER)

    if args.cmd == "refresh":
        service.update_insights()
        return 0

    maint = MemoryMaintenance(
        GPT_MEMORY_MANAGER,
        interval=args.interval,
        knowledge_service=service,
        insight_interval=args.insight_interval,
    )
    maint.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        maint.stop()
    return 0


def main(argv: Iterable[str] | None = None) -> None:  # pragma: no cover - CLI glue
    sys.exit(cli(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
