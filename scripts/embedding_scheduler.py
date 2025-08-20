"""Lightweight embedding backfill scheduler.

This script listens for ``embedding:backfill`` events and triggers the
backfilling of vector embeddings for the requested database.  It also performs
periodic full sweeps to ensure no records are missed.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running as a standalone script without installing the package
sys.path.append(str(Path(__file__).resolve().parents[1]))

from unified_event_bus import UnifiedEventBus  # type: ignore
from vector_service.embedding_backfill import schedule_backfill


async def _handle_event(_topic: str, payload: object) -> None:
    db_name = None
    if isinstance(payload, dict):
        db_name = payload.get("db")
    await schedule_backfill(dbs=[db_name] if db_name else None)


async def main(interval: int = 60) -> None:
    """Run a periodic embedding backfill job."""

    bus = UnifiedEventBus()
    bus.subscribe_async("embedding:backfill", _handle_event)

    while True:
        await schedule_backfill()
        await asyncio.sleep(interval)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    asyncio.run(main())

