from __future__ import annotations

"""Central registry for supervisor-managed runnable services."""

from dataclasses import dataclass

from intended_production_bots import INTENDED_PRODUCTION_BOTS
from production_bot_manifest import PRODUCTION_BOT_MANIFEST


@dataclass(frozen=True)
class RunnableBotEntry:
    name: str
    startup_module: str
    startup_callable: str
    health_endpoint: str | None = None
    liveness_check: str = "process_alive"
    critical: bool = False
    enabled_if_env: str | None = None
    needs_context_builder: bool = False


RUNNABLE_BOT_REGISTRY: tuple[RunnableBotEntry, ...] = tuple(
    RunnableBotEntry(
        name=entry.name,
        startup_module=entry.startup_module,
        startup_callable=entry.startup_callable,
        health_endpoint=entry.health_endpoint,
        liveness_check=entry.liveness_check,
        critical=entry.critical,
        enabled_if_env=entry.enabled_if_env,
        needs_context_builder=entry.needs_context_builder,
    )
    for entry in PRODUCTION_BOT_MANIFEST
)


__all__ = ["RunnableBotEntry", "RUNNABLE_BOT_REGISTRY", "INTENDED_PRODUCTION_BOTS"]
