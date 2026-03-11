from __future__ import annotations

"""Derived list of services intended for production supervisor launch.

Production intent is declared in :mod:`production_bot_manifest` via the
``ProductionBotManifestEntry.intended_for_production`` marker.
:data:`INTENDED_PRODUCTION_BOTS` is generated from that canonical manifest to
avoid drift between intent and supervisor registry wiring.
"""

from production_bot_manifest import PRODUCTION_BOT_MANIFEST

INTENDED_PRODUCTION_BOTS: tuple[str, ...] = tuple(
    entry.name for entry in PRODUCTION_BOT_MANIFEST if entry.intended_for_production
)


__all__ = ["INTENDED_PRODUCTION_BOTS"]
