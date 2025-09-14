from __future__ import annotations

"""Provide a shared :class:`EvolutionOrchestrator` instance.

This helper lazily constructs an :class:`EvolutionOrchestrator` with minimal
dependencies and returns the same instance for all callers.  Bots retrieving
the orchestrator can therefore coordinate through a common patch provenance
token.
"""

from .data_bot import DataBot
from .capital_management_bot import CapitalManagementBot
from .system_evolution_manager import SystemEvolutionManager
from .self_coding_engine import SelfCodingEngine
from .evolution_orchestrator import EvolutionOrchestrator

_shared_orchestrator: EvolutionOrchestrator | None = None


def get_orchestrator(
    bot_name: str, data_bot: DataBot, engine: SelfCodingEngine
) -> EvolutionOrchestrator:
    """Return a singleton ``EvolutionOrchestrator``.

    Parameters
    ----------
    bot_name:
        Name of the bot requesting orchestration. The first caller determines
        the initial bot list for :class:`SystemEvolutionManager`.
    data_bot:
        Shared :class:`DataBot` instance.
    engine:
        :class:`SelfCodingEngine` powering self-improvement.
    """
    global _shared_orchestrator
    if _shared_orchestrator is None:
        capital_bot = CapitalManagementBot()
        evolution_manager = SystemEvolutionManager(bots=[bot_name])
        _shared_orchestrator = EvolutionOrchestrator(
            data_bot, capital_bot, engine, evolution_manager
        )
    return _shared_orchestrator
