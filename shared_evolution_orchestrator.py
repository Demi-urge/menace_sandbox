from __future__ import annotations

"""Provide a shared :class:`EvolutionOrchestrator` instance.

This helper lazily constructs an :class:`EvolutionOrchestrator` with minimal
dependencies and returns the same instance for all callers.  Bots retrieving
the orchestrator can therefore coordinate through a common patch provenance
token.
"""

from contextlib import contextmanager
from typing import Iterator

from .data_bot import DataBot
from .capital_management_bot import CapitalManagementBot
from .system_evolution_manager import SystemEvolutionManager
from .self_coding_engine import SelfCodingEngine
from .evolution_orchestrator import EvolutionOrchestrator

_shared_orchestrator: EvolutionOrchestrator | None = None


@contextmanager
def _capital_bot_manual_mode() -> Iterator[None]:
    """Temporarily disable self-coding requirements for ``CapitalManagementBot``.

    ``CapitalManagementBot`` is decorated with :func:`self_coding_managed`, which
    normally enforces the presence of a :class:`SelfCodingManager` instance when
    the self-coding runtime is available.  The shared orchestrator is used early
    during bootstrap before a manager exists, so the instantiation would raise
    ``RuntimeError``.  The decorator exposes a ``_self_coding_manual_mode`` flag
    to bypass this requirement.  This context manager toggles the flag only for
    the duration of the instantiation, restoring the previous value afterwards.
    """

    unset = object()
    previous = getattr(CapitalManagementBot, "_self_coding_manual_mode", unset)
    CapitalManagementBot._self_coding_manual_mode = True  # type: ignore[attr-defined]
    try:
        yield
    finally:
        if previous is unset:
            delattr(CapitalManagementBot, "_self_coding_manual_mode")
        else:
            CapitalManagementBot._self_coding_manual_mode = previous  # type: ignore[attr-defined]


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
        with _capital_bot_manual_mode():
            capital_bot = CapitalManagementBot()
        evolution_manager = SystemEvolutionManager(bots=[bot_name])
        _shared_orchestrator = EvolutionOrchestrator(
            data_bot, capital_bot, engine, evolution_manager
        )
    return _shared_orchestrator
