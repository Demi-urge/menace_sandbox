"""Provide a shared :class:`EvolutionOrchestrator` instance.

This helper lazily constructs an :class:`EvolutionOrchestrator` with minimal
dependencies and returns the same instance for all callers. Bots retrieving the
orchestrator can therefore coordinate through a common patch provenance token.

To avoid circular import errors the heavy dependencies are imported only inside
``get_orchestrator``.  This keeps module initialisation trivial so callers can
``import orchestrator_loader`` even when :mod:`capital_management_bot` is still
being evaluated.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Iterator, Mapping, TYPE_CHECKING, Any

from bootstrap_gate import resolve_bootstrap_placeholders
from coding_bot_interface import get_active_bootstrap_pipeline
from .bootstrap_placeholder import advertise_broker_placeholder
from .bootstrap_helpers import bootstrap_state_snapshot, ensure_bootstrapped

_BOOTSTRAP_PLACEHOLDER: object | None = None
_BOOTSTRAP_SENTINEL: object | None = None
_BOOTSTRAP_BROKER: object | None = None
_BOOTSTRAP_GATE_TIMEOUT = 12.0


def _bootstrap_placeholders(
    *, bootstrap_state: Mapping[str, object] | None = None
) -> tuple[object, object, object]:
    """Advertise bootstrap placeholders once the readiness gate clears."""

    global _BOOTSTRAP_PLACEHOLDER, _BOOTSTRAP_SENTINEL, _BOOTSTRAP_BROKER
    # Always lean on the global readiness snapshot instead of triggering
    # bootstrap work from orchestrator imports.
    state = bootstrap_state or bootstrap_state_snapshot()
    if not state.get("ready") and not state.get("in_progress"):
        ensure_bootstrapped()
    if None not in (_BOOTSTRAP_PLACEHOLDER, _BOOTSTRAP_SENTINEL, _BOOTSTRAP_BROKER):
        return _BOOTSTRAP_PLACEHOLDER, _BOOTSTRAP_SENTINEL, _BOOTSTRAP_BROKER

    pipeline, manager = get_active_bootstrap_pipeline()
    if pipeline is not None or manager is not None:
        _BOOTSTRAP_PLACEHOLDER, _BOOTSTRAP_SENTINEL, _BOOTSTRAP_BROKER = (
            advertise_broker_placeholder(
                pipeline=pipeline,
                manager=manager,
            )
        )
        return _BOOTSTRAP_PLACEHOLDER, _BOOTSTRAP_SENTINEL, _BOOTSTRAP_BROKER

    pipeline, manager, broker = resolve_bootstrap_placeholders(
        timeout=_BOOTSTRAP_GATE_TIMEOUT,
        description="Orchestrator bootstrap gate",
    )
    _BOOTSTRAP_PLACEHOLDER, _BOOTSTRAP_SENTINEL, _BOOTSTRAP_BROKER = (
        advertise_broker_placeholder(
            dependency_broker=broker,
            pipeline=pipeline,
            manager=manager,
        )
    )
    return _BOOTSTRAP_PLACEHOLDER, _BOOTSTRAP_SENTINEL, _BOOTSTRAP_BROKER

if TYPE_CHECKING:  # pragma: no cover - typing only import
    from .data_bot import DataBot
    from .self_coding_engine import SelfCodingEngine
    from .evolution_orchestrator import EvolutionOrchestrator
    from .capital_management_bot import CapitalManagementBot

logger = logging.getLogger(__name__)


class _LazyEvolutionManager:
    """Lazily construct :class:`SystemEvolutionManager` on first use.

    Importing ``system_evolution_manager`` performs significant work including
    instantiating several bots.  During bootstrap many of those bots import
    :mod:`orchestrator_loader`, so eagerly importing the manager introduces a
    circular dependency.  The lazy wrapper defers the import and instantiation
    until an attribute is actually accessed, breaking the cycle while keeping
    runtime behaviour unchanged.

    The wrapper is thread-safe and degrades gracefully by falling back to a
    no-op manager when the real manager cannot be created (for example when
    optional dependencies are unavailable on Windows).
    """

    def __init__(self, bot_name: str) -> None:
        self._bot_name = bot_name
        self._delegate: Any | None = None
        self._lock = threading.RLock()

    def _ensure_delegate(self) -> Any:
        with self._lock:
            if self._delegate is None:
                try:
                    from .system_evolution_manager import SystemEvolutionManager

                    self._delegate = SystemEvolutionManager(bots=[self._bot_name])
                except Exception as exc:  # pragma: no cover - degraded path
                    logger.warning(
                        "SystemEvolutionManager unavailable; using inert fallback: %s",
                        exc,
                    )
                    self._delegate = _NullEvolutionManager()
            return self._delegate

    def __getattr__(self, item: str) -> Any:
        return getattr(self._ensure_delegate(), item)


@dataclass(slots=True)
class _NullEvolutionResult:
    """Placeholder result mimicking :class:`EvolutionCycleResult`."""

    ga_results: dict[str, float]
    predictions: list[Any]
    trending_topic: str | None = None


class _NullEvolutionManager:
    """Minimal stand-in used when the real manager cannot be constructed."""

    bots: list[str]

    def __init__(self) -> None:
        self.bots = []

    def run_if_signals(self, **_kwargs: Any) -> None:
        return None

    def run_cycle(self) -> _NullEvolutionResult:  # pragma: no cover - stub
        return _NullEvolutionResult(ga_results={}, predictions=[])

    def radar_refactors(self) -> list[Any]:  # pragma: no cover - stub
        return []


_shared_orchestrator: "EvolutionOrchestrator" | None = None


@contextmanager
def _capital_bot_manual_mode(capital_cls: type[Any]) -> Iterator[None]:
    """Temporarily disable self-coding requirements for ``CapitalManagementBot``.

    ``CapitalManagementBot`` is decorated with :func:`self_coding_managed`,
    which normally enforces the presence of a :class:`SelfCodingManager`
    instance when the self-coding runtime is available. The shared orchestrator
    is used early during bootstrap before a manager exists, so the instantiation
    would raise ``RuntimeError``. The decorator exposes a
    ``_self_coding_manual_mode`` flag to bypass this requirement. This context
    manager toggles the flag only for the duration of the instantiation,
    restoring the previous value afterwards.
    """

    unset = object()
    previous = getattr(capital_cls, "_self_coding_manual_mode", unset)
    capital_cls._self_coding_manual_mode = True  # type: ignore[attr-defined]
    try:
        yield
    finally:
        if previous is unset:
            delattr(capital_cls, "_self_coding_manual_mode")
        else:
            capital_cls._self_coding_manual_mode = previous  # type: ignore[attr-defined]


def get_orchestrator(
    bot_name: str,
    data_bot: "DataBot",
    engine: "SelfCodingEngine",
    *,
    bootstrap_state: Mapping[str, object] | None = None,
) -> "EvolutionOrchestrator":
    """Return a singleton ``EvolutionOrchestrator``."""

    print("[debug] get_orchestrator invoked")
    _bootstrap_placeholders(bootstrap_state=bootstrap_state)
    global _shared_orchestrator
    if _shared_orchestrator is None:
        from .evolution_orchestrator import EvolutionOrchestrator
        from .capital_management_bot import CapitalManagementBot

        with _capital_bot_manual_mode(CapitalManagementBot):
            capital_bot = CapitalManagementBot()
        evolution_manager = _LazyEvolutionManager(bot_name)
        _shared_orchestrator = EvolutionOrchestrator(
            data_bot, capital_bot, engine, evolution_manager
        )
        print(f"[debug] Orchestrator loaded: {_shared_orchestrator}")
    else:
        print(f"[debug] Orchestrator loaded: {_shared_orchestrator}")
    return _shared_orchestrator


__all__ = ["get_orchestrator"]
