from __future__ import annotations

"""Utilities for registering coding bots with the central registries.

Note:
    Always decorate new coding bot classes with ``@self_coding_managed`` so
    they are automatically registered with the system's helpers.
"""

from functools import wraps
import inspect
import logging
from typing import Any, Callable, TypeVar, TYPE_CHECKING
import contextvars

from .self_coding_manager import SelfCodingManager
try:  # pragma: no cover - allow tests to stub engine
    from .self_coding_engine import MANAGER_CONTEXT
except Exception:  # pragma: no cover - fallback when engine unavailable
    MANAGER_CONTEXT = contextvars.ContextVar("MANAGER_CONTEXT")

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from .bot_registry import BotRegistry
    from .data_bot import DataBot
    from .evolution_orchestrator import EvolutionOrchestrator
else:  # pragma: no cover - runtime placeholders
    BotRegistry = Any  # type: ignore
    DataBot = Any  # type: ignore
    EvolutionOrchestrator = Any  # type: ignore


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def manager_generate_helper(
    manager: SelfCodingManager, description: str, **kwargs: Any
) -> str:
    """Invoke :meth:`SelfCodingEngine.generate_helper` under a manager token."""
    token = MANAGER_CONTEXT.set(manager)
    try:
        return manager.engine.generate_helper(description, **kwargs)
    finally:
        MANAGER_CONTEXT.reset(token)


def _resolve_helpers(
    obj: Any, registry: BotRegistry | None, data_bot: DataBot | None
) -> tuple[BotRegistry, DataBot, str]:
    """Resolve ``BotRegistry``/``DataBot`` and return defining module path.

    ``inspect.getfile(obj.__class__)`` is used to determine the file where the
    class of ``obj`` is defined.  This path is returned alongside the resolved
    helpers so callers can update the registry with accurate module metadata.

    A :class:`RuntimeError` is raised if either dependency cannot be resolved
    from the provided arguments, instance attributes or ``manager`` attribute.
    """
    if registry is None:
        registry = getattr(obj, "bot_registry", None)
        if registry is None:
            manager = getattr(obj, "manager", None)
            registry = getattr(manager, "bot_registry", None)
    if registry is None:
        raise RuntimeError("BotRegistry is required but was not provided")

    if data_bot is None:
        data_bot = getattr(obj, "data_bot", None)
        if data_bot is None:
            manager = getattr(obj, "manager", None)
            data_bot = getattr(manager, "data_bot", None)
    if data_bot is None:
        raise RuntimeError("DataBot is required but was not provided")

    try:
        module_path = inspect.getfile(obj.__class__)
    except Exception:  # pragma: no cover - best effort
        module_path = ""
    return registry, data_bot, module_path


def self_coding_managed(cls: type) -> type:
    """Class decorator registering bots with :class:`BotRegistry` and :class:`DataBot`.

    Classes wrapped with ``@self_coding_managed`` automatically register their
    ``name`` (or ``bot_name`` attribute) with :class:`BotRegistry` and ensure
    baseline ROI/error metrics are logged via :class:`DataBot` on construction.
    The decorator looks for ``bot_registry`` and ``data_bot`` either passed as
    keyword arguments during initialisation or available on the instance or its
    ``manager`` attribute.  A ``manager`` attribute of type
    :class:`SelfCodingManager` is required; if missing the decorator will attempt
    to construct one from the resolved helpers and otherwise raise a
    :class:`RuntimeError`.
    """

    orig_init = cls.__init__  # type: ignore[attr-defined]

    @wraps(orig_init)
    def wrapped_init(self, *args: Any, **kwargs: Any) -> None:
        registry = kwargs.pop("bot_registry", None)
        data_bot = kwargs.pop("data_bot", None)
        orchestrator: EvolutionOrchestrator | None = kwargs.pop(
            "evolution_orchestrator", None
        )
        orig_init(self, *args, **kwargs)
        try:
            registry, data_bot, module_path = _resolve_helpers(
                self, registry, data_bot
            )
        except RuntimeError as exc:
            raise RuntimeError(f"{cls.__name__}: {exc}") from exc
        if orchestrator is None:
            orchestrator = getattr(self, "evolution_orchestrator", None)
            if orchestrator is None:
                manager_for_orch = getattr(self, "manager", None)
                orchestrator = getattr(manager_for_orch, "evolution_orchestrator", None)
        if orchestrator is None:
            raise RuntimeError(
                f"{cls.__name__}: EvolutionOrchestrator is required but was not provided"
            )
        manager = getattr(self, "manager", None)
        name = getattr(self, "name", getattr(self, "bot_name", cls.__name__))
        if not isinstance(manager, SelfCodingManager):
            try:
                manager = SelfCodingManager(
                    getattr(self, "engine", None),
                    getattr(self, "pipeline", None),
                    bot_name=name,
                    bot_registry=registry,
                    data_bot=data_bot,
                )
                self.manager = manager
            except Exception as exc:  # pragma: no cover - best effort
                raise RuntimeError(
                    "failed to initialise SelfCodingManager; provide a manager"
                ) from exc
        registry.register_bot(name)
        try:
            registry.update_bot(name, module_path)
        except Exception:  # pragma: no cover - best effort
            logger.exception("bot update failed for %s", name)
        orchestrator.register_bot(name)
        if data_bot and getattr(data_bot, "db", None):
            try:
                roi = float(data_bot.roi(name)) if hasattr(data_bot, "roi") else 0.0
                data_bot.db.log_eval(name, "roi", roi)
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed logging roi for %s", name)
            try:
                data_bot.db.log_eval(name, "errors", 0.0)
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed logging errors for %s", name)

    cls.__init__ = wrapped_init  # type: ignore[assignment]
    return cls
