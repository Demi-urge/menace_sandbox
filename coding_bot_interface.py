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
import time

try:  # pragma: no cover - optional self-coding dependency
    from .self_coding_manager import SelfCodingManager
except ImportError:  # pragma: no cover - self-coding unavailable
    SelfCodingManager = Any  # type: ignore
try:  # pragma: no cover - allow tests to stub engine
    from .self_coding_engine import MANAGER_CONTEXT
except Exception as exc:  # pragma: no cover - fail fast when engine unavailable
    raise ImportError("Self-coding engine is required for operation") from exc

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
    obj: Any,
    registry: BotRegistry | None,
    data_bot: DataBot | None,
    orchestrator: EvolutionOrchestrator | None,
) -> tuple[
    BotRegistry,
    DataBot,
    EvolutionOrchestrator | None,
    str,
    SelfCodingManager | None,
]:
    """Resolve helper objects for *obj*.

    ``BotRegistry`` and ``DataBot`` are mandatory helpers.  When available, the
    existing ``EvolutionOrchestrator`` and ``SelfCodingManager`` references are
    also returned so callers can reuse or extend them.  A
    :class:`RuntimeError` is raised if either ``BotRegistry`` or ``DataBot``
    cannot be resolved from the provided arguments, instance attributes or the
    ``manager`` attribute.
    """

    manager = getattr(obj, "manager", None)

    if registry is None:
        registry = getattr(obj, "bot_registry", None)
        if registry is None and manager is not None:
            registry = getattr(manager, "bot_registry", None)
    if registry is None:
        raise RuntimeError("BotRegistry is required but was not provided")

    if data_bot is None:
        data_bot = getattr(obj, "data_bot", None)
        if data_bot is None and manager is not None:
            data_bot = getattr(manager, "data_bot", None)
    if data_bot is None:
        raise RuntimeError("DataBot is required but was not provided")

    if orchestrator is None:
        orchestrator = getattr(obj, "evolution_orchestrator", None)
        if orchestrator is None and manager is not None:
            orchestrator = getattr(manager, "evolution_orchestrator", None)

    try:
        module_path = inspect.getfile(obj.__class__)
    except Exception:  # pragma: no cover - best effort
        module_path = ""

    return registry, data_bot, orchestrator, module_path, manager


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
            (
                registry,
                data_bot,
                orchestrator,
                module_path,
                manager,
            ) = _resolve_helpers(self, registry, data_bot, orchestrator)
        except RuntimeError as exc:
            raise RuntimeError(f"{cls.__name__}: {exc}") from exc

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

        if orchestrator is None:
            try:
                from .capital_management_bot import CapitalManagementBot  # type: ignore
                from .self_improvement.engine import SelfImprovementEngine  # type: ignore
                from .system_evolution_manager import SystemEvolutionManager  # type: ignore
                from .evolution_orchestrator import EvolutionOrchestrator as _EO

                capital = CapitalManagementBot(data_bot=data_bot)
                improv = SelfImprovementEngine(data_bot=data_bot, bot_name=name)
                bot_list: list[str] = []
                try:
                    bot_list = list(getattr(registry, "graph", {}).keys())
                except Exception:
                    bot_list = []
                evol_mgr = SystemEvolutionManager(bot_list)
                orchestrator = _EO(
                    data_bot=data_bot,
                    capital_bot=capital,
                    improvement_engine=improv,
                    evolution_manager=evol_mgr,
                    selfcoding_manager=manager,
                )
                self.evolution_orchestrator = orchestrator
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    f"{cls.__name__}: EvolutionOrchestrator is required but could not be instantiated"
                ) from exc
        if getattr(manager, "quick_fix", None) is None:
            try:
                from .quick_fix_engine import QuickFixEngine  # type: ignore
                from .error_bot import ErrorDB
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    f"{cls.__name__}: QuickFixEngine is required but could not be imported",
                ) from exc
            engine = getattr(manager, "engine", None)
            clayer = getattr(engine, "cognition_layer", None)
            builder = getattr(clayer, "context_builder", None)
            if builder is None:
                raise RuntimeError(
                    f"{cls.__name__}: QuickFixEngine requires a context_builder",
                )
            error_db = getattr(self, "error_db", None) or getattr(manager, "error_db", None)
            if error_db is None:
                try:
                    error_db = ErrorDB()
                except Exception as exc:  # pragma: no cover - instantiation errors
                    raise RuntimeError(
                        f"{cls.__name__}: failed to initialise ErrorDB for QuickFixEngine",
                    ) from exc
            try:
                manager.quick_fix = QuickFixEngine(error_db, manager, context_builder=builder)
                manager.error_db = error_db
            except Exception as exc:  # pragma: no cover - instantiation errors
                raise RuntimeError(
                    f"{cls.__name__}: failed to initialise QuickFixEngine",
                ) from exc
        registry.register_bot(name)
        try:
            registry.update_bot(name, module_path)
        except Exception:  # pragma: no cover - best effort
            logger.exception("bot update failed for %s", name)
        try:
            orchestrator.register_bot(name)
            logger.info("registered %s with EvolutionOrchestrator", name)
        except Exception:  # pragma: no cover - best effort
            logger.exception(
                "evolution orchestrator registration failed for %s", name
            )
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

    for method_name in ("run", "execute"):
        orig_method = getattr(cls, method_name, None)
        if callable(orig_method):
            @wraps(orig_method)
            def wrapped_method(self, *args: Any, _orig=orig_method, **kwargs: Any):
                start = time.time()
                errors = 0
                try:
                    result = _orig(self, *args, **kwargs)
                except Exception:
                    errors = 1
                    raise
                finally:
                    response_time = time.time() - start
                    try:
                        data_bot = getattr(self, "data_bot", None)
                        if data_bot is None:
                            manager = getattr(self, "manager", None)
                            data_bot = getattr(manager, "data_bot", None)
                        if data_bot:
                            name = getattr(self, "name", getattr(self, "bot_name", cls.__name__))
                            data_bot.collect(name, response_time=response_time, errors=errors)
                    except Exception:  # pragma: no cover - best effort
                        logger.exception("failed logging metrics for %s", cls.__name__)
                return result

            setattr(cls, method_name, wrapped_method)

    return cls
