from __future__ import annotations

"""Utilities for registering coding bots with the central registries.

Note:
    Always decorate new coding bot classes with ``@self_coding_managed`` so
    they are automatically registered with the system's helpers.  The decorator
    accepts a ``SelfCodingManager`` instance to reuse existing state across
    instances.

Example:
    >>> @self_coding_managed(
    ...     bot_registry=registry,
    ...     data_bot=data_bot,
    ...     manager=manager,
    ... )
    ... class ExampleBot:
    ...     ...
"""

from functools import wraps
import inspect
import logging
from typing import Any, Callable, TypeVar, TYPE_CHECKING
import time

from .self_coding_thresholds import update_thresholds, _load_config

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
    manager: SelfCodingManager | None,
) -> tuple[
    BotRegistry,
    DataBot,
    EvolutionOrchestrator | None,
    str,
    SelfCodingManager,
]:
    """Resolve helper objects for *obj*.

    ``BotRegistry`` and ``DataBot`` are mandatory helpers.  When available, the
    existing ``EvolutionOrchestrator`` reference is also returned so callers can
    reuse or extend it.  ``manager`` takes precedence over any existing
    ``manager`` attribute on the object.  A :class:`RuntimeError` is raised if
    any mandatory helper cannot be resolved.
    """

    manager = manager or getattr(obj, "manager", None)

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

    if manager is None:
        raise RuntimeError("SelfCodingManager is required but was not provided")

    return registry, data_bot, orchestrator, module_path, manager


def _ensure_threshold_entry(name: str, thresholds: Any) -> None:
    """Persist default threshold config for *name* when missing."""

    try:
        bots = (_load_config(None) or {}).get("bots", {})
    except Exception:  # pragma: no cover - best effort
        bots = {}
    if name in bots:
        return
    try:  # pragma: no cover - best effort
        update_thresholds(
            name,
            roi_drop=getattr(thresholds, "roi_drop", None),
            error_increase=getattr(thresholds, "error_threshold", None),
            test_failure_increase=getattr(thresholds, "test_failure_threshold", None),
        )
    except Exception:
        logger.exception("failed to persist thresholds for %s", name)


def self_coding_managed(
    *,
    bot_registry: BotRegistry,
    data_bot: DataBot,
    manager: SelfCodingManager | None = None,
) -> Callable[[type], type]:
    """Class decorator registering bots with helper services.

    ``bot_registry`` and ``data_bot`` instances must be provided when applying
    the decorator.  When ``manager`` is supplied, instances will default to this
    :class:`SelfCodingManager` when resolving helpers.  The bot's name is
    registered with ``bot_registry`` during class creation so registration does
    not depend on instantiation.
    """

    if bot_registry is None or data_bot is None:
        raise RuntimeError("BotRegistry and DataBot instances are required")

    def decorator(cls: type) -> type:
        orig_init = cls.__init__  # type: ignore[attr-defined]

        name = getattr(cls, "name", getattr(cls, "bot_name", cls.__name__))
        try:
            module_path = inspect.getfile(cls)
        except Exception:  # pragma: no cover - best effort
            module_path = ""
        roi_t = err_t = None
        if hasattr(data_bot, "reload_thresholds"):
            try:
                t = data_bot.reload_thresholds(name)
                roi_t = getattr(t, "roi_drop", None)
                err_t = getattr(t, "error_threshold", None)
                _ensure_threshold_entry(name, t)
            except Exception:  # pragma: no cover - best effort
                logger.exception("threshold reload failed for %s", name)
        try:
            bot_registry.register_bot(
                name,
                roi_threshold=roi_t,
                error_threshold=err_t,
                manager=manager,
                data_bot=data_bot,
            )
        except TypeError:  # pragma: no cover - legacy registries
            try:
                bot_registry.register_bot(name, manager=manager, data_bot=data_bot)
            except TypeError:
                bot_registry.register_bot(name)
        try:
            bot_registry.update_bot(name, module_path)
        except Exception:  # pragma: no cover - best effort
            logger.exception("bot update failed for %s", name)

        cls.bot_registry = bot_registry  # type: ignore[attr-defined]
        cls.data_bot = data_bot  # type: ignore[attr-defined]
        cls.manager = manager  # type: ignore[attr-defined]

        @wraps(orig_init)
        def wrapped_init(self, *args: Any, **kwargs: Any) -> None:
            orchestrator: EvolutionOrchestrator | None = kwargs.pop(
                "evolution_orchestrator", None
            )
            manager_local: SelfCodingManager | None = kwargs.get("manager", manager)
            orig_init(self, *args, **kwargs)
            try:
                (
                    registry,
                    d_bot,
                    orchestrator,
                    _module_path,
                    manager_local,
                ) = _resolve_helpers(
                    self, bot_registry, data_bot, orchestrator, manager_local
                )
            except RuntimeError as exc:
                raise RuntimeError(f"{cls.__name__}: {exc}") from exc

            name_local = getattr(self, "name", getattr(self, "bot_name", name))
            thresholds = None
            if hasattr(d_bot, "reload_thresholds"):
                try:
                    thresholds = d_bot.reload_thresholds(name_local)
                    update_thresholds(
                        name_local,
                        roi_drop=thresholds.roi_drop,
                        error_increase=thresholds.error_threshold,
                        test_failure_increase=thresholds.test_failure_threshold,
                    )
                except Exception:  # pragma: no cover - best effort
                    logger.exception(
                        "failed to initialise thresholds for %s", name_local
                    )
            if not isinstance(manager_local, SelfCodingManager):
                raise RuntimeError("SelfCodingManager instance is required")
            self.manager = manager_local

            if orchestrator is None:
                try:
                    from .capital_management_bot import CapitalManagementBot  # type: ignore
                    from .self_improvement.engine import SelfImprovementEngine  # type: ignore
                    from .system_evolution_manager import SystemEvolutionManager  # type: ignore
                    from .evolution_orchestrator import EvolutionOrchestrator as _EO

                    capital = CapitalManagementBot(data_bot=d_bot)
                    improv = SelfImprovementEngine(data_bot=d_bot, bot_name=name_local)
                    bot_list: list[str] = []
                    try:
                        bot_list = list(getattr(registry, "graph", {}).keys())
                    except Exception:
                        bot_list = []
                    evol_mgr = SystemEvolutionManager(bot_list)
                    orchestrator = _EO(
                        data_bot=d_bot,
                        capital_bot=capital,
                        improvement_engine=improv,
                        evolution_manager=evol_mgr,
                        selfcoding_manager=manager_local,
                    )
                except Exception as exc:  # pragma: no cover - optional dependency
                    raise RuntimeError(
                        f"{cls.__name__}: "
                        "EvolutionOrchestrator is required but could not be instantiated"
                    ) from exc

            if orchestrator is not None:
                self.evolution_orchestrator = orchestrator
                try:
                    manager_local.evolution_orchestrator = orchestrator
                except Exception:
                    pass
            if getattr(manager_local, "quick_fix", None) is None:
                try:
                    from .quick_fix_engine import QuickFixEngine  # type: ignore
                    from .error_bot import ErrorDB
                    from .self_coding_manager import (
                        _manager_generate_helper_with_builder as _helper_fn,
                    )
                except Exception as exc:  # pragma: no cover - optional dependency
                    raise RuntimeError(
                        f"{cls.__name__}: QuickFixEngine is required but could not be imported"
                    ) from exc
                engine = getattr(manager_local, "engine", None)
                clayer = getattr(engine, "cognition_layer", None)
                builder = getattr(clayer, "context_builder", None)
                if builder is None:
                    raise RuntimeError(
                        f"{cls.__name__}: QuickFixEngine requires a context_builder"
                    )
                error_db = getattr(self, "error_db", None) or getattr(
                    manager_local, "error_db", None
                )
                if error_db is None:
                    try:
                        error_db = ErrorDB()
                    except Exception as exc:  # pragma: no cover - instantiation errors
                        raise RuntimeError(
                            f"{cls.__name__}: failed to initialise ErrorDB for QuickFixEngine"
                        ) from exc
                try:
                    manager_local.quick_fix = QuickFixEngine(
                        error_db,
                        manager_local,
                        context_builder=builder,
                        helper_fn=_helper_fn,
                    )
                    manager_local.error_db = error_db
                except Exception as exc:  # pragma: no cover - instantiation errors
                    raise RuntimeError(
                        f"{cls.__name__}: failed to initialise QuickFixEngine"
                    ) from exc
            try:
                registry.register_bot(
                    name_local,
                    roi_threshold=getattr(thresholds, "roi_drop", None),
                    error_threshold=getattr(thresholds, "error_threshold", None),
                    manager=manager_local,
                    data_bot=d_bot,
                )
            except Exception:  # pragma: no cover - best effort
                logger.exception("bot registration failed for %s", name_local)
            if orchestrator is not None:
                try:
                    orchestrator.register_bot(name_local)
                    logger.info("registered %s with EvolutionOrchestrator", name_local)
                except Exception:  # pragma: no cover - best effort
                    logger.exception(
                        "evolution orchestrator registration failed for %s", name_local
                    )
            if d_bot and getattr(d_bot, "db", None):
                try:
                    roi = float(d_bot.roi(name_local)) if hasattr(d_bot, "roi") else 0.0
                    d_bot.db.log_eval(name_local, "roi", roi)
                except Exception:  # pragma: no cover - best effort
                    logger.exception("failed logging roi for %s", name_local)
                try:
                    d_bot.db.log_eval(name_local, "errors", 0.0)
                except Exception:  # pragma: no cover - best effort
                    logger.exception("failed logging errors for %s", name_local)

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
                            d_bot_local = getattr(self, "data_bot", None)
                            if d_bot_local is None:
                                manager = getattr(self, "manager", None)
                                d_bot_local = getattr(manager, "data_bot", None)
                            if d_bot_local:
                                name_local2 = getattr(
                                    self,
                                    "name",
                                    getattr(self, "bot_name", name),
                                )
                                d_bot_local.collect(
                                    name_local2,
                                    response_time=response_time,
                                    errors=errors,
                                )
                        except Exception:  # pragma: no cover - best effort
                            logger.exception("failed logging metrics for %s", name)
                    return result

                setattr(cls, method_name, wrapped_method)

        return cls

    return decorator
