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

import contextvars
import importlib.util
import sys
from pathlib import Path
from functools import wraps
import inspect
import json
import logging
from types import SimpleNamespace
from typing import Any, Callable, TypeVar, TYPE_CHECKING
import time

_HELPER_NAME = "import_compat"
_PACKAGE_NAME = "menace_sandbox"

try:  # pragma: no cover - prefer package import when installed
    from menace_sandbox import import_compat  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - support flat execution
    _helper_path = Path(__file__).resolve().parent / f"{_HELPER_NAME}.py"
    _spec = importlib.util.spec_from_file_location(
        f"{_PACKAGE_NAME}.{_HELPER_NAME}",
        _helper_path,
    )
    if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
        raise
    import_compat = importlib.util.module_from_spec(_spec)
    sys.modules[f"{_PACKAGE_NAME}.{_HELPER_NAME}"] = import_compat
    sys.modules[_HELPER_NAME] = import_compat
    _spec.loader.exec_module(import_compat)
else:  # pragma: no cover - ensure helper aliases exist
    sys.modules.setdefault(_HELPER_NAME, import_compat)
    sys.modules.setdefault(f"{_PACKAGE_NAME}.{_HELPER_NAME}", import_compat)

import_compat.bootstrap(__name__, __file__)
load_internal = import_compat.load_internal

logger = logging.getLogger(__name__)

create_context_builder = load_internal("context_builder_util").create_context_builder


class _ThresholdModuleFallback:
    """Gracefully degrade when ``self_coding_thresholds`` is unavailable.

    The Windows command prompt environments that ship with the autonomous
    sandbox frequently omit scientific Python dependencies such as
    :mod:`pydantic`.  Importing :mod:`self_coding_thresholds` in that situation
    raises a :class:`ModuleNotFoundError` which used to abort the import of this
    module entirely, cascading into repeated internalisation retries for coding
    bots.  The fallback implementation below keeps the decorator operational
    while clearly surfacing the degraded behaviour via structured logging.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self._update_logged = False
        self._load_logged = False

    def update_thresholds(self, name: str, *args: Any, **kwargs: Any) -> None:
        if not self._update_logged:
            logger.warning(
                "self_coding_thresholds unavailable; skipping threshold updates (%s)",
                self.reason,
            )
            self._update_logged = True
        else:
            logger.debug(
                "suppressed threshold update for %s due to missing dependencies", name
            )

    def load_config(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if not self._load_logged:
            logger.info(
                "returning empty self-coding threshold config because dependencies are missing (%s)",
                self.reason,
            )
            self._load_logged = True
        return {}


try:
    _self_coding_thresholds = load_internal("self_coding_thresholds")
except ModuleNotFoundError as exc:
    fallback = _ThresholdModuleFallback(f"module not found: {exc}")
    update_thresholds = fallback.update_thresholds

    def _load_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return fallback.load_config(*args, **kwargs)

except Exception as exc:  # pragma: no cover - defensive degradation
    fallback = _ThresholdModuleFallback(f"import failure: {exc}")
    update_thresholds = fallback.update_thresholds

    def _load_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return fallback.load_config(*args, **kwargs)

else:
    update_thresholds = _self_coding_thresholds.update_thresholds
    _load_config = _self_coding_thresholds._load_config

try:  # pragma: no cover - optional self-coding dependency
    SelfCodingManager = load_internal("self_coding_manager").SelfCodingManager
except ModuleNotFoundError as exc:  # pragma: no cover - degrade gracefully when absent
    fallback_mod = sys.modules.get("menace.self_coding_manager")
    if fallback_mod and hasattr(fallback_mod, "SelfCodingManager"):
        SelfCodingManager = getattr(fallback_mod, "SelfCodingManager")  # type: ignore[assignment]
        logger.debug(
            "using stub SelfCodingManager from menace.self_coding_manager after import failure", exc_info=exc
        )
    else:
        logger.warning(
            "self_coding_manager could not be imported; self-coding will run in disabled mode",
            exc_info=exc,
        )
        SelfCodingManager = Any  # type: ignore
except Exception as exc:  # pragma: no cover - degrade gracefully when unavailable
    fallback_mod = sys.modules.get("menace.self_coding_manager")
    if fallback_mod and hasattr(fallback_mod, "SelfCodingManager"):
        SelfCodingManager = getattr(fallback_mod, "SelfCodingManager")  # type: ignore[assignment]
        logger.debug(
            "using stub SelfCodingManager from menace.self_coding_manager after runtime error", exc_info=exc
        )
    else:
        logger.warning(
            "self_coding_manager import failed; self-coding will run in disabled mode",
            exc_info=exc,
        )
        SelfCodingManager = Any  # type: ignore

_ENGINE_AVAILABLE = True
_ENGINE_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - allow tests to stub engine
    _self_coding_engine = load_internal("self_coding_engine")
except ModuleNotFoundError as exc:  # pragma: no cover - propagate requirement
    fallback_engine = sys.modules.get("menace.self_coding_engine")
    if fallback_engine is not None:
        _self_coding_engine = fallback_engine  # type: ignore[assignment]
        MANAGER_CONTEXT = getattr(
            fallback_engine,
            "MANAGER_CONTEXT",
            contextvars.ContextVar("MANAGER_CONTEXT", default=None),
        )
        _ENGINE_AVAILABLE = True
        _ENGINE_IMPORT_ERROR = None
        logger.debug(
            "using stub self_coding_engine from menace.self_coding_engine after import failure",
            exc_info=exc,
        )
    else:
        _ENGINE_AVAILABLE = False
        _ENGINE_IMPORT_ERROR = exc
        MANAGER_CONTEXT = contextvars.ContextVar("MANAGER_CONTEXT", default=None)
except Exception as exc:  # pragma: no cover - fail fast when engine unavailable
    fallback_engine = sys.modules.get("menace.self_coding_engine")
    if fallback_engine is not None:
        _self_coding_engine = fallback_engine  # type: ignore[assignment]
        MANAGER_CONTEXT = getattr(
            fallback_engine,
            "MANAGER_CONTEXT",
            contextvars.ContextVar("MANAGER_CONTEXT", default=None),
        )
        _ENGINE_AVAILABLE = True
        _ENGINE_IMPORT_ERROR = None
        logger.debug(
            "using stub self_coding_engine from menace.self_coding_engine after runtime error",
            exc_info=exc,
        )
    else:
        _ENGINE_AVAILABLE = False
        _ENGINE_IMPORT_ERROR = exc
        MANAGER_CONTEXT = contextvars.ContextVar("MANAGER_CONTEXT", default=None)
else:
    MANAGER_CONTEXT = getattr(
        _self_coding_engine,
        "MANAGER_CONTEXT",
        contextvars.ContextVar("MANAGER_CONTEXT", default=None),
    )
    if not hasattr(_self_coding_engine, "MANAGER_CONTEXT"):
        _ENGINE_AVAILABLE = False
        _ENGINE_IMPORT_ERROR = AttributeError(
            "self_coding_engine.MANAGER_CONTEXT is not available"
        )

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from menace_sandbox.bot_registry import BotRegistry
    from menace_sandbox.data_bot import DataBot
    from menace_sandbox.evolution_orchestrator import EvolutionOrchestrator
else:  # pragma: no cover - runtime placeholders
    BotRegistry = Any  # type: ignore
    DataBot = Any  # type: ignore
    EvolutionOrchestrator = Any  # type: ignore


if not _ENGINE_AVAILABLE and _ENGINE_IMPORT_ERROR is not None:
    logger.warning(
        "self_coding_engine could not be imported; coding bot helpers are in limited mode",
        exc_info=_ENGINE_IMPORT_ERROR,
    )


def _self_coding_runtime_available() -> bool:
    return _ENGINE_AVAILABLE and isinstance(SelfCodingManager, type)


class _DisabledSelfCodingManager:
    """Fallback manager used when the runtime cannot support self-coding."""

    __slots__ = (
        "bot_registry",
        "data_bot",
        "engine",
        "quick_fix",
        "error_db",
        "evolution_orchestrator",
        "_last_patch_id",
        "_last_commit_hash",
    )

    def __init__(self, *, bot_registry: Any, data_bot: Any) -> None:
        self.bot_registry = bot_registry
        self.data_bot = data_bot
        self.engine = SimpleNamespace(
            cognition_layer=SimpleNamespace(context_builder=None)
        )
        # Mark quick_fix as initialised so downstream code skips heavy bootstrap.
        self.quick_fix = object()
        self.error_db = None
        self.evolution_orchestrator = None
        self._last_patch_id = None
        self._last_commit_hash = None

    def register_patch_cycle(self, *_args: Any, **_kwargs: Any) -> None:
        logger.debug(
            "self-coding disabled; ignoring register_patch_cycle invocation"
        )

    def run_post_patch_cycle(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        logger.debug(
            "self-coding disabled; ignoring run_post_patch_cycle invocation"
        )
        return {}

    def refresh_quick_fix_context(self) -> Any:
        return getattr(self.engine, "context_builder", None)


def _bootstrap_manager(
    name: str,
    bot_registry: BotRegistry,
    data_bot: DataBot,
) -> Any:
    """Instantiate a ``SelfCodingManager`` with progressive fallbacks."""

    if not _self_coding_runtime_available():
        raise RuntimeError("self-coding runtime is unavailable")

    try:
        return SelfCodingManager(  # type: ignore[call-arg]
            bot_registry=bot_registry,
            data_bot=data_bot,
        )
    except TypeError:
        pass

    try:
        code_db_cls = _load_optional_module("code_database").CodeDB
        memory_cls = _load_optional_module("gpt_memory").GPTMemoryManager
        engine_mod = _load_optional_module(
            "self_coding_engine", fallback="menace.self_coding_engine"
        )
        pipeline_mod = _load_optional_module(
            "model_automation_pipeline", fallback="menace.model_automation_pipeline"
        )
        ctx_builder = create_context_builder()
        engine = engine_mod.SelfCodingEngine(
            code_db_cls(),
            memory_cls(),
            context_builder=ctx_builder,
        )
        pipeline = pipeline_mod.ModelAutomationPipeline(
            context_builder=ctx_builder,
            bot_registry=bot_registry,
        )
        manager_mod = _load_optional_module(
            "self_coding_manager", fallback="menace.self_coding_manager"
        )
        return manager_mod.SelfCodingManager(
            engine,
            pipeline,
            bot_name=name,
            data_bot=data_bot,
            bot_registry=bot_registry,
        )
    except Exception as exc:  # pragma: no cover - heavy bootstrap fallback
        raise RuntimeError(f"manager bootstrap failed: {exc}") from exc


def _load_optional_module(name: str, *, fallback: str | None = None) -> Any:
    """Attempt to load *name* falling back to ``fallback`` module when present."""

    try:
        return load_internal(name)
    except ModuleNotFoundError as exc:
        if fallback:
            module = sys.modules.get(fallback)
            if module is not None:
                logger.debug(
                    "using stub module %s for %s after import error",
                    fallback,
                    name,
                    exc_info=exc,
                )
                return module
        raise
    except Exception as exc:
        if fallback:
            module = sys.modules.get(fallback)
            if module is not None:
                logger.debug(
                    "using stub module %s for %s after runtime error",
                    fallback,
                    name,
                    exc_info=exc,
                )
                return module
        raise


F = TypeVar("F", bound=Callable[..., Any])


if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from vector_service.context_builder import ContextBuilder
else:  # pragma: no cover - runtime placeholder avoids hard dependency
    ContextBuilder = Any  # type: ignore


def manager_generate_helper(
    manager: SelfCodingManager,
    description: str,
    *,
    context_builder: "ContextBuilder",
    **kwargs: Any,
) -> str:
    """Invoke :meth:`SelfCodingEngine.generate_helper` under a manager token."""

    if not _ENGINE_AVAILABLE:
        message = "Self-coding engine is unavailable"
        if _ENGINE_IMPORT_ERROR is not None:
            message = f"{message}: {_ENGINE_IMPORT_ERROR}"
        raise RuntimeError(message)

    if context_builder is None:  # pragma: no cover - defensive
        raise TypeError("context_builder is required")

    engine = getattr(manager, "engine", None)
    if engine is None:  # pragma: no cover - defensive guard
        raise RuntimeError("manager must provide an engine for helper generation")

    token = MANAGER_CONTEXT.set(manager)
    previous_builder = getattr(engine, "context_builder", None)
    try:
        engine.context_builder = context_builder
        return engine.generate_helper(description, **kwargs)
    finally:
        engine.context_builder = previous_builder
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
        if _self_coding_runtime_available():
            try:
                name_local = getattr(
                    obj,
                    "name",
                    getattr(obj, "bot_name", obj.__class__.__name__),
                )
                manager = _bootstrap_manager(name_local, registry, data_bot)
            except Exception as exc:
                logger.warning(
                    "SelfCodingManager bootstrap failed for %s: %s",
                    name_local,
                    exc,
                )
                manager = _DisabledSelfCodingManager(
                    bot_registry=registry,
                    data_bot=data_bot,
                )
        else:
            manager = _DisabledSelfCodingManager(
                bot_registry=registry,
                data_bot=data_bot,
            )

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
        manager_instance = manager
        if manager_instance is None:
            if _self_coding_runtime_available():
                try:
                    manager_instance = _bootstrap_manager(name, bot_registry, data_bot)
                except Exception as exc:
                    logger.warning(
                        "automatic SelfCodingManager bootstrap failed for %s: %s",
                        name,
                        exc,
                    )
                    manager_instance = _DisabledSelfCodingManager(
                        bot_registry=bot_registry,
                        data_bot=data_bot,
                    )
            else:
                manager_instance = _DisabledSelfCodingManager(
                    bot_registry=bot_registry,
                    data_bot=data_bot,
                )
                logger.warning(
                    "self-coding runtime unavailable; %s will run without autonomous patching",
                    name,
                )

        register_kwargs = dict(
            name=name,
            roi_threshold=roi_t,
            error_threshold=err_t,
            manager=manager_instance,
            data_bot=data_bot,
            module_path=module_path,
            is_coding_bot=True,
        )
        try:
            bot_registry.register_bot(**register_kwargs)
        except TypeError:  # pragma: no cover - legacy registries
            try:
                fallback_kwargs = dict(register_kwargs)
                fallback_kwargs.pop("roi_threshold", None)
                fallback_kwargs.pop("error_threshold", None)
                bot_registry.register_bot(**fallback_kwargs)
            except TypeError:
                bot_registry.register_bot(name, is_coding_bot=True)
                if module_path:
                    try:
                        node = bot_registry.graph.nodes.get(name)
                        if node is not None:
                            node["module"] = module_path
                        bot_registry.modules[name] = module_path
                    except Exception:  # pragma: no cover - best effort bookkeeping
                        logger.debug(
                            "failed to persist module path for %s after legacy registration",
                            name,
                            exc_info=True,
                        )
        registries_seen = getattr(cls, "_self_coding_registry_ids", None)
        if not isinstance(registries_seen, set):
            registries_seen = set()
        registry_managers = getattr(cls, "_self_coding_registry_managers", None)
        if not isinstance(registry_managers, dict):
            registry_managers = {}
        registry_id = id(bot_registry)
        registries_seen.add(registry_id)
        registry_managers[registry_id] = manager_instance
        cls._self_coding_registry_ids = registries_seen
        cls._self_coding_registry_managers = registry_managers
        update_kwargs: dict[str, Any] = {}
        should_update = True
        try:
            update_sig = inspect.signature(bot_registry.update_bot)
        except (AttributeError, TypeError, ValueError):  # pragma: no cover - best effort
            update_sig = None

        if update_sig is not None:
            params = update_sig.parameters
            expects_provenance = "patch_id" in params and "commit" in params
        else:  # pragma: no cover - defensive default
            expects_provenance = False

        if expects_provenance:
            patch_id = None
            commit = None
            manager_sources = []
            if manager is not None:
                manager_sources.append(manager)
            try:
                ctx_manager = MANAGER_CONTEXT.get(None)
            except LookupError:  # pragma: no cover - defensive
                ctx_manager = None
            if ctx_manager is not None and ctx_manager not in manager_sources:
                manager_sources.append(ctx_manager)

            for candidate in manager_sources:
                patch_id = patch_id or getattr(candidate, "_last_patch_id", None)
                commit = commit or getattr(candidate, "_last_commit_hash", None)
                if patch_id is not None and commit is not None:
                    break

            if patch_id is not None and commit is None:
                try:  # pragma: no cover - best effort metadata recovery
                    from .patch_provenance import PatchProvenanceService

                    service = PatchProvenanceService()
                    record = service.db.get(patch_id)
                    summary = getattr(record, "summary", None)
                    if summary:
                        try:
                            commit = json.loads(summary).get("commit")
                        except Exception:
                            commit = None
                except Exception:
                    logger.debug(
                        "failed to backfill provenance for %s", name,
                        exc_info=True,
                    )

            if patch_id is not None and commit is not None:
                update_kwargs["patch_id"] = patch_id
                update_kwargs["commit"] = commit
            else:
                should_update = False
                logger.info(
                    "skipping bot update for %s due to missing provenance metadata",
                    name,
                )

        if should_update:
            try:
                bot_registry.update_bot(name, module_path, **update_kwargs)
            except Exception:  # pragma: no cover - best effort
                logger.exception("bot update failed for %s", name)

        cls.bot_registry = bot_registry  # type: ignore[attr-defined]
        cls.data_bot = data_bot  # type: ignore[attr-defined]
        cls.manager = manager_instance  # type: ignore[attr-defined]

        @wraps(orig_init)
        def wrapped_init(self, *args: Any, **kwargs: Any) -> None:
            orchestrator: EvolutionOrchestrator | None = kwargs.pop(
                "evolution_orchestrator", None
            )
            manager_local: SelfCodingManager | None = kwargs.get(
                "manager", manager_instance
            )
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
            if _self_coding_runtime_available() and not isinstance(
                manager_local, SelfCodingManager
            ):
                raise RuntimeError("SelfCodingManager instance is required")
            self.manager = manager_local

            if not _self_coding_runtime_available():
                if orchestrator is not None:
                    self.evolution_orchestrator = orchestrator
                return

            orchestrator_boot_failed = False
            if orchestrator is None:
                try:
                    _capital_module = _load_optional_module(
                        "capital_management_bot",
                        fallback="menace.capital_management_bot",
                    )
                    CapitalManagementBot = _capital_module.CapitalManagementBot
                    _improvement_module = _load_optional_module(
                        "self_improvement.engine",
                        fallback="menace.self_improvement.engine",
                    )
                    SelfImprovementEngine = _improvement_module.SelfImprovementEngine
                    _evolution_manager_module = _load_optional_module(
                        "system_evolution_manager",
                        fallback="menace.system_evolution_manager",
                    )
                    SystemEvolutionManager = (
                        _evolution_manager_module.SystemEvolutionManager
                    )
                    _eo_module = _load_optional_module(
                        "evolution_orchestrator",
                        fallback="menace.evolution_orchestrator",
                    )
                    _EO = _eo_module.EvolutionOrchestrator

                    capital = CapitalManagementBot(data_bot=d_bot)
                    builder = create_context_builder()
                    improv = SelfImprovementEngine(
                        context_builder=builder,
                        data_bot=d_bot,
                        bot_name=name_local,
                    )
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
                except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                    logger.warning(
                        "%s: EvolutionOrchestrator dependencies unavailable: %s",
                        cls.__name__,
                        exc,
                    )
                    orchestrator_boot_failed = True
                    orchestrator = None
                except Exception as exc:  # pragma: no cover - optional dependency
                    raise RuntimeError(
                        f"{cls.__name__}: EvolutionOrchestrator is required but could not be instantiated"
                    ) from exc

            if orchestrator is not None:
                self.evolution_orchestrator = orchestrator
                try:
                    manager_local.evolution_orchestrator = orchestrator
                except Exception:
                    pass
            elif orchestrator_boot_failed:
                self.evolution_orchestrator = None

            if getattr(manager_local, "quick_fix", None) is None:
                try:
                    _quick_fix_module = _load_optional_module(
                        "quick_fix_engine", fallback="menace.quick_fix_engine"
                    )
                    QuickFixEngine = _quick_fix_module.QuickFixEngine
                    ErrorDB = _load_optional_module(
                        "error_bot", fallback="menace.error_bot"
                    ).ErrorDB
                    _helper_fn = _load_optional_module(
                        "self_coding_manager", fallback="menace.self_coding_manager"
                    )._manager_generate_helper_with_builder
                except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                    logger.warning(
                        "%s: QuickFixEngine dependencies unavailable: %s",
                        cls.__name__,
                        exc,
                    )
                    manager_local.quick_fix = manager_local.quick_fix or None
                    ErrorDB = None
                    QuickFixEngine = None
                except Exception as exc:  # pragma: no cover - optional dependency
                    logger.warning(
                        "%s: QuickFixEngine initialisation failed: %s",
                        cls.__name__,
                        exc,
                    )
                    manager_local.quick_fix = manager_local.quick_fix or None
                    ErrorDB = None
                    QuickFixEngine = None
                if QuickFixEngine is not None and ErrorDB is not None:
                    engine = getattr(manager_local, "engine", None)
                    clayer = getattr(engine, "cognition_layer", None)
                    if clayer is None:
                        logger.warning(
                            "%s: QuickFixEngine requires a cognition_layer; skipping bootstrap",
                            cls.__name__,
                        )
                    else:
                        try:
                            builder = clayer.context_builder
                        except AttributeError as exc:
                            logger.warning(
                                "%s: QuickFixEngine missing context_builder: %s",
                                cls.__name__,
                                exc,
                            )
                        else:
                            error_db = getattr(self, "error_db", None) or getattr(
                                manager_local, "error_db", None
                            )
                            if error_db is None:
                                try:
                                    error_db = ErrorDB()
                                except Exception as exc:  # pragma: no cover
                                    logger.warning(
                                        "%s: failed to initialise ErrorDB for QuickFixEngine: %s",
                                        cls.__name__,
                                        exc,
                                    )
                                    error_db = None
                            if error_db is not None:
                                try:
                                    manager_local.quick_fix = QuickFixEngine(
                                        error_db,
                                        manager_local,
                                        context_builder=builder,
                                        helper_fn=_helper_fn,
                                    )
                                    manager_local.error_db = error_db
                                except Exception as exc:  # pragma: no cover - instantiation errors
                                    logger.warning(
                                        "%s: failed to initialise QuickFixEngine: %s",
                                        cls.__name__,
                                        exc,
                                    )
                                    manager_local.quick_fix = manager_local.quick_fix or None
            registries_seen = getattr(cls, "_self_coding_registry_ids", set())
            if not isinstance(registries_seen, set):
                registries_seen = set()
            registry_managers = getattr(cls, "_self_coding_registry_managers", None)
            if not isinstance(registry_managers, dict):
                registry_managers = {}
            registry_id = id(registry)
            previous_manager = registry_managers.get(registry_id)
            new_disabled = isinstance(manager_local, _DisabledSelfCodingManager)
            previous_disabled = isinstance(previous_manager, _DisabledSelfCodingManager)
            manager_replaced = (
                previous_manager is not None and previous_manager is not manager_local
            )
            should_register = (
                registry_id not in registries_seen
                or (previous_disabled and not new_disabled)
                or (manager_replaced and not (previous_disabled and new_disabled))
                or (registry_id not in registry_managers and not new_disabled)
            )
            if should_register:
                try:
                    registry.register_bot(
                        name_local,
                        roi_threshold=getattr(thresholds, "roi_drop", None),
                        error_threshold=getattr(thresholds, "error_threshold", None),
                        manager=manager_local,
                        data_bot=d_bot,
                        is_coding_bot=True,
                    )
                except Exception:  # pragma: no cover - best effort
                    logger.exception("bot registration failed for %s", name_local)
                else:
                    registries_seen.add(registry_id)
                    registry_managers[registry_id] = manager_local
                    cls._self_coding_registry_ids = registries_seen
                    cls._self_coding_registry_managers = registry_managers
            else:
                registry_managers[registry_id] = manager_local
                cls._self_coding_registry_managers = registry_managers
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
