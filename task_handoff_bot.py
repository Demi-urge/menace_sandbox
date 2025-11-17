"""Task Handoff Bot for packaging and sending tasks to Stage 4.

Embeddings auto-refresh via ``EmbeddingBackfill.watch_events``.
Run ``menace embed --db workflow`` to backfill manually.
"""

from __future__ import annotations

import importlib.util
import _thread
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import uuid
from functools import wraps
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    cast,
)
from urllib.parse import urlparse
import socket


logger = logging.getLogger(__name__)


def _can_reach_broker(mq_url: str, timeout: float = 0.25) -> bool:
    """Return True when the RabbitMQ broker is reachable."""

    try:
        parsed = urlparse(mq_url)
        if not parsed.hostname:
            return False
        port = parsed.port or 5672
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True
    except OSError as exc:
        logger.debug("RabbitMQ broker not reachable at %s: %s", mq_url, exc)
        return False
    except Exception as exc:
        logger.debug("RabbitMQ broker check failed for %s: %s", mq_url, exc)
        return False


_HELPER_NAME = "import_compat"
_PACKAGE_NAME = "menace_sandbox"

try:  # pragma: no cover - prefer package import when installed
    from menace_sandbox import import_compat  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - allow flat execution layout
    _helper_path = Path(__file__).resolve().parent / f"{_HELPER_NAME}.py"
    _spec = importlib.util.spec_from_file_location(
        f"{_PACKAGE_NAME}.{_HELPER_NAME}",
        _helper_path,
    )
    if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
        raise
    import_compat = importlib.util.module_from_spec(_spec)  # type: ignore[assignment]
    sys.modules[f"{_PACKAGE_NAME}.{_HELPER_NAME}"] = import_compat
    sys.modules[_HELPER_NAME] = import_compat
    _spec.loader.exec_module(import_compat)
else:  # pragma: no cover - ensure helper aliases exist
    sys.modules.setdefault(_HELPER_NAME, import_compat)
    sys.modules.setdefault(f"{_PACKAGE_NAME}.{_HELPER_NAME}", import_compat)

import_compat.bootstrap(__name__, __file__)
load_internal = import_compat.load_internal

BotRegistry = load_internal("bot_registry").BotRegistry
_data_bot_fallback = False
try:
    _data_bot_module = load_internal("data_bot")
except Exception as exc:  # pragma: no cover - optional dependency fallback
    logger.warning(
        "DataBot unavailable for TaskHandoffBot; using inert stub: %s",
        exc,
    )
    _data_bot_fallback = True

    class _FallbackDataBot:
        """Minimal stand-in when the data bot stack is unavailable."""

        def __init__(self, *args: Any, event_bus: Any | None = None, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs
            self.event_bus = event_bus
            self._degradation_callbacks: list[Callable[[dict[str, Any]], None]] = []

        def start(self) -> None:  # pragma: no cover - compatibility no-op
            return None

        def subscribe_degradation(self, callback: Callable[[dict[str, Any]], None]) -> None:
            self._degradation_callbacks.append(callback)

        def subscribe_threshold_breaches(self, callback: Callable[[dict[str, Any]], None]) -> None:
            self._degradation_callbacks.append(callback)

        def check_degradation(self, *args: Any, **kwargs: Any) -> bool:
            return False

        def reload_thresholds(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {}

    DataBot = _FallbackDataBot  # type: ignore[assignment]
else:
    try:
        DataBot = _data_bot_module.DataBot
    except AttributeError as exc:  # pragma: no cover - circular import guard
        module_name = f"{_PACKAGE_NAME}.data_bot"
        if _is_module_initializing(module_name):
            logger.debug(
                "DataBot import deferred because %s is still initialising; using proxy",
                module_name,
            )

            class _DeferredDataBot:
                """Proxy that resolves :class:`DataBot` once initialisation completes."""

                def __new__(cls, *args: Any, **kwargs: Any):
                    module = load_internal("data_bot")
                    try:
                        real_cls = getattr(module, "DataBot")
                    except AttributeError as real_exc:  # pragma: no cover - defensive
                        raise AttributeError(
                            "menace_sandbox.data_bot does not define DataBot"
                        ) from real_exc
                    return real_cls(*args, **kwargs)

            DataBot = cast("type[Any]", _DeferredDataBot)
        else:
            logger.warning(
                "DataBot attribute missing; falling back to inert stub: %s",
                exc,
            )
            _data_bot_fallback = True

            class _FallbackDataBot:
                """Minimal stand-in when the data bot stack is unavailable."""

                def __init__(
                    self,
                    *args: Any,
                    event_bus: Any | None = None,
                    **kwargs: Any,
                ) -> None:
                    self.args = args
                    self.kwargs = kwargs
                    self.event_bus = event_bus
                    self._degradation_callbacks: list[Callable[[dict[str, Any]], None]] = []

                def start(self) -> None:  # pragma: no cover - compatibility no-op
                    return None

                def subscribe_degradation(
                    self, callback: Callable[[dict[str, Any]], None]
                ) -> None:
                    self._degradation_callbacks.append(callback)

                def subscribe_threshold_breaches(
                    self, callback: Callable[[dict[str, Any]], None]
                ) -> None:
                    self._degradation_callbacks.append(callback)

                def check_degradation(self, *args: Any, **kwargs: Any) -> bool:
                    return False

                def reload_thresholds(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
                    return {}

            DataBot = cast("type[Any]", _FallbackDataBot)

_self_coding_import_warning_emitted = False
_self_coding_disabled_warning_emitted = False


def _is_module_initializing(module_name: str) -> bool:
    """Return ``True`` when *module_name* is still in the import bootstrap."""

    module = sys.modules.get(module_name)
    if module is None:
        return False
    spec = getattr(module, "__spec__", None)
    return bool(spec and getattr(spec, "_initializing", False))


@dataclass(slots=True)
class _DeferredSelfCodingDecoration:
    """Record pending decorator applications waiting for helpers."""

    cls: type
    kwargs: dict[str, Any]
    attempts: int = 0


_PENDING_SELF_CODING_DECORATIONS: list[_DeferredSelfCodingDecoration] = []
_PENDING_SELF_CODING_LOCK = threading.Lock()
_DEFERRED_DECORATION_TIMER: threading.Timer | None = None
_MAX_DECORATION_ATTEMPTS = 5


def _schedule_deferred_self_coding(delay: float = 0.1) -> None:
    """Ensure deferred decorator processing runs after imports settle."""

    global _DEFERRED_DECORATION_TIMER
    timer = _DEFERRED_DECORATION_TIMER
    if timer is not None and timer.is_alive():
        return
    try:
        timer = threading.Timer(delay, _process_pending_self_coding)
        timer.daemon = True
        timer.start()
    except Exception:  # pragma: no cover - timer setup best effort
        logger.exception("failed to schedule deferred self-coding decorator")
    else:
        _DEFERRED_DECORATION_TIMER = timer


def _apply_self_coding_decorator(item: _DeferredSelfCodingDecoration) -> bool:
    """Attempt to decorate ``item.cls`` once helpers are importable."""

    try:
        interface_mod = load_internal("coding_bot_interface")
    except Exception:
        logger.debug(
            "retrying deferred self-coding decorator for %s", item.cls.__name__, exc_info=True
        )
        return False

    decorator_factory = getattr(interface_mod, "self_coding_managed", None)
    if not callable(decorator_factory):
        logger.debug(
            "coding_bot_interface.self_coding_managed unavailable when decorating %s",
            item.cls.__name__,
        )
        return False

    try:
        decorated = decorator_factory(**item.kwargs)(item.cls)
    except Exception:
        logger.exception(
            "deferred self-coding decoration failed for %s", item.cls.__name__
        )
        return False

    module = sys.modules.get(item.cls.__module__)
    if module is not None:
        setattr(module, item.cls.__name__, decorated)
    globals()[item.cls.__name__] = decorated
    return True


def _process_pending_self_coding() -> None:
    """Attempt to apply any queued self-coding decorators."""

    global _DEFERRED_DECORATION_TIMER, _self_coding_import_warning_emitted
    with _PENDING_SELF_CODING_LOCK:
        pending = list(_PENDING_SELF_CODING_DECORATIONS)
        _PENDING_SELF_CODING_DECORATIONS.clear()
        _DEFERRED_DECORATION_TIMER = None

    if not pending:
        return

    retry: list[_DeferredSelfCodingDecoration] = []
    for item in pending:
        if _apply_self_coding_decorator(item):
            continue
        item.attempts += 1
        if item.attempts < _MAX_DECORATION_ATTEMPTS:
            retry.append(item)
        else:
            if not _self_coding_import_warning_emitted:
                logger.warning(
                    "Self-coding interface unavailable for %s after %d retries; integration remains disabled.",
                    item.cls.__name__,
                    item.attempts,
                )
                _self_coding_import_warning_emitted = True

    if retry:
        with _PENDING_SELF_CODING_LOCK:
            _PENDING_SELF_CODING_DECORATIONS.extend(retry)
        next_delay = min(1.0, 0.1 * (retry[0].attempts + 1))
        _schedule_deferred_self_coding(delay=next_delay)


def _queue_self_coding_decorator(kwargs: dict[str, Any]) -> Callable[[type], type]:
    """Return a decorator that defers application until helpers are ready."""

    copied = dict(kwargs)

    def decorator(cls: type) -> type:
        with _PENDING_SELF_CODING_LOCK:
            _PENDING_SELF_CODING_DECORATIONS.append(
                _DeferredSelfCodingDecoration(cls=cls, kwargs=copied)
            )
        _schedule_deferred_self_coding()
        return cls

    return decorator


def _noop_self_coding_decorator(**_kwargs: Any) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        return cls

    return decorator


def self_coding_managed(**kwargs: Any) -> Callable[[type], type]:
    """Defer importing the coding bot interface to avoid circular imports."""

    global _self_coding_import_warning_emitted
    global _self_coding_disabled_warning_emitted

    interface_module_name = f"{_PACKAGE_NAME}.coding_bot_interface"

    if _data_bot_fallback:
        if not _self_coding_disabled_warning_emitted:
            logger.warning(
                "Self-coding integration disabled because DataBot fallback is active.",
            )
            _self_coding_disabled_warning_emitted = True
        return _noop_self_coding_decorator(**kwargs)

    if _is_module_initializing(interface_module_name):
        logger.debug(
            "Deferring TaskHandoffBot self-coding decoration until coding_bot_interface completes import (module still initializing)."
        )
        return _queue_self_coding_decorator(kwargs)

    try:
        interface_mod = load_internal("coding_bot_interface")
    except Exception as exc:  # pragma: no cover - optional dependency fallback
        if not _self_coding_import_warning_emitted:
            logger.warning(
                "Self-coding interface unavailable for TaskHandoffBot; disabling integration: %s",
                exc,
            )
            _self_coding_import_warning_emitted = True
        return _noop_self_coding_decorator(**kwargs)

    if _is_module_initializing(interface_module_name):
        logger.debug(
            "Deferring TaskHandoffBot self-coding decoration until coding_bot_interface completes import (module still initializing)."
        )
        return _queue_self_coding_decorator(kwargs)

    try:
        decorator_factory = interface_mod.self_coding_managed  # type: ignore[attr-defined]
    except AttributeError as exc:  # pragma: no cover - compatibility fallback
        if not _self_coding_import_warning_emitted:
            logger.warning(
                "Self-coding interface unavailable for TaskHandoffBot; disabling integration: %s",
                exc,
            )
            _self_coding_import_warning_emitted = True
        return _noop_self_coding_decorator(**kwargs)

    return decorator_factory(**kwargs)

UnifiedEventBus = load_internal("unified_event_bus").UnifiedEventBus
WorkflowGraph = load_internal("workflow_graph").WorkflowGraph

_import_guard_module = load_internal("shared.self_coding_import_guard")
try:
    self_coding_import_depth = _import_guard_module.self_coding_import_depth  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - compatibility fallback
    def self_coding_import_depth(_module_path: Any | None = None) -> int:
        return 0

try:
    _vector_service = load_internal("vector_service")
except Exception as exc:  # pragma: no cover - optional dependency fallback
    logger.warning(
        "Vector service unavailable for TaskHandoffBot; embeddings disabled: %s",
        exc,
    )

    class _FallbackEmbeddableDBMixin:
        """Minimal stub when the vector service cannot be imported."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._embeddings_enabled = False

        def backfill_embeddings(self) -> list[Any]:
            return []

        def search_by_vector(self, *args: Any, **kwargs: Any) -> list[Any]:
            return []

    class _FallbackEmbeddingBackfill:
        """No-op backfill watcher for disabled embeddings."""

        def watch_events(self, *, bus: Any | None = None) -> None:
            logger.debug("embedding backfill disabled; skipping watch (bus=%s)", bus)

    def _generalise_passthrough(text: str) -> str:
        return text

    EmbeddableDBMixin = _FallbackEmbeddableDBMixin
    EmbeddingBackfill = _FallbackEmbeddingBackfill
    generalise = _generalise_passthrough

    def safe_super_init(cls: type, instance: object, *args: Any, **kwargs: Any) -> None:
        try:
            mro = type(instance).__mro__
            next_cls = mro[mro.index(cls) + 1]
        except (ValueError, IndexError):
            next_cls = None

        if next_cls is object:
            if kwargs:
                logger.debug(
                    "[task-handoff] Dropping kwargs for object.__init__: cls=%s kwargs=%s",
                    cls.__name__,
                    list(kwargs),
                )
            if args:
                logger.debug(
                    "[task-handoff] Dropping args for object.__init__: cls=%s args=%s",
                    cls.__name__,
                    args,
                )
            super(cls, instance).__init__()
            return

        super(cls, instance).__init__(*args, **kwargs)

    def safe_super_init_or_warn(
        cls: type, instance: object, *args: Any, logger: logging.Logger | None = None, **kwargs: Any
    ) -> None:
        if logger is None:
            logger_ref = globals().get("logger")
        else:
            logger_ref = logger

        try:
            mro = type(instance).__mro__
            next_cls = mro[mro.index(cls) + 1]
        except (ValueError, IndexError):
            next_cls = None

        if next_cls is object and (args or kwargs):
            if logger_ref is not None:
                logger_ref.debug(
                    "[task-handoff] Dropping cooperative args for %s -> object.__init__: args=%s kwargs=%s",
                    cls.__name__,
                    args,
                    kwargs,
                )
        safe_super_init(cls, instance, *args, **kwargs)
else:
    EmbeddableDBMixin = _vector_service.EmbeddableDBMixin
    EmbeddingBackfill = _vector_service.EmbeddingBackfill
    try:
        generalise = load_internal("vector_service.text_preprocessor").generalise
    except Exception as exc:  # pragma: no cover - optional dependency fallback
        logger.warning(
            "Vector text preprocessor unavailable; using passthrough: %s",
            exc,
        )

        def generalise(text: str) -> str:
            return text

    safe_super_init = getattr(_vector_service, "safe_super_init", None)
    safe_super_init_or_warn = getattr(_vector_service, "safe_super_init_or_warn", None)

    if safe_super_init is None:  # pragma: no cover - compatibility guard

        def safe_super_init(
            cls: type, instance: object, *args: Any, **kwargs: Any
        ) -> None:
            try:
                mro = type(instance).__mro__
                next_cls = mro[mro.index(cls) + 1]
            except (ValueError, IndexError):
                next_cls = None

            if next_cls is object:
                if kwargs:
                    logger.debug(
                        "[task-handoff] Dropping kwargs for object.__init__: cls=%s kwargs=%s",
                        cls.__name__,
                        list(kwargs),
                    )
                if args:
                    logger.debug(
                        "[task-handoff] Dropping args for object.__init__: cls=%s args=%s",
                        cls.__name__,
                        args,
                    )
                super(cls, instance).__init__()
                return

            super(cls, instance).__init__(*args, **kwargs)

    if safe_super_init_or_warn is None:  # pragma: no cover - compatibility guard

        def safe_super_init_or_warn(
            cls: type,
            instance: object,
            *args: Any,
            logger: logging.Logger | None = None,
            **kwargs: Any,
        ) -> None:
            target_logger = logger or globals().get("logger")
            try:
                mro = type(instance).__mro__
                next_cls = mro[mro.index(cls) + 1]
            except (ValueError, IndexError):
                next_cls = None

            if next_cls is object and (args or kwargs) and target_logger is not None:
                target_logger.debug(
                    "[task-handoff] Dropping cooperative args for %s -> object.__init__: args=%s kwargs=%s",
                    cls.__name__,
                    args,
                    kwargs,
                )

            safe_super_init(cls, instance, *args, **kwargs)

_db_router = load_internal("db_router")
DBRouter = _db_router.DBRouter
GLOBAL_ROUTER = _db_router.GLOBAL_ROUTER
LOCAL_TABLES = _db_router.LOCAL_TABLES
SHARED_TABLES = _db_router.SHARED_TABLES
init_db_router = _db_router.init_db_router
queue_insert = _db_router.queue_insert

_db_dedup = load_internal("db_dedup")
insert_if_unique = _db_dedup.insert_if_unique
ensure_content_hash_column = _db_dedup.ensure_content_hash_column

_scope_utils = load_internal("scope_utils")
Scope = _scope_utils.Scope
build_scope_clause = _scope_utils.build_scope_clause

_dynamic_path_router = load_internal("dynamic_path_router")
get_project_root = _dynamic_path_router.get_project_root
resolve_path = _dynamic_path_router.resolve_path

try:
    _summarise_text = load_internal("menace_memory_manager")._summarise_text  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback
    from menace_memory_manager import _summarise_text  # type: ignore

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore
try:
    import zmq  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    zmq = None  # type: ignore
try:
    import pika  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pika = None  # type: ignore


registry = BotRegistry()


def _wrap_update_bot_once(reg: BotRegistry) -> None:
    """Install a re-entrancy guard around ``BotRegistry.update_bot``.

    ``self_coding_managed`` invokes :meth:`BotRegistry.update_bot` while the
    module defining the bot is still being executed.  ``update_bot`` in turn
    hot-swaps the module by re-importing it which re-executes the decorator and
    causes another immediate ``update_bot`` call.  In production environments
    this feedback loop never terminates because the original call is still in
    progress when the nested invocation begins.

    The guard tracks whether ``update_bot`` is already handling an update for
    this module path and turns nested calls into no-ops.  The outermost call
    proceeds normally and performs the actual hot-swap, ensuring behaviour is
    unchanged aside from breaking the loop.
    """

    attr = getattr(reg.update_bot, "__func__", reg.update_bot)
    if getattr(attr, "__task_handoff_guard__", False):
        return

    original = reg.update_bot
    guard_state = threading.local()

    def _resolve_module_path(candidate: str) -> Path | None:
        """Best-effort normalisation of ``candidate`` to a file on disk."""

        if not candidate:
            return None

        try:
            path = Path(candidate)
        except Exception:
            path = None
        else:
            try:
                if path.exists():
                    return path.resolve()
            except Exception:
                # ``Path.exists`` may raise on certain virtual paths (for
                # example when the working directory is unavailable).  Fall
                # through to import based resolution in that case.
                pass

        try:
            spec = importlib.util.find_spec(candidate)
        except (ImportError, AttributeError, ValueError):
            spec = None
        origin: str | None = getattr(spec, "origin", None) if spec else None
        if origin:
            try:
                return Path(origin).resolve()
            except Exception:
                return None
        return None

    current_module_path = Path(__file__).resolve()

    @wraps(attr)
    def _guarded_update_bot(self: BotRegistry, name: str, module_path: str, **kwargs: Any) -> bool:
        resolved = _resolve_module_path(module_path)
        guard_key: str | None
        if resolved is not None:
            guard_key = str(resolved)
            same_module = resolved == current_module_path
        else:
            guard_key = module_path or None
            same_module = module_path in {
                __name__,
                "task_handoff_bot",
                f"{_PACKAGE_NAME}.task_handoff_bot",
                str(current_module_path),
            }
        if same_module and guard_key:
            with _UPDATE_LOCK:
                if guard_key in _UPDATES_IN_PROGRESS or getattr(guard_state, "active", False):
                    logger.debug(
                        "Skipping re-entrant update_bot call for %s due to in-progress hot swap",
                        name,
                    )
                    return False
                _UPDATES_IN_PROGRESS.add(guard_key)
            guard_state.active = True
            try:
                return original(name, module_path, **kwargs)
            finally:
                guard_state.active = False
                with _UPDATE_LOCK:
                    _UPDATES_IN_PROGRESS.discard(guard_key)
        return original(name, module_path, **kwargs)

    _guarded_update_bot.__task_handoff_guard__ = True  # type: ignore[attr-defined]
    reg.update_bot = MethodType(_guarded_update_bot, reg)


# ``BotRegistry.update_bot`` hot swaps modules by re-importing them.  When this
# module is re-imported mid-call a new ``BotRegistry`` instance is created which
# would recursively trigger another hot swap.  Guard against this by tracking
# active updates globally so nested invocations are ignored regardless of the
# registry instance they originate from.
# ``importlib.reload`` re-executes this module which would normally re-create the
# guard primitives below while :meth:`BotRegistry.update_bot` is still running.
# Nested hot-swap calls rely on the ``_UPDATES_IN_PROGRESS`` set populated by the
# outer invocation, so replacing it during a reload would cause the guard to miss
# the in-flight update and recurse indefinitely.  Preserve the existing lock and
# tracking set when available so a reload continues to coordinate with the
# original call stack.
_prev_lock = globals().get("_UPDATE_LOCK")
if isinstance(_prev_lock, _thread.RLock):  # type: ignore[attr-defined]
    _UPDATE_LOCK = _prev_lock
else:
    _UPDATE_LOCK = threading.RLock()

_prev_updates = globals().get("_UPDATES_IN_PROGRESS")
if isinstance(_prev_updates, set):
    _UPDATES_IN_PROGRESS = _prev_updates
else:
    _UPDATES_IN_PROGRESS: set[str] = set()


_wrap_update_bot_once(registry)
data_bot = DataBot(start_server=False)


class _PassiveSelfCodingManager:
    """Minimal stub satisfying self-coding registration requirements."""

    __slots__ = ("bot_registry", "data_bot", "evolution_orchestrator")

    def __init__(self, bot_registry: BotRegistry, data_bot: DataBot) -> None:
        self.bot_registry = bot_registry
        self.data_bot = data_bot
        self.evolution_orchestrator = None

    def register(self, *_args: Any, **_kwargs: Any) -> None:
        """No-op compatibility shim for passive manager mode."""
        logger.debug("passive self-coding manager skipping register call")


manager = _PassiveSelfCodingManager(registry, data_bot)

_WATCH_THREAD: threading.Thread | None = None


def _ensure_backfill_watcher(bus: "UnifiedEventBus" | None) -> None:
    """Start ``EmbeddingBackfill.watch_events`` once for this module."""

    global _WATCH_THREAD
    if bus is None or _WATCH_THREAD is not None:
        return
    try:
        thread = threading.Thread(
            target=EmbeddingBackfill().watch_events,
            kwargs={"bus": bus},
            daemon=True,
        )
        thread.start()
        _WATCH_THREAD = thread
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed to start embedding watcher")


@dataclass
class TaskInfo:
    """Detailed information about a task."""

    name: str
    dependencies: List[str]
    resources: Dict[str, float]
    schedule: str
    code: str
    metadata: Dict[str, Any]


@dataclass
class TaskPackage:
    """Package of tasks ready for Stage 4."""

    tasks: List[TaskInfo]
    version: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class WorkflowRecord:
    """Stored workflow broken down from a plan."""

    workflow: List[str]
    action_chains: List[str] = field(default_factory=list)
    argument_strings: List[str] = field(default_factory=list)
    assigned_bots: List[str] = field(default_factory=list)
    enhancements: List[str] = field(default_factory=list)
    title: str = ""
    description: str = ""
    task_sequence: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    category: str = ""
    type_: str = ""
    status: str = "pending"
    rejection_reason: str = ""
    workflow_duration: float = 0.0
    performance_data: str = ""
    estimated_profit_per_bot: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    wid: int = 0


# Columns that contribute to the deduplication hash.
_WORKFLOW_HASH_FIELDS = sorted([
    "workflow",
    "action_chains",
    "argument_strings",
    "description",
])


class WorkflowDB(EmbeddableDBMixin):
    """SQLite storage for generated workflows with vector search."""

    DB_FILE = "workflows.db"

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        event_bus: Optional[UnifiedEventBus] = None,
        workflow_graph: Optional[WorkflowGraph] = None,
        vector_backend: str = "annoy",
        vector_index_path: Path | str = "workflow_embeddings.index",
        embedding_version: int = 1,
        router: DBRouter | None = None,
    ) -> None:
        logger.debug(
            "[trace] MRO chain: %s",
            [cls.__name__ for cls in type(self).__mro__],
        )

        if path is None:
            try:
                resolved = resolve_path(self.DB_FILE)
            except FileNotFoundError:
                resolved = get_project_root() / "sandbox_data" / self.DB_FILE
                resolved.parent.mkdir(parents=True, exist_ok=True)
        else:
            resolved = Path(path)
            if not resolved.is_absolute():
                resolved = get_project_root() / resolved
        self.path = resolved.resolve()
        self.event_bus = event_bus
        _ensure_backfill_watcher(self.event_bus)
        self.graph = workflow_graph
        self.vector_backend = vector_backend  # kept for compatibility
        LOCAL_TABLES.add("workflows")
        self.router = router or GLOBAL_ROUTER or init_db_router(
            "task_handoff", local_db_path=str(self.path), shared_db_path=str(self.path)
        )
        self.conn = self.router.get_connection("workflows")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workflows(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow TEXT,
                action_chains TEXT,
                argument_strings TEXT,
                assigned_bots TEXT,
                enhancements TEXT,
                title TEXT,
                description TEXT,
                task_sequence TEXT,
                tags TEXT,
                dependencies TEXT,
                reasons TEXT,
                category TEXT,
                type TEXT,
                status TEXT,
                rejection_reason TEXT,
                workflow_duration REAL,
                performance_data TEXT,
                estimated_profit_per_bot REAL,
                timestamp TEXT,
                source_menace_id TEXT,
                content_hash TEXT NOT NULL
            )
            """,
        )
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(workflows)").fetchall()]
        if "action_chains" not in cols:
            self.conn.execute("ALTER TABLE workflows ADD COLUMN action_chains TEXT")
        if "argument_strings" not in cols:
            self.conn.execute("ALTER TABLE workflows ADD COLUMN argument_strings TEXT")
        if "dependencies" not in cols:
            self.conn.execute("ALTER TABLE workflows ADD COLUMN dependencies TEXT")
        if "reasons" not in cols:
            self.conn.execute("ALTER TABLE workflows ADD COLUMN reasons TEXT")
        if "source_menace_id" not in cols:
            self.conn.execute(
                "ALTER TABLE workflows ADD COLUMN "
                "source_menace_id TEXT NOT NULL DEFAULT ''"
            )
        ensure_content_hash_column("workflows", conn=self.conn)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS "
            "idx_workflows_source_menace_id ON workflows(source_menace_id)"
        )
        self.conn.commit()
        safe_super_init_or_warn(
            WorkflowDB,
            self,
            index_path=vector_index_path,
            embedding_version=embedding_version,
            backend=vector_backend,
            event_bus=event_bus,
            logger=logger,
        )

    # --------------------------------------------------------------
    # helpers
    def _row_to_record(self, row: sqlite3.Row) -> WorkflowRecord:
        return WorkflowRecord(
            workflow=row["workflow"].split(",") if row["workflow"] else [],
            action_chains=row["action_chains"].split(",") if row["action_chains"] else [],
            argument_strings=row["argument_strings"].split(",") if row["argument_strings"] else [],
            assigned_bots=row["assigned_bots"].split(",") if row["assigned_bots"] else [],
            enhancements=row["enhancements"].split(",") if row["enhancements"] else [],
            title=row["title"] or "",
            description=row["description"] or "",
            task_sequence=row["task_sequence"].split(",") if row["task_sequence"] else [],
            tags=row["tags"].split(",") if row["tags"] else [],
            dependencies=row["dependencies"].split(",") if row["dependencies"] else [],
            reasons=row["reasons"].split(",") if row["reasons"] else [],
            category=row["category"] or "",
            type_=row["type"] or "",
            status=row["status"] or "",
            rejection_reason=row["rejection_reason"] or "",
            workflow_duration=row["workflow_duration"] or 0.0,
            performance_data=row["performance_data"] or "",
            estimated_profit_per_bot=row["estimated_profit_per_bot"] or 0.0,
            timestamp=row["timestamp"],
            wid=row["id"],
        )

    def _vector_text(self, rec: WorkflowRecord) -> str:
        """Build a descriptive, summarised text representation of ``rec``."""

        actions = rec.workflow or rec.task_sequence
        sequence = rec.task_sequence if rec.workflow != rec.task_sequence else []
        boilerplate = {"start", "end", "noop", "return", "pass"}
        lines: list[str] = []
        for lst in (actions, sequence, rec.argument_strings or []):
            for item in lst:
                if not item or item.strip().lower() in boilerplate:
                    continue
                lines.extend(
                    s.strip()
                    for s in re.split(r"(?<=[.!?])\s+", item)
                    if s.strip()
                )
        summary = _summarise_text(" ".join(lines)) if lines else ""
        parts: list[str] = []
        if summary:
            parts.append(summary)
        if rec.title:
            parts.append(f"title: {rec.title}")
        if rec.description:
            parts.append(f"description: {rec.description}")
        if rec.tags:
            parts.append("tags: " + ", ".join(rec.tags))
        if rec.category:
            parts.append(f"category: {rec.category}")
        if rec.type_:
            parts.append(f"type: {rec.type_}")
        return "\n".join(generalise(p) for p in parts if p)

    def usage_rate(self, workflow_id: int) -> int:
        """Return count of bots using a workflow.

        Prefers the ``assigned_bots`` list stored on the workflow record. If
        absent, falls back to counting relationships in ``bot_workflow`` when
        that table is present in the connected database.
        """
        row = self.conn.execute(
            "SELECT assigned_bots FROM workflows WHERE id=?",
            (workflow_id,),
        ).fetchone()
        if row and row["assigned_bots"]:
            bots = [b for b in row["assigned_bots"].split(",") if b]
            return len(bots)
        try:
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM bot_workflow WHERE workflow_id=?",
                (workflow_id,),
            )
            count = cur.fetchone()
            return int(count[0]) if count else 0
        except sqlite3.Error:
            return 0

    # --------------------------------------------------------------
    # status updates
    def update_status(self, workflow_id: int, status: str) -> None:
        """Update the status for a single workflow and refresh embedding."""
        self.conn.execute(
            "UPDATE workflows SET status=? WHERE id=?",
            (status, workflow_id),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT * FROM workflows WHERE id=?",
            (workflow_id,),
        ).fetchone()
        if row:
            rec = self._row_to_record(row)
            self.try_add_embedding(workflow_id, rec, "workflow", source_id=str(workflow_id))
        if self.event_bus:
            try:
                payload = {"workflow_id": workflow_id, "status": status}
                self.event_bus.publish("workflows:update", payload)
                self.event_bus.publish(
                    "embedding:backfill", {"db": self.__class__.__name__}
                )
            except Exception as exc:
                logger.warning(
                    "failed to publish workflow status %s for %s: %s",
                    status,
                    workflow_id,
                    exc,
                )
        if self.graph:
            try:
                self.graph.update(str(workflow_id), "update")
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("failed to update workflow graph for %s: %s", workflow_id, exc)

    def update_statuses(self, workflow_ids: Iterable[int], status: str) -> None:
        """Bulk update workflow status and refresh embeddings."""
        ids = list(workflow_ids)
        for wid in ids:
            self.conn.execute(
                "UPDATE workflows SET status=? WHERE id=?",
                (status, wid),
            )
        self.conn.commit()
        for wid in ids:
            row = self.conn.execute(
                "SELECT * FROM workflows WHERE id=?",
                (wid,),
            ).fetchone()
            if row:
                rec = self._row_to_record(row)
                self.try_add_embedding(wid, rec, "workflow", source_id=str(wid))
                if self.graph:
                    try:
                        self.graph.update(str(wid), "update")
                    except Exception as exc:  # pragma: no cover - best effort
                        logger.warning("failed to update workflow graph for %s: %s", wid, exc)
        if self.event_bus:
            try:
                for wid in ids:
                    payload = {"workflow_id": wid, "status": status}
                    self.event_bus.publish("workflows:update", payload)
                self.event_bus.publish(
                    "embedding:backfill", {"db": self.__class__.__name__}
                )
            except Exception as exc:
                logger.warning(
                    "failed to publish workflow status %s for %s: %s",
                    status,
                    ids,
                    exc,
                )

    # --------------------------------------------------------------
    # insert/fetch
    def add(self, wf: WorkflowRecord, source_menace_id: str = "") -> int | None:
        menace_id = source_menace_id or self.router.menace_id
        values = {
            "source_menace_id": menace_id,
            "workflow": ",".join(wf.workflow),
            "action_chains": ",".join(wf.action_chains),
            "argument_strings": ",".join(wf.argument_strings),
            "assigned_bots": ",".join(wf.assigned_bots),
            "enhancements": ",".join(wf.enhancements),
            "title": wf.title,
            "description": wf.description,
            "task_sequence": ",".join(wf.task_sequence),
            "tags": ",".join(wf.tags),
            "dependencies": ",".join(wf.dependencies),
            "reasons": ",".join(wf.reasons),
            "category": wf.category,
            "type": wf.type_,
            "status": wf.status,
            "rejection_reason": wf.rejection_reason,
            "workflow_duration": wf.workflow_duration,
            "performance_data": wf.performance_data,
            "estimated_profit_per_bot": wf.estimated_profit_per_bot,
            "timestamp": wf.timestamp,
        }
        if "workflows" in SHARED_TABLES:
            payload = dict(values)
            payload["hash_fields"] = _WORKFLOW_HASH_FIELDS
            queue_insert("workflows", payload, menace_id)
            wf.wid = 0
        else:
            with self.router.get_connection("workflows", "write") as conn:
                wf.wid = insert_if_unique(
                    "workflows",
                    values,
                    _WORKFLOW_HASH_FIELDS,
                    menace_id,
                    conn=conn,
                    logger=logger,
                )

        if wf.wid:
            self.try_add_embedding(wf.wid, wf, "workflow", source_id=str(wf.wid))

        if self.event_bus:
            try:
                self.event_bus.publish("workflows:new", asdict(wf))
                self.event_bus.publish("db.record_changed", {"db": "workflow"})
                self.event_bus.publish(
                    "embedding:backfill", {"db": self.__class__.__name__}
                )
            except Exception as exc:
                logger.warning(
                    "failed to publish new workflow %s: %s",
                    wf.wid,
                    exc,
                )
        if self.graph:
            try:
                self.graph.update(str(wf.wid), "add")
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("failed to update workflow graph for %s: %s", wf.wid, exc)
        return wf.wid

    def remove(self, workflow_id: int) -> None:
        """Delete a workflow and emit an event."""
        self.conn.execute("DELETE FROM workflows WHERE id=?", (workflow_id,))
        self.conn.commit()
        if self.event_bus:
            try:
                self.event_bus.publish(
                    "workflows:deleted", {"workflow_id": workflow_id}
                )
            except Exception as exc:
                logger.warning(
                    "failed to publish workflow deletion %s: %s", workflow_id, exc
                )
        if self.graph:
            try:
                self.graph.update(str(workflow_id), "remove")
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning(
                    "failed to update workflow graph for deletion %s: %s", workflow_id, exc
                )

    def replace(self, workflow_id: int, wf: WorkflowRecord) -> int:
        """Replace an existing workflow with ``wf``.

        The original workflow is removed and a new one inserted; the identifier
        of the new workflow is returned.
        """

        self.remove(workflow_id)
        return self.add(wf)

    def fetch(self, limit: int = 20) -> List[WorkflowRecord]:
        self.conn.row_factory = sqlite3.Row
        rows = self.conn.execute(
            "SELECT * FROM workflows ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results: List[WorkflowRecord] = []
        for row in rows:
            results.append(self._row_to_record(row))
        return results

    def fetch_workflows(
        self,
        *,
        scope: Literal["local", "global", "all"] = "local",
        sort_by: Literal["confidence", "outcome_score", "timestamp"] = "timestamp",
        limit: int = 100,
        include_embeddings: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return stored workflows filtered by menace scope and ordered.

        Parameters mirror those used by other fetch helpers. ``scope`` controls
        which menace IDs to include via :func:`build_scope_clause`. ``sort_by``
        selects the column used for ordering with the following mappings:

        - ``"confidence"``    -> ``estimated_profit_per_bot``
        - ``"outcome_score"`` -> ``workflow_duration``
        - ``"timestamp"``     -> ``timestamp``

        When ``include_embeddings`` is true, each record includes an ``embedding``
        field containing the vector representation produced by :meth:`vector`.
        ``confidence`` and ``outcome_score`` are populated from existing
        columns as described above.
        """

        menace_id = getattr(self.router, "menace_id", "")
        clause, params = build_scope_clause("workflows", Scope(scope), menace_id)
        order_map = {
            "confidence": "estimated_profit_per_bot",
            "outcome_score": "workflow_duration",
            "timestamp": "timestamp",
        }
        order_col = order_map.get(sort_by, "timestamp")
        sql = "SELECT * FROM workflows"
        if clause:
            sql += f" WHERE {clause}"
        sql += f" ORDER BY {order_col} DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        results: List[Dict[str, Any]] = []
        for row in rows:
            rec = self._row_to_record(row)
            data = asdict(rec)
            data["confidence"] = row["estimated_profit_per_bot"]
            data["outcome_score"] = row["workflow_duration"]
            if include_embeddings:
                try:
                    data["embedding"] = self.vector(rec)
                except Exception:  # pragma: no cover - best effort
                    data["embedding"] = None
            results.append(data)
        return results

    def backfill_embeddings(self, batch_size: int = 100) -> None:
        """Delegate to :class:`EmbeddableDBMixin` for compatibility."""
        EmbeddableDBMixin.backfill_embeddings(self)

    def iter_records(self) -> Iterator[tuple[int, WorkflowRecord, str]]:
        """Yield all workflow records for embedding backfill."""
        cur = self.conn.execute("SELECT * FROM workflows")
        for row in cur.fetchall():
            rec = self._row_to_record(row)
            yield rec.wid, rec, "workflow"

    # --------------------------------------------------------------
    # embedding/search
    def vector(self, rec: Any) -> list[float]:
        """Embed ``rec`` into a vector using its action chain and arguments."""

        if isinstance(rec, (int, str)):
            row = self.conn.execute(
                "SELECT * FROM workflows WHERE id=?",
                (int(rec),),
            ).fetchone()
            if not row:
                raise ValueError("record not found")
            rec = self._row_to_record(row)
        elif isinstance(rec, sqlite3.Row):
            rec = self._row_to_record(rec)
        elif not isinstance(rec, WorkflowRecord):
            raise TypeError("unsupported record type")
        text = self._vector_text(rec)
        prepared = self._prepare_text_for_embedding(text, db_key="workflow")
        return self._embed(prepared)

    def _embed(self, text: str) -> list[float]:
        """Encode ``text`` to a vector (overridable for tests)."""
        return self.encode_text(text)

    def search_by_vector(self, vector: Iterable[float], top_k: int = 5) -> List[WorkflowRecord]:
        matches = EmbeddableDBMixin.search_by_vector(self, vector, top_k)
        results: List[WorkflowRecord] = []
        for rec_id, dist in matches:
            row = self.conn.execute(
                "SELECT * FROM workflows WHERE id=?",
                (rec_id,),
            ).fetchone()
            if row:
                rec = self._row_to_record(row)
                setattr(rec, "_distance", dist)
                results.append(rec)
        return results


@self_coding_managed(bot_registry=registry, data_bot=data_bot, manager=manager)
class TaskHandoffBot:
    """Compile tasks, record workflows and transmit them to Stage 4."""

    def __init__(
        self,
        api_url: str = "http://localhost:9000/handoff",
        mq_url: str | None = None,
        pair_addr: str | None = None,
        workflow_db: Optional[WorkflowDB] = None,
        *,
        event_bus: Optional[UnifiedEventBus] = None,
        manager: "SelfCodingManager | None" = None,
    ) -> None:
        self.api_url = api_url
        if zmq:
            self.context = zmq.Context.instance()
            self.socket = self.context.socket(zmq.PAIR)
            self.addr = pair_addr or f"inproc://handoff-{uuid.uuid4().hex}"
            self.socket.bind(self.addr)
            self._noblock = zmq.NOBLOCK
            self._again = zmq.Again
            self._error = zmq.ZMQError
        else:  # pragma: no cover - zmq unavailable
            from multiprocessing.connection import Listener, Client, Connection
            import os

            class _IpcSocket:
                """Simple IPC socket using :mod:`multiprocessing.connection`."""

                def __init__(self) -> None:
                    self._listener: Listener | None = None
                    self._conn: Connection | None = None
                    self._address: str | None = None

                def bind(self, address: str) -> None:  # noqa: D401 - bind address
                    self._address = address or f"/tmp/handoff-{uuid.uuid4().hex}.sock"
                    if os.path.exists(self._address):
                        try:
                            os.unlink(self._address)
                        except OSError:
                            pass
                    self._listener = Listener(self._address, family="AF_UNIX")

                def connect(self, address: str) -> None:
                    self._address = address
                    self._conn = Client(address, family="AF_UNIX")

                def _ensure_conn(self) -> None:
                    if self._conn is None and self._listener is not None:
                        try:
                            self._conn = self._listener.accept()
                        except Exception as exc:  # pragma: no cover - accept error
                            raise _Again("no connection") from exc

                def send_json(self, msg: dict, *a, flags: int | None = None, **k) -> None:
                    self._ensure_conn()
                    if self._conn is None:
                        raise _Again("not connected")
                    try:
                        self._conn.send(msg)
                    except Exception as exc:
                        if flags is not None:
                            raise _Again() from exc
                        raise

                def recv_json(self, *a, flags: int | None = None, **k) -> dict:
                    self._ensure_conn()
                    if self._conn is None:
                        raise _Again("not connected")
                    if flags is not None and not self._conn.poll():
                        raise _Again()
                    try:
                        return self._conn.recv()
                    except Exception as exc:
                        if flags is not None:
                            raise _Again() from exc
                        raise

                def close(self) -> None:
                    if self._conn is not None:
                        try:
                            self._conn.close()
                        except Exception:
                            pass
                        self._conn = None
                    if self._listener is not None:
                        try:
                            self._listener.close()
                        except Exception:
                            pass
                        if isinstance(self._listener.address, str):
                            try:
                                os.unlink(self._listener.address)
                            except Exception:
                                pass
                        self._listener = None

            class _Again(Exception):
                """Raised when a non-blocking IPC operation would block."""
                pass

            self.context = None
            self.socket = _IpcSocket()
            self.addr = pair_addr or f"/tmp/handoff-{uuid.uuid4().hex}.sock"
            self.socket.bind(self.addr)
            self._noblock = 0
            self._again = _Again
            self._error = Exception
        self.channel = None
        self.workflow_db = workflow_db or WorkflowDB(event_bus=event_bus)
        if mq_url and pika and _can_reach_broker(mq_url):
            try:
                params = pika.URLParameters(mq_url)
                conn = pika.BlockingConnection(params)
                ch = conn.channel()
                ch.queue_declare(queue="handoff", durable=True)
                self.channel = ch
            except Exception:
                logger.info(
                    "RabbitMQ unreachable at %s; using local handoff queue", mq_url
                )
                self.channel = None
        elif mq_url:
            logger.info(
                "RabbitMQ not available at %s; using in-process handoff queue",
                mq_url,
            )

    def compile(self, tasks: List[TaskInfo]) -> TaskPackage:
        return TaskPackage(tasks=tasks)

    def _split_tasks(self, names: List[str], min_size: int = 3) -> List[List[str]]:
        chunks = [names]
        size = len(names)
        while size > min_size:
            size = max(min_size, size // 2)
            for i in range(0, len(names), size):
                chunk = names[i:i + size]
                if chunk and chunk not in chunks:
                    chunks.append(chunk)
        return chunks

    def store_plan(
        self,
        tasks: Iterable[TaskInfo],
        enhancements: Iterable[str] | None = None,
        title: str = "",
        description: str = "",
    ) -> List[int]:
        names = [t.name for t in tasks]
        ids: List[int] = []
        for chunk in self._split_tasks(names):
            rec = WorkflowRecord(
                workflow=chunk,
                title=title,
                description=description,
                task_sequence=chunk,
                enhancements=list(enhancements or []),
            )
            wid = self.workflow_db.add(rec)
            if wid is not None:
                ids.append(wid)
        return ids

    # --------------------------------------------------------------
    # Status helpers
    # --------------------------------------------------------------

    def mark_workflows(self, workflow_ids: Iterable[int], status: str) -> None:
        """Update status for a collection of workflows."""
        ids = list(workflow_ids)
        try:
            self.workflow_db.update_statuses(ids, status)
            logger.info("marked workflows %s as %s", ids, status)
        except Exception as exc:
            logger.warning(
                "failed to update status %s for %s: %s", status, ids, exc
            )

    def mark_active(self, workflow_ids: Iterable[int]) -> None:
        self.mark_workflows(workflow_ids, "active")

    def mark_failed(self, workflow_ids: Iterable[int]) -> None:
        self.mark_workflows(workflow_ids, "failed")

    def send_package(self, package: TaskPackage) -> None:
        data = package.to_json()
        try:
            requests.post(
                self.api_url,
                data=data,
                headers={"Content-Type": "application/json"},
                timeout=3,
            )
        except Exception as exc:
            logger.warning("primary handoff failed: %s", exc)
            if self.channel:
                try:
                    self.channel.basic_publish("", "handoff", data)
                except Exception as mq_exc:
                    logger.warning("mq publish failed: %s", mq_exc)

    def respond_to_queries(self, handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        try:
            msg = self.socket.recv_json(flags=self._noblock)
        except self._again:
            return
        resp = handler(msg)
        try:
            self.socket.send_json(resp, flags=self._noblock)
        except self._error as exc:
            logger.warning("failed to send response: %s", exc)

    def close(self) -> None:
        try:
            self.socket.close()
        except Exception as exc:
            logger.warning("failed to close handoff socket: %s", exc)


_POST_IMPORT_UPDATE_SCHEDULED = False
_POST_IMPORT_UPDATE_COMPLETED = False


def _normalise_module_reference(value: Any | None) -> str | None:
    """Return a comparable representation for *value* or ``None``."""

    if value is None:
        return None
    try:
        return str(Path(value).resolve())
    except Exception:
        try:
            return os.fspath(value)  # type: ignore[arg-type]
        except Exception:
            return str(value)


def _needs_post_import_update(reg: BotRegistry, name: str, module_path: str) -> bool:
    """Return ``True`` when the registry metadata is out of date."""

    target = _normalise_module_reference(module_path)
    if not target:
        return False
    try:
        node = reg.graph.nodes.get(name)
    except Exception:
        node = None
    if not node:
        return True
    recorded = node.get("module") or reg.modules.get(name)
    if not recorded:
        return True
    current = _normalise_module_reference(recorded)
    return current != target


def _finalise_self_coding_registration() -> None:
    """Retry ``BotRegistry.update_bot`` once imports have settled."""

    global _POST_IMPORT_UPDATE_SCHEDULED, _POST_IMPORT_UPDATE_COMPLETED
    try:
        reg = getattr(TaskHandoffBot, "bot_registry")
    except AttributeError:
        reg = None
    if reg is None:
        _POST_IMPORT_UPDATE_SCHEDULED = False
        return
    name = getattr(TaskHandoffBot, "name", getattr(TaskHandoffBot, "bot_name", "TaskHandoffBot"))
    module_path = str(Path(__file__).resolve())
    if not _needs_post_import_update(reg, name, module_path):
        _POST_IMPORT_UPDATE_COMPLETED = True
        return
    try:
        reg.update_bot(name, module_path)
    except Exception:
        _POST_IMPORT_UPDATE_SCHEDULED = False
        logger.exception("post-import bot update failed for %s", name)
    else:
        _POST_IMPORT_UPDATE_COMPLETED = True


def ensure_task_handoff_registration(force: bool = False) -> None:
    """Synchronise registry metadata for :class:`TaskHandoffBot`."""

    global _POST_IMPORT_UPDATE_SCHEDULED, _POST_IMPORT_UPDATE_COMPLETED
    if force:
        _POST_IMPORT_UPDATE_SCHEDULED = False
        _POST_IMPORT_UPDATE_COMPLETED = False
    if _POST_IMPORT_UPDATE_COMPLETED:
        return
    _finalise_self_coding_registration()


def _schedule_post_import_update(delay: float = 0.1) -> None:
    """Schedule a single registry update retry after import-time recursion."""

    global _POST_IMPORT_UPDATE_SCHEDULED
    if _POST_IMPORT_UPDATE_SCHEDULED or _POST_IMPORT_UPDATE_COMPLETED:
        return
    _POST_IMPORT_UPDATE_SCHEDULED = True
    try:
        timer = threading.Timer(delay, _finalise_self_coding_registration)
        timer.daemon = True
        timer.start()
    except Exception:
        _POST_IMPORT_UPDATE_SCHEDULED = False
        logger.exception("failed to defer TaskHandoffBot registry update")


if self_coding_import_depth(Path(__file__)) > 1:
    _schedule_post_import_update()


__all__ = [
    "TaskInfo",
    "TaskPackage",
    "WorkflowRecord",
    "WorkflowDB",
    "TaskHandoffBot",
    "ensure_task_handoff_registration",
]
if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .self_coding_manager import SelfCodingManager
else:  # pragma: no cover - runtime fallback when manager is unused
    SelfCodingManager = object  # type: ignore[assignment]
