from __future__ import annotations

"""Stub provider using a language model selected via configuration.

The module persists generated stubs on disk and coordinates concurrent access
to the cache using a :class:`filelock.FileLock`.  Set ``SANDBOX_STUB_MODEL`` to
the desired model name understood by the active backend.  When no model is
configured deterministic rule based stubs are produced.
"""

from typing import Any, Dict, List, Tuple, Callable, Awaitable
import asyncio
import inspect
import json
import os
import re
import warnings
import uuid
from pathlib import Path
from dynamic_path_router import resolve_path, path_for_prompt
from collections import Counter, OrderedDict, defaultdict
import atexit
import importlib
from importlib import metadata
import random
import threading
from contextlib import AbstractAsyncContextManager
from typing import get_origin, get_args, Union
from dataclasses import dataclass, field
from filelock import FileLock, Timeout

from logging_utils import get_logger, set_correlation_id, log_record
from vector_service.context_builder import build_prompt as cb_build_prompt
try:  # pragma: no cover - allow flat import
    from metrics_exporter import (
        stub_generation_requests_total,
        stub_generation_failures_total,
        stub_generation_retries_total,
    )
except Exception:  # pragma: no cover - fallback when packaged
    from .metrics_exporter import (  # type: ignore
        stub_generation_requests_total,
        stub_generation_failures_total,
        stub_generation_retries_total,
    )

from .input_history_db import InputHistoryDB
from sandbox_settings import SandboxSettings
from model_registry import get_client
from llm_interface import Prompt

# Optional dependencies loaded lazily
pipeline = None  # type: ignore
openai = None  # type: ignore

logger = get_logger(__name__)


class StubCacheWarning(UserWarning):
    """Warning emitted when the on-disk stub cache cannot be used."""


_GENERATOR = None
# use OrderedDict for LRU eviction semantics
_CACHE: "OrderedDict[Tuple[str, str], Dict[str, Any]]" = OrderedDict()

# protect cache mutations across threads and async tasks
_CACHE_LOCK = threading.Lock()

# track stub usage statistics per target to avoid repetition
_TARGET_STATS: dict[str, Counter[str]] = defaultdict(Counter)

# Entry-point group for discovering available text generation models
MODEL_ENTRY_POINT_GROUP = "sandbox.stub_models"


@dataclass
class StubProviderConfig:
    """Configuration for stub generation and caching."""

    timeout: float
    retries: int
    retry_base: float
    retry_max: float
    cache_max: int
    cache_path: Path
    fallback_model: str
    save_timeout: float = 5.0
    max_concurrency: int = 1
    enabled_backends: Tuple[str, ...] = ()
    rate_limit: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        self.rate_limit = asyncio.Semaphore(self.max_concurrency)


FALLBACK_MODEL: str  # backwards compatibility alias
_SETTINGS: SandboxSettings | None = None
_CONFIG: StubProviderConfig | None = None


class _SaveTaskManager(AbstractAsyncContextManager):
    """Track background save tasks and await them on shutdown."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[None]] = set()
        self._lock = threading.Lock()

    def add(self, task: asyncio.Task[None]) -> None:
        with self._lock:
            self._tasks.add(task)
        task.add_done_callback(lambda t: self._tasks.discard(t))

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        with self._lock:
            tasks = list(self._tasks)
            self._tasks.clear()
        if tasks:
            await asyncio.gather(
                *(asyncio.shield(t) for t in tasks), return_exceptions=True
            )


_SAVE_TASKS = _SaveTaskManager()


def _feature_enabled(name: str) -> bool:
    """Return True if feature flag *name* is truthy."""
    val = os.getenv(name, "").lower()
    return val in {"1", "true", "yes"}


def _available_models(settings: Any | None = None) -> set[str]:
    """Return names of available text generation models."""

    models: set[str] = set()
    if settings is not None:
        models.update(getattr(settings, "stub_models", []) or [])

    try:
        eps = metadata.entry_points(group=MODEL_ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - legacy API
        eps = metadata.entry_points().get(MODEL_ENTRY_POINT_GROUP, [])
    except Exception as exc:  # pragma: no cover - best effort
        logger.exception("failed to gather stub model entry points", exc_info=exc)
        eps = []
    for ep in eps:
        models.add(ep.name)
    return models


def _load_config(settings: SandboxSettings) -> StubProviderConfig:
    """Validate environment configuration and produce :class:`StubProviderConfig`."""

    def _float_env(name: str, default: float) -> float:
        val = os.getenv(name)
        if val is None:
            return default
        try:
            f = float(val)
            if f <= 0:
                raise ValueError
            return f
        except ValueError:
            logger.warning("invalid %s=%r; using default %s", name, val, default)
            return default

    def _int_env(name: str, default: int) -> int:
        val = os.getenv(name)
        if val is None:
            return default
        try:
            i = int(val)
            if i < 1:
                raise ValueError
            return i
        except ValueError:
            logger.warning("invalid %s=%r; using default %s", name, val, default)
            return default

    cache_path = Path(
        os.getenv(
            "SANDBOX_STUB_CACHE",
            str(resolve_path("sandbox_data/stub_cache.json")),
        )
    )
    cfg = StubProviderConfig(
        timeout=_float_env("SANDBOX_STUB_TIMEOUT", settings.stub_timeout),
        retries=_int_env("SANDBOX_STUB_RETRIES", settings.stub_retries),
        retry_base=_float_env("SANDBOX_STUB_RETRY_BASE", settings.stub_retry_base),
        retry_max=_float_env("SANDBOX_STUB_RETRY_MAX", settings.stub_retry_max),
        cache_max=_int_env("SANDBOX_STUB_CACHE_MAX", settings.stub_cache_max),
        cache_path=cache_path,
        fallback_model=os.getenv(
            "SANDBOX_STUB_FALLBACK_MODEL", settings.stub_fallback_model
        ),
        save_timeout=_float_env(
            "SANDBOX_STUB_SAVE_TIMEOUT", settings.stub_save_timeout
        ),
        max_concurrency=_int_env("SANDBOX_STUB_MAX_CONCURRENCY", 1),
    )
    model = settings.sandbox_stub_model
    if model:
        available = _available_models(settings)
        if available and model not in available:
            msg = (
                f"unknown SANDBOX_STUB_MODEL {model!r}; available: {sorted(available)}"
            )
            logger.error(msg)
            raise ValueError(msg)
        if not available:
            logger.warning(
                "SANDBOX_STUB_MODEL=%s but no stub models are configured", model
            )

    cfg.enabled_backends = ()

    global FALLBACK_MODEL
    FALLBACK_MODEL = cfg.fallback_model
    return cfg


def get_settings(refresh: bool = False) -> SandboxSettings:
    """Return cached :class:`SandboxSettings`, refreshing if requested."""
    global _SETTINGS, _CONFIG
    if _SETTINGS is None or refresh:
        _SETTINGS = SandboxSettings()
        _CONFIG = _load_config(_SETTINGS)
    return _SETTINGS


def get_config(refresh: bool = False) -> StubProviderConfig:
    """Return cached :class:`StubProviderConfig`, refreshing if requested."""
    global _CONFIG
    settings = get_settings(refresh=refresh)
    if _CONFIG is None or refresh:
        _CONFIG = _load_config(settings)
    return _CONFIG


# Initialise settings/config on import for backward compatibility
SETTINGS = get_settings()
CONFIG = get_config()


async def _call_with_retry(
    func: Callable[[], Awaitable[Any]],
    config: StubProviderConfig | None = None,
) -> Any:
    """Invoke *func* with retry, timeout and rate limiting."""
    config = config or get_config()
    delay = config.retry_base
    for attempt in range(config.retries):
        try:
            async with config.rate_limit:
                return await asyncio.wait_for(func(), timeout=config.timeout)
        except Exception as exc:
            if attempt == config.retries - 1:
                raise
            stub_generation_retries_total.inc()
            logger.warning("generation attempt %d failed: %s", attempt + 1, exc)
            jitter = random.uniform(0, delay)
            await asyncio.sleep(jitter)
            delay = min(delay * 2, config.retry_max)


def _type_matches(value: Any, annotation: Any) -> bool:
    """Return True if *value* conforms to *annotation* (best effort)."""
    if annotation in (inspect._empty, Any):
        return True
    origin = get_origin(annotation)
    if origin is None:
        if annotation in (int, "int"):
            return isinstance(value, int) and not isinstance(value, bool)
        if annotation in (float, "float"):
            return isinstance(value, (float, int)) and not isinstance(value, bool)
        if annotation in (bool, "bool"):
            return isinstance(value, bool)
        if annotation in (str, "str"):
            return isinstance(value, str)
        return True
    if origin is list:
        (arg,) = get_args(annotation) or (Any,)
        return isinstance(value, list) and all(_type_matches(v, arg) for v in value)
    if origin is dict:
        key_type, val_type = get_args(annotation) or (Any, Any)
        return isinstance(value, dict) and all(
            _type_matches(k, key_type) and _type_matches(v, val_type)
            for k, v in value.items()
        )
    if origin is tuple:
        args = get_args(annotation)
        if ... in args:
            elem_type = args[0]
            return isinstance(value, tuple) and all(
                _type_matches(v, elem_type) for v in value
            )
        return (
            isinstance(value, tuple)
            and len(value) == len(args)
            and all(_type_matches(v, t) for v, t in zip(value, args))
        )
    if origin is set:
        (arg,) = get_args(annotation) or (Any,)
        return isinstance(value, set) and all(_type_matches(v, arg) for v in value)
    if origin is Union:
        return any(_type_matches(value, arg) for arg in get_args(annotation))
    return True


def _validate_stub_signature(stub: Dict[str, Any], func: Any) -> bool:
    """Return True if ``stub`` matches the signature of ``func``."""
    if func is None:
        return True
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError) as exc:
        logger.debug("signature inspection failed for %s: %s", func, exc)
        return True
    params = [
        (n, p)
        for n, p in sig.parameters.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    for name, param in params:
        if name not in stub:
            return False
        if not _type_matches(stub[name], param.annotation):
            return False
    return True


def _load_cache(
    config: StubProviderConfig | None = None,
) -> "OrderedDict[Tuple[str, str], Dict[str, Any]]":
    config = config or get_config()
    path = config.cache_path
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                raise ValueError("invalid cache format")
            cache: "OrderedDict[Tuple[str, str], Dict[str, Any]]" = OrderedDict()
            corrupted = False
            for item in data:
                if not (
                    isinstance(item, list)
                    and len(item) == 2
                    and isinstance(item[0], str)
                    and isinstance(item[1], dict)
                ):
                    corrupted = True
                    continue
                parts = item[0].split("::", 1)
                if len(parts) != 2:
                    corrupted = True
                    continue
                key = (parts[0], parts[1])
                value = item[1]
                if not _valid_cache_item(key, value):
                    corrupted = True
                    continue
                cache[key] = value
            if corrupted:
                warnings.warn(
                    "stub cache corrupted; rebuilding from valid entries",
                    StubCacheWarning,
                )
                try:
                    items = [[f"{k[0]}::{k[1]}", v] for k, v in cache.items()]
                    tmp = path.with_suffix(".tmp")
                    with open(tmp, "w", encoding="utf-8") as fh:
                        json.dump(items, fh)
                        fh.flush()
                        os.fsync(fh.fileno())
                    tmp.replace(path)
                except OSError as exc:
                    logger.exception("failed to rebuild stub cache", exc_info=exc)
            return cache
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.exception("failed to load stub cache", exc_info=exc)
        warnings.warn(
            "stub cache unreadable; using empty in-memory cache",
            StubCacheWarning,
        )
        try:
            backup = path.with_suffix(".corrupt")
            path.replace(backup)
        except OSError as backup_exc:
            logger.exception("failed to back up corrupt cache", exc_info=backup_exc)
    return OrderedDict()


def _valid_cache_item(key: Tuple[str, str], value: Dict[str, Any]) -> bool:
    if not (
        isinstance(key, tuple)
        and len(key) == 2
        and all(isinstance(p, str) for p in key)
    ):
        return False
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value.keys()):
        return False
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _save_cache(config: StubProviderConfig | None = None) -> None:
    config = config or get_config()
    path = config.cache_path
    lock_path = str(path) + ".lock"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(lock_path)
        with lock:
            with _CACHE_LOCK:
                items: List[List[Any]] = []
                invalid: List[Tuple[str, str]] = []
                for k, v in _CACHE.items():
                    if _valid_cache_item(k, v):
                        items.append([f"{k[0]}::{k[1]}", v])
                    else:
                        invalid.append(k)
                for k in invalid:
                    _CACHE.pop(k, None)
                data = items
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
                fh.flush()
                os.fsync(fh.fileno())
            tmp.replace(path)
    except (OSError, TypeError, Timeout) as exc:
        logger.exception("failed to save stub cache", exc_info=exc)
        warnings.warn(
            "failed to persist stub cache; using in-memory cache only",
            StubCacheWarning,
        )


async def _async_load_cache(
    config: StubProviderConfig | None = None,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Asynchronously load stub cache from disk with a file lock."""
    config = config or get_config()

    async def _locked_load() -> Dict[Tuple[str, str], Dict[str, Any]]:
        return await asyncio.to_thread(_load_cache)

    lock_path = str(config.cache_path) + ".lock"
    delay = 0.05
    for attempt in range(3):
        lock = FileLock(lock_path)
        try:
            lock.acquire(timeout=0.1)
            try:
                return await _locked_load()
            finally:
                lock.release()
        except Timeout:
            logger.warning(
                "stub cache load lock busy (attempt %d)", attempt + 1
            )
            await asyncio.sleep(delay)
            delay *= 2
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("unexpected cache load error", exc_info=exc)
            warnings.warn(
                "stub cache unavailable; using empty in-memory cache",
                StubCacheWarning,
            )
            return OrderedDict()
    warnings.warn(
        "stub cache load lock timeout; using empty in-memory cache",
        StubCacheWarning,
    )
    return OrderedDict()


async def _async_save_cache(config: StubProviderConfig | None = None) -> None:
    """Asynchronously persist stub cache to disk with a file lock."""
    config = config or get_config()

    async def _locked_save() -> None:
        await asyncio.to_thread(_save_cache)

    lock_path = str(config.cache_path) + ".lock"
    delay = 0.05
    for attempt in range(3):
        lock = FileLock(lock_path)
        try:
            lock.acquire(timeout=0.1)
            try:
                await _locked_save()
                return
            finally:
                lock.release()
        except Timeout:
            logger.warning(
                "stub cache save lock busy (attempt %d)", attempt + 1
            )
            await asyncio.sleep(delay)
            delay *= 2
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("unexpected cache save error", exc_info=exc)
            warnings.warn(
                "stub cache save failed; using in-memory cache only",
                StubCacheWarning,
            )
            return
    logger.error("giving up on saving stub cache after lock timeouts")
    warnings.warn(
        "stub cache save lock timeout; using in-memory cache only",
        StubCacheWarning,
    )

# backward compatible aliases
_aload_cache = _async_load_cache
_asave_cache = _async_save_cache


def _cache_evict(config: StubProviderConfig) -> None:
    """Evict least recently used cache entries when exceeding limit.

    Caller must hold ``_CACHE_LOCK``.
    """
    while len(_CACHE) > config.cache_max:
        try:
            _CACHE.popitem(last=False)
        except KeyError:
            break


def _schedule_cache_persist(config: StubProviderConfig) -> None:
    """Persist the cache in the background."""

    async def _runner() -> None:
        try:
            await asyncio.shield(_async_save_cache())
        except Exception as exc:
            logger.exception("failed to save stub cache", exc_info=exc)

    task = asyncio.create_task(_runner())
    _SAVE_TASKS.add(task)


# load persistent cache at import time
with _CACHE_LOCK:
    _CACHE.update(_load_cache(CONFIG))
    _cache_evict(CONFIG)


def _atexit_save_cache() -> None:
    """Persist the cache on shutdown without blocking the event loop."""

    config = get_config()

    async def _wait_and_save() -> None:
        await _SAVE_TASKS.__aexit__(None, None, None)
        try:
            await asyncio.to_thread(_save_cache, config)
        except RuntimeError:
            _save_cache(config)

    try:
        loop: asyncio.AbstractEventLoop | None = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None or loop.is_closed():
            asyncio.run(_wait_and_save())
        else:
            loop.run_until_complete(_wait_and_save())
    except Exception as exc:
        logger.exception("cache save failed", exc_info=exc)


atexit.register(_atexit_save_cache)


def flush_caches(config: StubProviderConfig | None = None) -> None:
    """Persist and clear in-memory caches."""

    cid = f"stub-flush-{uuid.uuid4()}"
    set_correlation_id(cid)
    logger.info("flush caches start", extra=log_record(event="shutdown"))
    cfg = config or get_config()

    async def _wait() -> None:
        # gather pending save tasks and log any failures
        with _SAVE_TASKS._lock:
            tasks = list(_SAVE_TASKS._tasks)
            _SAVE_TASKS._tasks.clear()
        if not tasks:
            return
        try:
            done, pending = await asyncio.wait(tasks, timeout=cfg.save_timeout)
        except Exception as exc:
            logger.exception("cache save task await failed", exc_info=exc)
        else:
            for t in done:
                if (exc := t.exception()) is not None:
                    logger.exception("cache save task failed", exc_info=exc)
            if pending:
                for t in pending:
                    logger.warning(
                        "cache save task exceeded timeout of %s seconds", cfg.save_timeout
                    )
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None or loop.is_closed():
            asyncio.run(_wait())
        else:  # pragma: no cover - requires running loop
            loop.run_until_complete(_wait())
    except Exception as exc:  # pragma: no cover - best effort
        logger.exception("failed to await cache tasks", exc_info=exc)

    with _CACHE_LOCK:
        try:
            if _CACHE:
                _save_cache(cfg)
        except Exception as exc:  # pragma: no cover - best effort
            logger.exception("failed to save stub cache", exc_info=exc)
        finally:
            _CACHE.clear()
            _TARGET_STATS.clear()
    logger.info("flush caches complete", extra=log_record(event="shutdown"))
    set_correlation_id(None)


def cleanup_cache_files(config: StubProviderConfig | None = None) -> None:
    """Remove obsolete on-disk cache artefacts."""

    cid = f"stub-clean-{uuid.uuid4()}"
    set_correlation_id(cid)
    logger.info("cleanup cache files start", extra=log_record(event="shutdown"))

    cfg = config or get_config()
    with _CACHE_LOCK:
        if _CACHE:
            set_correlation_id(None)
            return

    paths = [
        cfg.cache_path,
        cfg.cache_path.with_suffix(".tmp"),
        Path(str(cfg.cache_path) + ".lock"),
    ]
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug(
                "failed to remove cache file %s: %s", path_for_prompt(path), exc
            )
    logger.info(
        "cleanup cache files complete", extra=log_record(event="shutdown")
    )
    set_correlation_id(None)


def _cache_key(func_name: str, stub: Dict[str, Any]) -> Tuple[str, str]:
    """Return a stable cache key for *func_name* and *stub*."""
    try:
        stub_key = json.dumps(stub, sort_keys=True, default=str)
    except TypeError:
        stub_key = repr(stub)
    return func_name, stub_key


class ModelLoadError(RuntimeError):
    """Raised when no suitable stub generation model can be loaded."""


async def _load_openai_generator() -> Any:
    """Load and return the OpenAI client if available.

    SelfCodingEngine now performs generation locally; this loader is kept for
    legacy paths that still rely on OpenAI.
    """

    if not _feature_enabled("SANDBOX_ENABLE_OPENAI"):
        raise ModelLoadError(
            "openai support disabled; set SANDBOX_ENABLE_OPENAI=1"
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise ModelLoadError("OPENAI_API_KEY missing for openai usage")
    try:
        global openai
        if openai is None:
            openai = importlib.import_module("openai")  # type: ignore
    except ImportError as exc:  # pragma: no cover - library not installed
        raise ModelLoadError(
            "openai library unavailable; install the 'openai' package"
        ) from exc
    openai.api_key = os.getenv("OPENAI_API_KEY")
    return openai


async def _load_fallback_pipeline(cfg: StubProviderConfig) -> Any:
    """Load the bundled lightweight model via :mod:`transformers`."""

    if not _feature_enabled("SANDBOX_ENABLE_TRANSFORMERS"):
        raise ModelLoadError(
            "transformers support disabled; set SANDBOX_ENABLE_TRANSFORMERS=1"
        )
    try:
        global pipeline
        if pipeline is None:
            transformers = importlib.import_module("transformers")
            pipeline = transformers.pipeline  # type: ignore[attr-defined]
    except ImportError as exc:  # pragma: no cover - library not installed
        raise ModelLoadError(
            "transformers library unavailable; install the 'transformers' package"
        ) from exc
    if pipeline is None:  # pragma: no cover - defensive
        raise ModelLoadError("transformers pipeline unavailable")
    try:
        gen = await asyncio.to_thread(
            pipeline,
            "text-generation",
            model=cfg.fallback_model,
            local_files_only=True,
        )
        return gen
    except Exception as exc:  # pragma: no cover - model load failures
        raise ModelLoadError(
            f"bundled fallback model {cfg.fallback_model} could not be loaded"
        ) from exc


async def _aload_generator(config: StubProviderConfig | None = None) -> Any:
    """Return an :class:`LLMClient` for stub generation."""
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR

    settings = get_settings()
    model = settings.sandbox_stub_model
    if not model:
        return None
    backend = getattr(settings, "llm_backend", "openai")
    try:
        _GENERATOR = get_client(backend, model=model)
        return _GENERATOR
    except Exception as exc:  # pragma: no cover - backend load failures
        raise ModelLoadError(f"stub model {model!r} could not be loaded") from exc


def _load_generator(config: StubProviderConfig | None = None):
    """Synchronous wrapper for :func:`_aload_generator`."""
    return asyncio.run(_aload_generator(config))


def _get_history_db() -> InputHistoryDB:
    path = os.getenv(
        "SANDBOX_INPUT_HISTORY",
        str(resolve_path("sandbox_data/input_history.db")),
    )
    return InputHistoryDB(path)


def _aggregate(records: List[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, List[Any]] = {}
    for rec in records:
        for k, v in rec.items():
            stats.setdefault(k, []).append(v)
    result: dict[str, Any] = {}
    for k, vals in stats.items():
        if all(isinstance(v, (int, float)) for v in vals):
            avg = sum(float(v) for v in vals) / len(vals)
            if all(isinstance(v, int) for v in vals):
                avg = int(round(avg))
            result[k] = avg
        else:
            cnt = Counter(vals)
            result[k] = cnt.most_common(1)[0][0]
    return result


async def async_generate_stubs(
    stubs: List[Dict[str, Any]], ctx: dict, config: StubProviderConfig | None = None
) -> List[Dict[str, Any]]:
    cid = ctx.get("correlation_id") or f"stub-{uuid.uuid4()}"
    set_correlation_id(cid)
    stub_generation_requests_total.inc()
    logger.info(
        "stub generation start", extra=log_record(strategy=ctx.get("strategy"))
    )
    try:
        return await _async_generate_stubs(stubs, ctx, config)
    except Exception:
        stub_generation_failures_total.inc()
        logger.exception(
            "stub generation failure", extra=log_record(event="failure")
        )
        raise
    finally:
        set_correlation_id(None)


async def _async_generate_stubs(
    stubs: List[Dict[str, Any]], ctx: dict, config: StubProviderConfig | None = None
) -> List[Dict[str, Any]]:
    """Generate or enhance ``stubs`` using recent history or a language model."""
    config = config or get_config()

    strategy = ctx.get("strategy")

    if not _CACHE:
        try:
            loaded = await _async_load_cache(config)
            with _CACHE_LOCK:
                if not _CACHE:
                    _CACHE.update(loaded)
                    _cache_evict(config)
        except Exception as exc:
            logger.exception("failed to load stub cache", exc_info=exc)
    if strategy == "history":
        try:
            records = _get_history_db().recent(50)
        except Exception as exc:
            logger.exception("failed to load input history", exc_info=exc)
            records = []
        if records:
            hist = _aggregate(records)
            func = ctx.get("target")
            if not _validate_stub_signature(hist, func):
                raise RuntimeError("historical stub does not match target signature")
            return [dict(hist) for _ in range(max(1, len(stubs)))]
        return stubs

    try:
        gen = await _aload_generator()
    except ModelLoadError as exc:
        logger.error("stub generation unavailable: %s", exc)
        try:
            records = _get_history_db().recent(50)
        except Exception as exc2:
            logger.exception("failed to load input history", exc_info=exc2)
            records = []
        if records:
            hist = _aggregate(records)
            func = ctx.get("target")
            if not _validate_stub_signature(hist, func):
                raise RuntimeError("historical stub does not match target signature")
            return [dict(hist) for _ in range(max(1, len(stubs)))]
        return stubs
    if gen is None:
        try:
            records = _get_history_db().recent(50)
        except Exception as exc:
            logger.exception("failed to load input history", exc_info=exc)
            records = []
        if records:
            hist = _aggregate(records)
            func = ctx.get("target")
            if not _validate_stub_signature(hist, func):
                raise RuntimeError("historical stub does not match target signature")
            return [dict(hist) for _ in range(max(1, len(stubs)))]
        return stubs

    template: str = ctx.get(
        "prompt_template",
        (
            "Create a JSON object for '{name}' using arguments with example values: {args}. "
            "Return only the JSON object."
        ),
    )

    new_stubs: List[Dict[str, Any]] = []
    changed = False
    for stub in stubs:
        func = ctx.get("target")
        name = getattr(func, "__name__", "function")
        key = _cache_key(name, stub)
        with _CACHE_LOCK:
            stats = _TARGET_STATS.setdefault(name, Counter())
            cached = _CACHE.get(key)
            stub_key = None
            if cached is not None:
                try:
                    if hasattr(_CACHE, "move_to_end"):
                        _CACHE.move_to_end(key)
                except Exception as exc:
                    logger.warning(
                        "failed to update cache LRU for key %s: %s", key, exc
                    )
                try:
                    stub_key = json.dumps(cached, sort_keys=True, default=str)
                except TypeError:
                    stub_key = repr(cached)
        func = ctx.get("target")
        if cached is not None:
            if not _validate_stub_signature(cached, func):
                with _CACHE_LOCK:
                    _CACHE.pop(key, None)
            else:
                with _CACHE_LOCK:
                    stats = _TARGET_STATS.setdefault(name, Counter())
                    if stub_key is None:
                        try:
                            stub_key = json.dumps(cached, sort_keys=True, default=str)
                        except TypeError:
                            stub_key = repr(cached)
                    stats[stub_key] += 1
                new_stubs.append(dict(cached))
                continue
        args = ", ".join(f"{k}={v!r}" for k, v in stub.items())
        intent_meta = {"stub_args": dict(stub)}
        prompt_obj = cb_build_prompt(
            template.format(name=name, args=args),
            intent_metadata=intent_meta,
        )

        async def _invoke() -> str:
            call = getattr(gen, "generate", gen)
            result = call(prompt_obj)  # type: ignore[attr-defined]
            if isinstance(result, asyncio.Task):
                result = await result
            if hasattr(result, "text"):
                return result.text
            if isinstance(result, list) and result and isinstance(result[0], dict):
                return result[0].get("generated_text", "")
            return str(result)

        try:
            text = await _call_with_retry(_invoke, config)
            match = re.search(r"{.*}", text, flags=re.S)
            if match:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    func = ctx.get("target")
                    if not _validate_stub_signature(data, func):
                        raise ValueError("type mismatch")
                    with _CACHE_LOCK:
                        _CACHE[key] = data
                        try:
                            if hasattr(_CACHE, "move_to_end"):
                                _CACHE.move_to_end(key)
                        except Exception as exc:
                            logger.warning(
                                "failed to update cache LRU for key %s: %s", key, exc
                            )
                        _cache_evict(config)
                        try:
                            stub_key = json.dumps(data, sort_keys=True, default=str)
                        except TypeError:
                            stub_key = repr(data)
                        _TARGET_STATS.setdefault(name, Counter())[stub_key] += 1
                    changed = True
                    new_stubs.append(dict(data))
                    continue
            raise ValueError("invalid generation output")
        except Exception as exc:  # pragma: no cover - generation failures
            logger.exception("stub generation failed")
            raise RuntimeError("stub generation failed") from exc

    if changed:
        _schedule_cache_persist(config)

    return new_stubs


def generate_stubs(
    stubs: List[Dict[str, Any]], ctx: dict, config: StubProviderConfig | None = None
) -> List[Dict[str, Any]]:
    """Synchronous wrapper for :func:`async_generate_stubs`."""
    config = config or get_config()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # no loop is running
        return asyncio.run(async_generate_stubs(stubs, ctx, config))
    else:  # pragma: no cover - requires active event loop
        fut = asyncio.ensure_future(async_generate_stubs(stubs, ctx, config), loop=loop)
        return loop.run_until_complete(fut)
