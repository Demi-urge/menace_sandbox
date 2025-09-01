from __future__ import annotations

"""Stub provider using a language model via ``transformers`` or OpenAI.

The module persists generated stubs on disk and coordinates concurrent access
to the cache using a :class:`filelock.FileLock`.  Loads and saves retry lock
acquisition briefly and emit log messages when contention is encountered so
that potential concurrency issues can be diagnosed.

Required configuration:

* ``SANDBOX_ENABLE_TRANSFORMERS`` with ``SANDBOX_STUB_MODEL`` and
  ``SANDBOX_HUGGINGFACE_TOKEN`` to load a HuggingFace model via
  :func:`transformers.pipeline`.
* ``SANDBOX_ENABLE_OPENAI`` with ``OPENAI_API_KEY`` to use the ``openai``
  backend.
* When neither backend is available a lightweight bundled model specified by
  ``SANDBOX_STUB_FALLBACK_MODEL`` is attempted.  If that model cannot be
  loaded a :class:`RuntimeError` is raised describing the missing
  dependencies.
"""

from typing import Any, Dict, List, Tuple, Callable, Awaitable
import asyncio
import inspect
import json
import logging
import os
import re
import warnings
from pathlib import Path
from collections import Counter, OrderedDict, defaultdict
import atexit
import importlib
from importlib import metadata
import random
import threading
from contextlib import AbstractAsyncContextManager
from typing import get_origin, get_args, Union
import ast
import dataclasses
from dataclasses import dataclass, field
from filelock import FileLock, Timeout

from .input_history_db import InputHistoryDB
from sandbox_settings import SandboxSettings

ROOT = Path(__file__).resolve().parents[1]


# Optional dependencies loaded lazily
pipeline = None  # type: ignore
openai = None  # type: ignore

logger = logging.getLogger(__name__)


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
        os.getenv("SANDBOX_STUB_CACHE", str(ROOT / "sandbox_data" / "stub_cache.json"))
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

    backends: list[str] = []
    transformers_enabled = _feature_enabled("SANDBOX_ENABLE_TRANSFORMERS")
    openai_enabled = _feature_enabled("SANDBOX_ENABLE_OPENAI")
    if transformers_enabled:
        model_env = os.getenv("SANDBOX_STUB_MODEL", settings.sandbox_stub_model)
        token_env = os.getenv("SANDBOX_HUGGINGFACE_TOKEN", settings.huggingface_token)
        if model_env and token_env:
            backends.append("transformers")
        elif model_env or token_env:
            logger.warning(
                "SANDBOX_ENABLE_TRANSFORMERS set but model or token missing"
            )
    if openai_enabled:
        if os.getenv("OPENAI_API_KEY"):
            backends.append("openai")
        else:
            logger.warning(
                "SANDBOX_ENABLE_OPENAI set but OPENAI_API_KEY is missing"
            )
    if transformers_enabled and cfg.fallback_model:
        backends.append("fallback")
    cfg.enabled_backends = tuple(backends)

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
            logger.warning("generation attempt %d failed: %s", attempt + 1, exc)
            jitter = random.uniform(0, delay)
            await asyncio.sleep(jitter)
            delay = min(delay * 2, config.retry_max)


def _rule_based_stub(stub: Dict[str, Any], func: Any | None) -> Dict[str, Any]:
    """Fill missing fields using deterministic rules with type awareness."""

    def _example_from_doc(doc: str, param_name: str) -> Any | None:
        pattern = re.compile(
            rf"{re.escape(param_name)}[^\n]*?e\.g\.[\s]*([^\n\.]+)", re.IGNORECASE
        )
        match = pattern.search(doc)
        if not match:
            return None
        text = match.group(1).strip().rstrip(",;.")
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text.strip("'\"")

    def _value_from_annotation(annotation: Any, name: str) -> Any:
        origin = get_origin(annotation)
        if origin is None:
            lname = name.lower()
            if annotation in (int, "int"):
                if any(t in lname for t in ["count", "num", "size", "len", "quantity"]):
                    return 1
                return 0
            if annotation in (float, "float"):
                if any(t in lname for t in ["ratio", "rate", "percent", "percentage"]):
                    return 0.5
                return float(
                    1
                    if any(
                        t in lname for t in ["count", "num", "size", "len", "quantity"]
                    )
                    else 0
                )
            if annotation in (bool, "bool"):
                return any(
                    t in lname
                    for t in ["is", "has", "can", "should", "enabled", "active"]
                )
            if annotation in (str, "str"):
                if any(t in lname for t in ["name", "title", "id"]):
                    return f"{lname}_example"
                return f"{lname}_value"
            if dataclasses.is_dataclass(annotation):
                kwargs = {
                    f.name: _value_from_annotation(f.type, f.name)
                    for f in dataclasses.fields(annotation)
                }
                return annotation(**kwargs)
            if inspect.isclass(annotation) and annotation not in (
                int,
                float,
                bool,
                str,
            ):
                try:
                    sig = inspect.signature(annotation)
                    kwargs: dict[str, Any] = {}
                    for p_name, param in sig.parameters.items():
                        if p_name == "self":
                            continue
                        if param.default is not inspect._empty:
                            kwargs[p_name] = param.default
                        else:
                            kwargs[p_name] = _value_from_annotation(
                                param.annotation, p_name
                            )
                    return annotation(**kwargs)
                except (TypeError, ValueError) as exc:
                    logger.debug("failed to instantiate %s: %s", annotation, exc)
                    return None
            return None
        args = get_args(annotation)
        if origin is list:
            (arg,) = args or (Any,)
            return [_value_from_annotation(arg, name)]
        if origin is set:
            (arg,) = args or (Any,)
            return {_value_from_annotation(arg, name)}
        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                return (_value_from_annotation(args[0], name),)
            return tuple(_value_from_annotation(a, name) for a in args)
        if origin is dict:
            key_ann, val_ann = args or (str, Any)
            key = _value_from_annotation(key_ann, f"{name}_key")
            val = _value_from_annotation(val_ann, name)
            return {key: val}
        if origin is Union:
            non_none = [a for a in args if a is not type(None)]
            return _value_from_annotation(non_none[0], name) if non_none else None
        return None

    if func is None:
        return dict(stub)
    try:
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""
    except (TypeError, ValueError) as exc:
        logger.debug("signature inspection failed for %s: %s", func, exc)
        return dict(stub)
    result = dict(stub)
    for name, param in sig.parameters.items():
        if name in result and result[name] is not None:
            continue
        if param.default is not inspect._empty:
            result[name] = param.default
            continue
        example = _example_from_doc(doc, name)
        if example is not None:
            result[name] = example
            continue
        result[name] = _value_from_annotation(param.annotation, name)
    return result


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
    """Load and return the OpenAI client if available."""

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
    await asyncio.to_thread(_seed_generator_from_history, openai)
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
        await asyncio.to_thread(_seed_generator_from_history, gen)
        return gen
    except Exception as exc:  # pragma: no cover - model load failures
        raise ModelLoadError(
            f"bundled fallback model {cfg.fallback_model} could not be loaded"
        ) from exc


async def _aload_generator(config: StubProviderConfig | None = None) -> Any:
    """Return a text generation pipeline or raise :class:`RuntimeError`."""
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR

    cfg = config or get_config()
    if not cfg.enabled_backends:
        raise RuntimeError(
            "No text generation backends are enabled; set SANDBOX_ENABLE_TRANSFORMERS "
            "or SANDBOX_ENABLE_OPENAI"
        )

    settings = get_settings()
    errors: dict[str, str] = {}
    for backend in cfg.enabled_backends:
        try:
            if backend == "transformers":
                global pipeline
                if pipeline is None:
                    transformers = importlib.import_module("transformers")
                    pipeline = transformers.pipeline  # type: ignore[attr-defined]
                _GENERATOR = await asyncio.to_thread(
                    pipeline,
                    "text-generation",
                    model=settings.sandbox_stub_model,
                    use_auth_token=settings.huggingface_token,
                )
                await asyncio.to_thread(_seed_generator_from_history, _GENERATOR)
                return _GENERATOR
            if backend == "openai":
                _GENERATOR = await _load_openai_generator()
                return _GENERATOR
            if backend == "fallback":
                _GENERATOR = await _load_fallback_pipeline(cfg)
                return _GENERATOR
        except Exception as exc:  # pragma: no cover - backend load failures
            errors[backend] = str(exc)
            continue
    if errors:
        detail = "; ".join(f"{b}: {e}" for b, e in errors.items())
        raise RuntimeError(f"No text generation backend could be loaded: {detail}")
    raise RuntimeError("No text generation backend could be loaded")


def _load_generator(config: StubProviderConfig | None = None):
    """Synchronous wrapper for :func:`_aload_generator`."""
    return asyncio.run(_aload_generator(config))


def _get_history_db() -> InputHistoryDB:
    path = os.getenv(
        "SANDBOX_INPUT_HISTORY", str(ROOT / "sandbox_data" / "input_history.db")
    )
    return InputHistoryDB(path)


def _seed_generator_from_history(gen: Any) -> None:
    """Seed *gen* with stored input examples when possible."""
    try:
        records = _get_history_db().recent(100)
    except Exception as exc:
        logger.debug("failed to load history for seeding: %s", exc, exc_info=exc)
        return
    if not records:
        return
    payload = "\n".join(json.dumps(r) for r in records)
    for attr in ("seed", "train", "fit"):
        if hasattr(gen, attr):
            try:
                getattr(gen, attr)(payload)
            except Exception as exc:
                logger.debug("stub generator %s failed: %s", attr, exc, exc_info=exc)
            break


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
            return [dict(hist) for _ in range(max(1, len(stubs)))]
        return stubs

    try:
        gen = await _aload_generator()
    except ModelLoadError as exc:
        logger.error("stub generation unavailable: %s", exc)
        return stubs
    if gen is None:
        func = ctx.get("target")
        base = stubs or [{}]
        return [_rule_based_stub(s, func) for s in base]
    use_openai = gen is openai

    template: str = ctx.get(
        "prompt_template",
        (
            "Create a JSON object for '{name}' using arguments with example values: {args}. "
            "Return only the JSON object."
        ),
    )
    temperature = ctx.get("temperature")
    top_p = ctx.get("top_p")
    use_stream = ctx.get("stream", False)

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
        if cached is not None:
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
        prompt = template.format(name=name, args=args)

        gen_kwargs = {"max_length": 64, "num_return_sequences": 1}
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

        async def _invoke() -> str:
            if use_openai:
                comp_kwargs = {
                    "model": os.getenv(
                        "OPENAI_STUB_COMPLETION_MODEL", "text-davinci-003"
                    ),
                    "prompt": prompt,
                    "max_tokens": 64,
                }
                if temperature is not None:
                    comp_kwargs["temperature"] = temperature
                if top_p is not None:
                    comp_kwargs["top_p"] = top_p
                result = await asyncio.to_thread(gen.Completion.create, **comp_kwargs)
                return result["choices"][0]["text"]  # type: ignore
            if use_stream and hasattr(gen, "stream"):
                stream_res = gen.stream(prompt, **gen_kwargs)
                if inspect.isasyncgen(stream_res):
                    parts = [p async for p in stream_res]
                else:
                    parts = list(stream_res)
                return "".join(parts)
            if inspect.iscoroutinefunction(getattr(gen, "__call__", gen)):
                result = await gen(prompt, **gen_kwargs)
            else:
                result = await asyncio.to_thread(gen, prompt, **gen_kwargs)
            return result[0].get("generated_text", "")

        try:
            text = await _call_with_retry(_invoke, config)
            match = re.search(r"{.*}", text, flags=re.S)
            if match:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    func = ctx.get("target")
                    params: List[Tuple[str, inspect.Parameter]] = []
                    if func is not None:
                        try:
                            sig = inspect.signature(func)
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
                        except (TypeError, ValueError) as exc:
                            logger.debug("signature inspection failed for %s: %s", func, exc)
                            params = []
                    for p_name, param in params:
                        if p_name not in data:
                            raise ValueError("missing field")
                        if not _type_matches(data[p_name], param.annotation):
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
