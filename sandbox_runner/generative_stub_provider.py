from __future__ import annotations

"""Stub provider using a language model via ``transformers``."""

from typing import Any, Dict, List, Tuple, Callable, Awaitable
import asyncio
import inspect
import json
import logging
import os
import re
from pathlib import Path
from collections import Counter, OrderedDict
import atexit
import importlib
import random
import threading
from contextlib import AbstractAsyncContextManager
from typing import get_origin, get_args, Union

from .input_history_db import InputHistoryDB

ROOT = Path(__file__).resolve().parents[1]

_STUB_CACHE_PATH = Path(
    os.getenv("SANDBOX_STUB_CACHE", str(ROOT / "sandbox_data" / "stub_cache.json"))
)


# Optional dependencies loaded lazily
pipeline = None  # type: ignore
openai = None  # type: ignore

logger = logging.getLogger(__name__)

_GENERATOR = None
# use OrderedDict for LRU eviction semantics
_CACHE: "OrderedDict[Tuple[str, str], Dict[str, Any]]" = OrderedDict()

# protect cache mutations across threads and async tasks
_CACHE_LOCK = threading.Lock()


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


_RATE_LIMIT = asyncio.Semaphore(int(os.getenv("SANDBOX_STUB_MAX_CONCURRENCY", "1")))
_GEN_TIMEOUT = 10.0
_GEN_RETRIES = 2
_RETRY_BASE = 0.5
_RETRY_MAX = 30.0
_CACHE_MAX = 1024


def _validate_env() -> None:
    """Validate environment configuration and log warnings."""

    def _float_env(name: str, default: float) -> float:
        val = os.getenv(name)
        if val is None:
            return default
        try:
            f = float(val)
            if f <= 0:
                raise ValueError
            return f
        except Exception:
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
        except Exception:
            logger.warning("invalid %s=%r; using default %s", name, val, default)
            return default

    global _GEN_TIMEOUT, _GEN_RETRIES, _RETRY_BASE, _RETRY_MAX, _CACHE_MAX
    _GEN_TIMEOUT = _float_env("SANDBOX_STUB_TIMEOUT", 10.0)
    _GEN_RETRIES = _int_env("SANDBOX_STUB_RETRIES", 2)
    _RETRY_BASE = _float_env("SANDBOX_STUB_RETRY_BASE", 0.5)
    _RETRY_MAX = _float_env("SANDBOX_STUB_RETRY_MAX", 30.0)
    _CACHE_MAX = _int_env("SANDBOX_STUB_CACHE_MAX", 1024)

    model = os.getenv("SANDBOX_STUB_MODEL")
    if model and model not in {"openai", "gpt2-large", "distilgpt2"}:
        logger.warning("unknown SANDBOX_STUB_MODEL %s", model)


_validate_env()


async def _call_with_retry(func: Callable[[], Awaitable[Any]]) -> Any:
    """Invoke *func* with retry, timeout and rate limiting."""
    delay = _RETRY_BASE
    for attempt in range(_GEN_RETRIES):
        try:
            async with _RATE_LIMIT:
                return await asyncio.wait_for(func(), timeout=_GEN_TIMEOUT)
        except Exception:
            if attempt == _GEN_RETRIES - 1:
                raise
            jitter = random.uniform(0, delay)
            await asyncio.sleep(jitter)
            delay = min(delay * 2, _RETRY_MAX)


def _rule_based_stub(stub: Dict[str, Any], func: Any | None) -> Dict[str, Any]:
    """Fill missing fields using simple deterministic, context aware rules."""
    if func is None:
        return dict(stub)
    try:
        sig = inspect.signature(func)
    except Exception:
        return dict(stub)
    result = dict(stub)
    for name, param in sig.parameters.items():
        if name in result and result[name] is not None:
            continue
        ann = param.annotation
        lname = name.lower()
        if ann in (int, "int"):
            val = 1
        elif ann in (float, "float"):
            val = 1.0
        elif ann in (bool, "bool"):
            val = True
        elif ann in (str, "str"):
            if any(token in lname for token in ["name", "title", "id"]):
                val = "example"
            else:
                val = "value"
        else:
            val = None
        result[name] = val
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
        return isinstance(value, tuple) and len(value) == len(args) and all(
            _type_matches(v, t) for v, t in zip(value, args)
        )
    if origin is set:
        (arg,) = get_args(annotation) or (Any,)
        return isinstance(value, set) and all(_type_matches(v, arg) for v in value)
    if origin is Union:
        return any(_type_matches(value, arg) for arg in get_args(annotation))
    return True


def _load_cache() -> "OrderedDict[Tuple[str, str], Dict[str, Any]]":
    try:
        if _STUB_CACHE_PATH.exists():
            with open(_STUB_CACHE_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                cache: "OrderedDict[Tuple[str, str], Dict[str, Any]]" = OrderedDict()
                for item in data:
                    if not (
                        isinstance(item, list)
                        and len(item) == 2
                        and isinstance(item[0], str)
                        and isinstance(item[1], dict)
                    ):
                        continue
                    parts = item[0].split("::", 1)
                    if len(parts) == 2:
                        cache[(parts[0], parts[1])] = item[1]
                return cache
    except Exception:
        logger.exception("failed to load stub cache")
        try:
            backup = _STUB_CACHE_PATH.with_suffix(".corrupt")
            _STUB_CACHE_PATH.replace(backup)
        except Exception:
            logger.exception("failed to back up corrupt cache")
    return OrderedDict()


def _save_cache() -> None:
    try:
        _STUB_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _CACHE_LOCK:
            data = [[f"{k[0]}::{k[1]}", v] for k, v in _CACHE.items()]
        tmp = _STUB_CACHE_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(_STUB_CACHE_PATH)
    except Exception:
        logger.exception("failed to save stub cache")


async def _aload_cache() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Asynchronously load stub cache from disk."""
    return await asyncio.to_thread(_load_cache)


async def _asave_cache() -> None:
    """Asynchronously persist stub cache to disk."""
    await asyncio.to_thread(_save_cache)


def _cache_evict() -> None:
    """Evict least recently used cache entries when exceeding limit.

    Caller must hold ``_CACHE_LOCK``.
    """
    while len(_CACHE) > _CACHE_MAX:
        try:
            _CACHE.popitem(last=False)
        except Exception:
            break


def _schedule_cache_persist() -> None:
    """Persist the cache in the background."""

    async def _runner() -> None:
        try:
            await asyncio.shield(_asave_cache())
        except Exception:
            logger.exception("failed to save stub cache")

    task = asyncio.create_task(_runner())
    _SAVE_TASKS.add(task)


# load persistent cache at import time
with _CACHE_LOCK:
    _CACHE.update(_load_cache())
    _cache_evict()


def _atexit_save_cache() -> None:
    """Persist the cache on shutdown without blocking the event loop."""

    async def _wait_and_save() -> None:
        await _SAVE_TASKS.__aexit__(None, None, None)
        try:
            await asyncio.to_thread(_save_cache)
        except RuntimeError:
            _save_cache()

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
    except Exception:
        logger.exception("cache save failed")


atexit.register(_atexit_save_cache)


def _cache_key(func_name: str, stub: Dict[str, Any]) -> Tuple[str, str]:
    """Return a stable cache key for *func_name* and *stub*."""
    try:
        stub_key = json.dumps(stub, sort_keys=True, default=str)
    except TypeError:
        stub_key = repr(stub)
    return func_name, stub_key


async def _aload_generator():
    """Return a text generation pipeline, the OpenAI module or ``None``."""
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR
    model = os.getenv("SANDBOX_STUB_MODEL")

    if model == "openai":
        if not _feature_enabled("SANDBOX_ENABLE_OPENAI"):
            logger.error("openai support disabled; set SANDBOX_ENABLE_OPENAI=1")
            return None
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY missing for openai usage")
            return None
        try:
            global openai
            if openai is None:
                openai = importlib.import_module("openai")  # type: ignore
        except Exception:
            logger.error("openai library unavailable")
            return None
        openai.api_key = os.getenv("OPENAI_API_KEY")
        _GENERATOR = openai
        return _GENERATOR

    if not _feature_enabled("SANDBOX_ENABLE_TRANSFORMERS"):
        logger.error("transformers support disabled; set SANDBOX_ENABLE_TRANSFORMERS=1")
        return None

    try:
        global pipeline
        if pipeline is None:
            transformers = importlib.import_module("transformers")
            pipeline = transformers.pipeline  # type: ignore[attr-defined]
    except Exception:
        logger.error("transformers library unavailable")
        return None

    candidates = ["gpt2-large", "distilgpt2"] if model is None else [model]
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
    for name in candidates:
        try:
            kwargs = {"model": name}
            if hf_token:
                kwargs["use_auth_token"] = hf_token
            _GENERATOR = await asyncio.to_thread(
                pipeline, "text-generation", **kwargs
            )
            break
        except Exception:  # pragma: no cover - model load failures
            logger.exception("failed to load model %s", name)
            _GENERATOR = None
    return _GENERATOR


def _load_generator():
    """Synchronous wrapper for :func:`_aload_generator`."""
    return asyncio.run(_aload_generator())


def _get_history_db() -> InputHistoryDB:
    path = os.getenv(
        "SANDBOX_INPUT_HISTORY", str(ROOT / "sandbox_data" / "input_history.db")
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


async def async_generate_stubs(stubs: List[Dict[str, Any]], ctx: dict) -> List[Dict[str, Any]]:
    """Generate or enhance ``stubs`` using recent history or a language model."""

    strategy = ctx.get("strategy")

    if not _CACHE:
        try:
            loaded = await _aload_cache()
            with _CACHE_LOCK:
                if not _CACHE:
                    _CACHE.update(loaded)
                    _cache_evict()
        except Exception:
            logger.exception("failed to load stub cache")
    if strategy == "history":
        try:
            records = _get_history_db().recent(50)
        except Exception:
            logger.exception("failed to load input history")
            records = []
        if records:
            hist = _aggregate(records)
            return [dict(hist) for _ in range(max(1, len(stubs)))]
        return stubs

    gen = await _aload_generator()
    use_openai = gen is openai
    if gen is None:
        func = ctx.get("target")
        base = stubs or [{}]
        return [_rule_based_stub(s, func) for s in base]

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
            cached = _CACHE.get(key)
            if cached is not None:
                try:
                    _CACHE.move_to_end(key)
                except Exception:
                    pass
        if cached is not None:
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
                    "model": os.getenv("OPENAI_STUB_COMPLETION_MODEL", "text-davinci-003"),
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
            text = await _call_with_retry(_invoke)
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
                        except Exception:
                            params = []
                    for p_name, param in params:
                        if p_name not in data:
                            raise ValueError("missing field")
                        if not _type_matches(data[p_name], param.annotation):
                            raise ValueError("type mismatch")
                    with _CACHE_LOCK:
                        _CACHE[key] = data
                        try:
                            _CACHE.move_to_end(key)
                        except Exception:
                            pass
                        _cache_evict()
                    changed = True
                    new_stubs.append(dict(data))
                    continue
            raise ValueError("invalid generation output")
        except Exception as exc:  # pragma: no cover - generation failures
            logger.exception("stub generation failed")
            raise RuntimeError("stub generation failed") from exc

    if changed:
        _schedule_cache_persist()

    return new_stubs


def generate_stubs(stubs: List[Dict[str, Any]], ctx: dict) -> List[Dict[str, Any]]:
    """Synchronous wrapper for :func:`async_generate_stubs`."""
    return asyncio.run(async_generate_stubs(stubs, ctx))
