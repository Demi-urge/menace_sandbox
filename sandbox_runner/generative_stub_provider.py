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
from collections import Counter
import atexit
import importlib

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
_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}

_SAVE_TASKS: set[asyncio.Task[None]] = set()


def _feature_enabled(name: str) -> bool:
    """Return True if feature flag *name* is truthy."""
    val = os.getenv(name, "").lower()
    return val in {"1", "true", "yes"}


_RATE_LIMIT = asyncio.Semaphore(int(os.getenv("SANDBOX_STUB_MAX_CONCURRENCY", "1")))
_GEN_TIMEOUT = float(os.getenv("SANDBOX_STUB_TIMEOUT", "10"))
_GEN_RETRIES = int(os.getenv("SANDBOX_STUB_RETRIES", "2"))


async def _call_with_retry(func: Callable[[], Awaitable[Any]]) -> Any:
    """Invoke *func* with retry, timeout and rate limiting."""
    for attempt in range(_GEN_RETRIES):
        try:
            async with _RATE_LIMIT:
                return await asyncio.wait_for(func(), timeout=_GEN_TIMEOUT)
        except Exception:
            if attempt == _GEN_RETRIES - 1:
                raise
            await asyncio.sleep(0.5 * (attempt + 1))


def _deterministic_stub(stub: Dict[str, Any], func: Any | None) -> Dict[str, Any]:
    """Fill missing fields in *stub* using simple deterministic defaults."""
    if func is None:
        return dict(stub)
    try:
        sig = inspect.signature(func)
    except Exception:
        return dict(stub)
    result: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        ann = param.annotation
        if ann in (int, "int"):
            val: Any = 0
        elif ann in (float, "float"):
            val = 0.0
        elif ann in (bool, "bool"):
            val = False
        elif ann in (str, "str"):
            val = ""
        else:
            val = None
        result[name] = val
    result.update(stub)
    return result


def _load_cache() -> Dict[Tuple[str, str], Dict[str, Any]]:
    try:
        if _STUB_CACHE_PATH.exists():
            with open(_STUB_CACHE_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
                for k, v in data.items():
                    if not isinstance(k, str) or not isinstance(v, dict):
                        continue
                    parts = k.split("::", 1)
                    if len(parts) == 2:
                        cache[(parts[0], parts[1])] = v
                return cache
    except Exception:
        logger.exception("failed to load stub cache")
        try:
            backup = _STUB_CACHE_PATH.with_suffix(".corrupt")
            _STUB_CACHE_PATH.replace(backup)
        except Exception:
            logger.exception("failed to back up corrupt cache")
    return {}


def _save_cache() -> None:
    try:
        _STUB_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {f"{k[0]}::{k[1]}": v for k, v in _CACHE.items()}
        tmp = _STUB_CACHE_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        tmp.replace(_STUB_CACHE_PATH)
    except Exception:
        logger.exception("failed to save stub cache")


async def _aload_cache() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Asynchronously load stub cache from disk."""
    return await asyncio.to_thread(_load_cache)


async def _asave_cache() -> None:
    """Asynchronously persist stub cache to disk."""
    await asyncio.to_thread(_save_cache)


def _schedule_cache_persist() -> None:
    """Persist the cache in the background."""

    async def _runner() -> None:
        try:
            await _asave_cache()
        except Exception:
            logger.exception("failed to save stub cache")

    task = asyncio.create_task(_runner())
    _SAVE_TASKS.add(task)
    task.add_done_callback(lambda t: _SAVE_TASKS.discard(t))


# load persistent cache at import time
_CACHE.update(_load_cache())


def _atexit_save_cache() -> None:
    """Persist the cache on shutdown without blocking the event loop."""
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None or loop.is_closed():
            _save_cache()
        else:
            loop.run_until_complete(asyncio.to_thread(_save_cache))
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
    for name in candidates:
        try:
            _GENERATOR = await asyncio.to_thread(pipeline, "text-generation", model=name)
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
            _CACHE.update(await _aload_cache())
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
        return [_deterministic_stub(s, func) for s in base]

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
        cached = _CACHE.get(key)
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
                    params: List[str] = []
                    if func is not None:
                        try:
                            sig = inspect.signature(func)
                            params = [
                                n
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
                    for p_name in params:
                        if p_name not in data:
                            raise ValueError("missing field")
                        if (
                            p_name in stub
                            and stub[p_name] is not None
                            and not isinstance(data[p_name], type(stub[p_name]))
                        ):
                            raise ValueError("type mismatch")
                    _CACHE[key] = data
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
