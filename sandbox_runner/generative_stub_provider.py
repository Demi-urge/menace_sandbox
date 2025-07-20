from __future__ import annotations

"""Stub provider using a language model via ``transformers``."""

from typing import Any, Dict, List, Tuple
import asyncio
import inspect
import json
import logging
import os
import re
from pathlib import Path
from collections import Counter
import atexit

from .input_history_db import InputHistoryDB

ROOT = Path(__file__).resolve().parents[1]

_STUB_CACHE_PATH = Path(
    os.getenv("SANDBOX_STUB_CACHE", str(ROOT / "sandbox_data" / "stub_cache.json"))
)

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional dependency
    pipeline = None  # type: ignore

logger = logging.getLogger(__name__)

_GENERATOR = None
_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


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
    return {}


def _save_cache() -> None:
    try:
        _STUB_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {f"{k[0]}::{k[1]}": v for k, v in _CACHE.items()}
        with open(_STUB_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
    except Exception:
        logger.exception("failed to save stub cache")


async def _aload_cache() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Asynchronously load stub cache from disk."""
    return await asyncio.to_thread(_load_cache)


async def _asave_cache() -> None:
    """Asynchronously persist stub cache to disk."""
    await asyncio.to_thread(_save_cache)


# load persistent cache at import time
_CACHE.update(_load_cache())


def _atexit_save_cache() -> None:
    """Persist the cache on shutdown without blocking the event loop."""
    try:
        asyncio.run(asyncio.to_thread(_save_cache))
    except Exception:
        pass


atexit.register(_atexit_save_cache)


def _cache_key(func_name: str, stub: Dict[str, Any]) -> Tuple[str, str]:
    """Return a stable cache key for *func_name* and *stub*."""
    try:
        stub_key = json.dumps(stub, sort_keys=True, default=str)
    except TypeError:
        stub_key = repr(stub)
    return func_name, stub_key


async def _aload_generator():
    """Return a text generation pipeline or ``None`` when unavailable."""
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR
    if pipeline is None:
        return None
    model = os.getenv("SANDBOX_STUB_MODEL")
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
    if gen is None:
        return stubs

    template: str = ctx.get(
        "prompt_template",
        "Create a JSON object for '{name}' using arguments with example values: {args}. Return only the JSON object.",
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

        try:
            if use_stream and hasattr(gen, "stream"):
                stream_res = gen.stream(prompt, **gen_kwargs)
                if inspect.isasyncgen(stream_res):
                    parts = [p async for p in stream_res]
                else:
                    parts = list(stream_res)
                text = "".join(parts)
            else:
                if inspect.iscoroutinefunction(getattr(gen, "__call__", gen)):
                    result = await gen(prompt, **gen_kwargs)
                else:
                    result = await asyncio.to_thread(gen, prompt, **gen_kwargs)
                text = result[0].get("generated_text", "")
            match = re.search(r"{.*}", text, flags=re.S)
            if match:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    _CACHE[key] = data
                    changed = True
                    new_stubs.append(dict(data))
                    continue
        except Exception:  # pragma: no cover - generation failures
            logger.exception("stub generation failed")
        _CACHE[key] = stub
        changed = True
        new_stubs.append(dict(stub))

    if changed:
        try:
            await _asave_cache()
        except Exception:
            logger.exception("failed to save stub cache")

    return new_stubs


def generate_stubs(stubs: List[Dict[str, Any]], ctx: dict) -> List[Dict[str, Any]]:
    """Synchronous wrapper for :func:`async_generate_stubs`."""
    return asyncio.run(async_generate_stubs(stubs, ctx))
