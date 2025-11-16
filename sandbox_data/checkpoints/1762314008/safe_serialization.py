"""Utilities for serialising complex objects without triggering recursion errors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
from typing import Any

_PRIMITIVE_TYPES = (str, int, float, bool, type(None))


def _is_dataclass_instance(value: Any) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def sanitize_for_json(value: Any, *, _memo: set[int] | None = None) -> Any:
    """Return *value* converted into a JSON-serialisable structure."""

    if isinstance(value, _PRIMITIVE_TYPES):
        return value

    if _memo is None:
        _memo = set()

    obj_id = id(value)
    if obj_id in _memo:
        return "<recursion>"

    _memo.add(obj_id)
    try:
        if _is_dataclass_instance(value):
            return sanitize_for_json(dataclasses.asdict(value), _memo=_memo)

        if isinstance(value, Mapping):
            return {str(k): sanitize_for_json(v, _memo=_memo) for k, v in value.items()}

        if isinstance(value, (set, frozenset)):
            return [sanitize_for_json(item, _memo=_memo) for item in sorted(value, key=repr)]

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [sanitize_for_json(item, _memo=_memo) for item in value]

        if hasattr(value, "__dict__"):
            data = {
                str(key): sanitize_for_json(val, _memo=_memo)
                for key, val in value.__dict__.items()
                if not key.startswith("_")
            }
            data.setdefault("__class__", value.__class__.__name__)
            return data

        if hasattr(value, "__slots__"):
            data = {}
            for slot in value.__slots__:
                if hasattr(value, slot):
                    data[str(slot)] = sanitize_for_json(getattr(value, slot), _memo=_memo)
            data.setdefault("__class__", value.__class__.__name__)
            return data

        return repr(value)
    finally:
        _memo.discard(obj_id)


def safe_json_dumps(value: Any, *, sort_keys: bool = False) -> str:
    """Serialise *value* to JSON, pruning recursion and unserialisable types."""

    cleaned = sanitize_for_json(value)
    try:
        return json.dumps(cleaned, sort_keys=sort_keys)
    except TypeError:
        return json.dumps(repr(cleaned), sort_keys=sort_keys)


__all__ = ["sanitize_for_json", "safe_json_dumps"]

