"""Utilities for restoring deprecated module-level symbols.

These helpers let package ``__getattr__`` implementations expose legacy names
without hard import failures during module import.
"""

from __future__ import annotations

from importlib import import_module
import warnings
from typing import Mapping


class _MissingLegacySymbol:
    """Placeholder object that raises when actively used."""

    def __init__(self, symbol: str, target: str, error: Exception) -> None:
        self._symbol = symbol
        self._target = target
        self._error = error

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise ImportError(
            f"Legacy symbol {self._symbol!r} could not resolve target {self._target!r}."
        ) from self._error


def resolve_legacy_symbol(
    *,
    symbol: str,
    aliases: Mapping[str, str],
    package_name: str,
) -> object:
    """Resolve *symbol* from ``aliases`` while warning about deprecation."""

    target = aliases[symbol]
    warnings.warn(
        (
            f"{package_name}.{symbol} is deprecated; import {target} instead. "
            "Legacy alias remains for compatibility."
        ),
        DeprecationWarning,
        stacklevel=3,
    )

    module_name, _, attr_name = target.partition(":")
    try:
        module = import_module(module_name)
    except Exception as exc:  # pragma: no cover - fallback path
        return _MissingLegacySymbol(symbol, target, exc)

    return getattr(module, attr_name) if attr_name else module
