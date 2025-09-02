"""Plugin hooks for input stub generation."""

from __future__ import annotations

import logging
import inspect
from importlib import metadata
from typing import Any, Callable, Dict, Iterable, List, Sequence

# Callback type for stub provider
StubProvider = Callable[[List[Dict[str, Any]], dict], List[Dict[str, Any]]]

logger = logging.getLogger(__name__)

# Entry-point group used to discover stub providers
STUB_PROVIDER_GROUP = "sandbox.stub_providers"


def _iter_entry_points() -> Iterable[metadata.EntryPoint]:
    """Return entry points registered for stub providers."""

    try:
        eps = metadata.entry_points(group=STUB_PROVIDER_GROUP)
    except TypeError:  # pragma: no cover - legacy API
        eps = metadata.entry_points().get(STUB_PROVIDER_GROUP, [])
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed to gather stub provider entry points")
        return []
    return list(eps)


def _validate_provider(func: Any) -> bool:
    """Return ``True`` if ``func`` matches :class:`StubProvider`."""

    if not callable(func):
        return False
    try:
        sig = inspect.signature(func)
    except Exception:  # pragma: no cover - defensive
        return False
    params = list(sig.parameters.values())
    return len(params) == 2


def _load_entry_point(ep: metadata.EntryPoint) -> StubProvider | None:
    try:
        func = ep.load()
    except Exception:
        logger.exception("failed to load stub provider %s", ep.name)
        return None
    if not _validate_provider(func):
        logger.warning("stub provider %s has invalid signature", ep.name)
        return None
    return func  # type: ignore[return-value]


def load_stub_providers(names: Sequence[str]) -> List[StubProvider]:
    """Return providers matching ``names`` discovered via entry points."""

    providers: List[StubProvider] = []
    ep_map = {ep.name: ep for ep in _iter_entry_points()}
    for name in names:
        ep = ep_map.get(name)
        if ep is None:
            logger.warning("stub provider %s not found", name)
            continue
        func = _load_entry_point(ep)
        if func:
            providers.append(func)
    return providers


def discover_stub_providers(settings: Any | None = None) -> List[StubProvider]:
    """Discover stub providers using entry points and settings."""

    if settings is None:
        try:  # pragma: no cover - sandbox_settings may not be available
            from sandbox_settings import SandboxSettings

            settings = SandboxSettings()
        except Exception:  # pragma: no cover - defensive
            settings = None

    enabled = set(getattr(settings, "stub_providers", []) or [])
    disabled = set(getattr(settings, "disabled_stub_providers", []) or [])

    eps = {ep.name: ep for ep in _iter_entry_points()}
    providers: List[StubProvider] = []
    if enabled:
        missing = sorted(enabled - eps.keys())
        if missing:
            raise RuntimeError(f"stub providers not found: {missing}")
        for name in enabled:
            if name in disabled:
                raise RuntimeError(f"stub provider {name} is disabled")
            func = _load_entry_point(eps[name])
            if func is None:
                raise RuntimeError(f"stub provider {name} misconfigured")
            providers.append(func)
    else:
        for name, ep in eps.items():
            if name in disabled:
                continue
            func = _load_entry_point(ep)
            if func is None:
                raise RuntimeError(f"stub provider {name} misconfigured")
            providers.append(func)
    if not providers:
        raise RuntimeError("no stub providers discovered")
    return providers
