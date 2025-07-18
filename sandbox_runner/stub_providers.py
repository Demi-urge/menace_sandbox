from __future__ import annotations

"""Plugin hooks for input stub generation."""

import importlib
import logging
import os
from typing import Callable, Dict, List, Sequence, Any

logger = logging.getLogger(__name__)

# Callback type for stub provider
StubProvider = Callable[[int | None, dict], List[Dict[str, Any]]]


def load_stub_providers(names: Sequence[str]) -> List[StubProvider]:
    """Import stub provider callbacks from *names*."""
    providers: List[StubProvider] = []
    for name in names:
        if not name:
            continue
        try:
            mod = importlib.import_module(name)
        except Exception:  # pragma: no cover - import errors
            logger.exception("failed to import stub provider %s", name)
            continue
        func = getattr(mod, "generate_stubs", None)
        if callable(func):
            providers.append(func)
        else:  # pragma: no cover - malformed plugin
            logger.warning("stub provider %s missing generate_stubs", name)
    return providers


def discover_stub_providers(env: dict | None = None) -> List[StubProvider]:
    """Discover stub providers using environment variables."""
    env = env or os.environ
    names = env.get("SANDBOX_STUB_PLUGINS", "")
    if not names:
        return []
    return load_stub_providers([n.strip() for n in names.split(os.pathsep) if n.strip()])
