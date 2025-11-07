from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import List

# ---------------------------------------------------------------------------
# Expose the rich neurosales package contained in ``neurosales/neurosales``.
# Import the inner package lazily so lightweight helpers (such as the heavy
# dependency installer) remain available even when optional ML dependencies are
# absent.  This mirrors the behaviour of namespace packages where attributes
# are resolved on-demand.
__all__: List[str] = []
_CORE_MODULE: ModuleType | None = None


def _load_core() -> ModuleType:
    """Import the heavy neurosales core package on demand."""

    global _CORE_MODULE
    if _CORE_MODULE is None:
        core = import_module(".neurosales", __name__)
        _CORE_MODULE = core
        exported = getattr(core, "__all__", [])
        for name in exported:
            if name not in globals():
                globals()[name] = getattr(core, name)
            if name not in __all__:
                __all__.append(name)
    return _CORE_MODULE


def __getattr__(name: str) -> object:
    """Defer attribute resolution to the heavy neurosales package."""

    try:
        return globals()[name]
    except KeyError:
        core = _load_core()
        if hasattr(core, name):
            value = getattr(core, name)
            globals()[name] = value
            return value
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    """Expose attributes from both the lightweight facade and core package."""

    core_names = getattr(_CORE_MODULE, "__all__", []) if _CORE_MODULE else []
    return sorted({*globals().keys(), *core_names})


# Mirror the inner package's ``__all__`` to provide expected attributes.  Core
# symbols are appended lazily when first accessed so that importing
# ``neurosales.scripts.setup_heavy_deps`` does not eagerly pull in optional ML
# dependencies.

# Ensure submodule imports such as ``neurosales.mongo_memory`` resolve to the
# inner package.  Using ``__path__`` allows the package to behave like a
# namespace that spans the real implementation and the compatibility helpers
# defined below.
_package_root = Path(__file__).resolve().parent
__path__ = [
    str(_package_root),
    str(_package_root / "neurosales"),
    str(_package_root / "scripts"),
]

# ---------------------------------------------------------------------------
# Lightweight in-process conversation memory helpers used by the sandbox.
from .memory_queue import (  # noqa: E402  - imported after __path__ setup
    MessageEntry,
    MemoryQueue,
    add_message,
    get_recent_messages,
)
from .memory_stack import (  # noqa: E402
    CTAChain,
    MemoryStack,
    clear_stack,
    peek_chain,
    pop_chain,
    push_chain,
)

__all__.extend(
    [
        "MessageEntry",
        "MemoryQueue",
        "add_message",
        "get_recent_messages",
        "CTAChain",
        "MemoryStack",
        "push_chain",
        "peek_chain",
        "pop_chain",
        "clear_stack",
    ]
)
