from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Expose the rich neurosales package contained in ``neurosales/neurosales``.
# The repository stores the lightweight runtime utilities alongside an
# extensive simulated package.  Import the inner package lazily so that the
# public ``neurosales`` module mirrors the original structure while still
# remaining importable inside the sandbox.
_core = import_module(".neurosales", __name__)

# Mirror the inner package's ``__all__`` to provide expected attributes.
__all__: List[str] = list(getattr(_core, "__all__", []))
for name in __all__:
    globals()[name] = getattr(_core, name)

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
