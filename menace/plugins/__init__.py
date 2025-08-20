"""Plugin loading for the Menace CLI."""
from __future__ import annotations

import sys
from importlib import metadata
from typing import Any


def load_plugins(subparsers: Any) -> None:
    """Load CLI plugins defined in the ``menace.plugins`` entry-point group.

    Each entry point should provide a callable that accepts an
    :class:`argparse._SubParsersAction` instance and registers one or more
    subcommands on it.
    """
    try:  # Python >=3.10
        eps = metadata.entry_points(group="menace.plugins")
    except TypeError:  # pragma: no cover - for older Python versions
        eps = metadata.entry_points().get("menace.plugins", [])  # type: ignore[arg-type]

    for ep in eps:
        try:
            register = ep.load()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to load plugin {ep.name}: {exc}", file=sys.stderr)
            continue
        try:
            register(subparsers)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to register plugin {ep.name}: {exc}", file=sys.stderr)
