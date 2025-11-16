#!/usr/bin/env python3
"""Backward compatible entry point for generating module maps."""

from __future__ import annotations

from scripts.generate_module_map import main as _main


def main(args: list[str] | None = None) -> None:
    """Delegate to :func:`scripts.generate_module_map.main`."""
    _main(args)


if __name__ == "__main__":  # pragma: no cover
    main()
