#!/usr/bin/env python3
"""Generate module grouping using the dynamic module mapper."""
from __future__ import annotations

import argparse

from dynamic_module_mapper import build_module_map


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build sandbox module map")
    parser.add_argument("repo", nargs="?", default=".", help="Repository path")
    opts = parser.parse_args(args)

    build_module_map(opts.repo)


if __name__ == "__main__":  # pragma: no cover
    main()
