#!/usr/bin/env python3
"""Generate module grouping using the dynamic module mapper."""
from __future__ import annotations

import argparse

from dynamic_module_mapper import build_module_map


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build sandbox module map")
    parser.add_argument("repo", nargs="?", default=".", help="Repository path")
    parser.add_argument("--algorithm", default="greedy", choices=["greedy", "label"])
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--semantic", action="store_true", help="Use docstring similarity")
    opts = parser.parse_args(args)

    build_module_map(
        opts.repo,
        algorithm=opts.algorithm,
        threshold=opts.threshold,
        use_semantic=opts.semantic,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
