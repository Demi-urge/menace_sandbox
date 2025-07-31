#!/usr/bin/env python3
"""Backward compatible entry point for generating module maps."""

from __future__ import annotations

from pathlib import Path
import argparse

from scripts.generate_module_map import generate_module_map


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build sandbox module map")
    parser.add_argument("repo", nargs="?", default=".", help="Repository path")
    parser.add_argument("--output", default="sandbox_data/module_map.json")
    parser.add_argument("--algorithm", default="greedy", choices=["greedy", "label"])
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--semantic", action="store_true", help="Use docstring similarity")
    opts = parser.parse_args(args)

    generate_module_map(
        output=Path(opts.output),
        root=Path(opts.repo),
        algorithm=opts.algorithm,
        threshold=opts.threshold,
        semantic=opts.semantic,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
