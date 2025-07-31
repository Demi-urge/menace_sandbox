#!/usr/bin/env python3
"""Generate module grouping using the module graph analyzer."""
from __future__ import annotations

import argparse

from pathlib import Path

from scripts.generate_module_map import generate_module_map


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build sandbox module map")
    parser.add_argument("repo", nargs="?", default=".", help="Repository path")
    parser.add_argument("--algorithm", default="greedy", choices=["greedy", "label"])
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--semantic", action="store_true", help="Use docstring similarity")
    parser.add_argument("--output", default="sandbox_data/module_map.json", help="Output file")
    opts = parser.parse_args(args)

    generate_module_map(
        Path(opts.output),
        root=Path(opts.repo),
        algorithm=opts.algorithm,
        threshold=opts.threshold,
        semantic=opts.semantic,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
