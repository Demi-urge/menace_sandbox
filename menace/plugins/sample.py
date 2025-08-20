"""Sample Menace CLI plugin."""
from __future__ import annotations

import argparse


def _handle(args: argparse.Namespace) -> int:
    print("hello from plugin")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the sample plugin subcommand."""
    parser = subparsers.add_parser("hello", help="Sample plugin greeting")
    parser.set_defaults(func=_handle)
