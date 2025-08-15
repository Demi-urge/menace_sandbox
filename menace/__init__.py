"""Compatibility package forwarding imports to project root."""

from __future__ import annotations

import os

# Allow importing modules from repository root using ``menace`` prefix.
__path__.append(os.path.dirname(os.path.dirname(__file__)))

# Default flag used by modules expecting it
RAISE_ERRORS = False

__all__ = ["RAISE_ERRORS"]
