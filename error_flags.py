"""Shared configuration flags controlling error propagation behaviour."""

from __future__ import annotations

import os

# Allow optional exception propagation in modules that normally swallow errors.
# When the ``MENACE_RAISE_ERRORS`` environment variable is set to ``"1"`` we
# prefer surfacing exceptions to aid debugging.
RAISE_ERRORS = os.getenv("MENACE_RAISE_ERRORS") == "1"

__all__ = ["RAISE_ERRORS"]
