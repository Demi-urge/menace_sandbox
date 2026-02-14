from __future__ import annotations

"""Compatibility shim for :mod:`human_alignment_flagger`."""

from human_alignment_flagger import *  # noqa: F401,F403
from human_alignment_flagger import __all__ as _public_all
from human_alignment_flagger import _collect_diff_data as _collect_diff_data

__all__ = [*_public_all, "_collect_diff_data"]
