from __future__ import annotations

"""Compatibility shim for :mod:`human_alignment_flagger`."""

from human_alignment_flagger import *  # noqa: F401,F403
from human_alignment_flagger import __all__ as _public_all
from human_alignment_flagger import _collect_diff_data

# Keep compatibility with the upstream public API while exporting this
# semiprivate helper explicitly for internal callers.
__all__ = list(dict.fromkeys([*_public_all, "_collect_diff_data"]))
