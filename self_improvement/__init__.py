from __future__ import annotations

"""Self-improvement engine public API."""

from .api import *  # noqa: F401,F403
from .api import __all__  # noqa: F401

# Backwards compatibility for the deprecated `state_snapshot` module
from . import snapshot_tracker as state_snapshot  # noqa: F401
import sys as _sys
_sys.modules[__name__ + ".state_snapshot"] = state_snapshot
