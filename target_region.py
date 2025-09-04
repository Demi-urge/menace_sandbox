"""Compatibility loader for :mod:`self_improvement.target_region`.

This module dynamically loads the implementation from
``self_improvement/target_region.py`` without importing the entire
``self_improvement`` package, which has heavy optional dependencies.
It exposes ``TargetRegion`` and helper functions for legacy imports.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).with_name("self_improvement") / "target_region.py"
_SPEC = importlib.util.spec_from_file_location(
    "self_improvement.target_region", _PATH
)
module = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault(_SPEC.name, module)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(module)  # type: ignore[attr-defined]

TargetRegion = module.TargetRegion
region_from_frame = module.region_from_frame
extract_target_region = module.extract_target_region

__all__ = ["TargetRegion", "region_from_frame", "extract_target_region"]

