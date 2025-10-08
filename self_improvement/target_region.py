"""Compatibility wrapper for :mod:`target_region`.

This module re-exports the canonical :class:`TargetRegion` utilities so existing
imports from ``self_improvement.target_region`` continue to function even when
executed outside the package context.
"""

try:
    from menace_sandbox.target_region import TargetRegion, region_from_frame, extract_target_region
except Exception:  # pragma: no cover - fallback for direct execution
    import importlib.util
    import sys
    from dynamic_path_router import resolve_path

    path = resolve_path("target_region.py")
    spec = importlib.util.spec_from_file_location(
        "_target_region_fallback",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_target_region_fallback"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    TargetRegion = module.TargetRegion  # type: ignore
    region_from_frame = module.region_from_frame  # type: ignore
    extract_target_region = module.extract_target_region  # type: ignore


__all__ = ["TargetRegion", "region_from_frame", "extract_target_region"]

