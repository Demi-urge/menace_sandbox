from __future__ import annotations

"""Self-improvement engine public API."""

from pathlib import Path as _Path
import importlib as _importlib
import sys as _sys
import types as _types
from typing import Any as _Any


if __name__ == "self_improvement":  # pragma: no cover - runtime import aliasing
    # Allow the package to operate both as ``menace_sandbox.self_improvement`` and
    # as the legacy top-level ``self_improvement`` module.  Older entrypoints
    # import the package directly, which breaks the ``from ..`` relative imports
    # defined throughout the module tree.  Registering this alias ensures Python
    # recognises ``menace_sandbox`` as the parent package so those imports
    # resolve correctly when running in a flat layout.
    __package__ = "menace_sandbox.self_improvement"
    parent = _sys.modules.get("menace_sandbox")
    if parent is None:
        parent = _types.ModuleType("menace_sandbox")
        parent.__path__ = [str(_Path(__file__).resolve().parent.parent)]
        _sys.modules["menace_sandbox"] = parent
    _sys.modules["menace_sandbox.self_improvement"] = _sys.modules[__name__]


def _load_api_module():
    """Import :mod:`self_improvement.api` on demand."""

    module = _sys.modules.get(__name__ + ".api")
    if module is None:
        module = _importlib.import_module(".api", __name__)
    return module


_API_EXPORTS = (
    "init_self_improvement",
    "settings",
    "_repo_path",
    "_data_dir",
    "_atomic_write",
    "get_default_synergy_weights",
    "integrate_orphans",
    "post_round_orphan_scan",
    "self_improvement_cycle",
    "start_self_improvement_cycle",
    "stop_self_improvement_cycle",
    "generate_patch",
    "apply_patch",
    "update_alignment_baseline",
    "router",
    "STABLE_WORKFLOWS",
    "SelfImprovementEngine",
    "SynergyWeightLearner",
    "DQNSynergyLearner",
    "DoubleDQNSynergyLearner",
    "SACSynergyLearner",
    "TD3SynergyLearner",
    "SynergyDashboard",
    "ImprovementEngineRegistry",
    "auto_x",
    "benchmark_workflow_variants",
    "load_synergy_history",
    "synergy_stats",
    "synergy_ma",
    "Snapshot",
    "capture_snapshot",
    "delta",
    "save_checkpoint",
)

__all__ = list(_API_EXPORTS)

_STATE_SNAPSHOT_PROXY: "_LazySubmodule | None" = None


def _ensure_api_exports(name: str):
    module = _load_api_module()
    attr = getattr(module, name)
    globals()[name] = attr
    return attr


class _LazySubmodule(_types.ModuleType):
    """Lightweight proxy that loads the target module on first attribute access."""

    def __init__(self, name: str, loader):
        super().__init__(name)
        self.__dict__["_loader"] = loader

    def _load(self):
        module = self.__dict__.get("_module")
        if module is None:
            module = self.__dict__["_loader"]()
            self.__dict__["_module"] = module
            _sys.modules[self.__name__] = module
        return module

    def __getattr__(self, item: str):
        module = self._load()
        return getattr(module, item)


def _load_state_snapshot_module():
    module = _importlib.import_module(".snapshot_tracker", __name__)
    globals()["state_snapshot"] = module
    _sys.modules[__name__ + ".state_snapshot"] = module
    return module


def __getattr__(name: str) -> _Any:
    if name in _API_EXPORTS:
        return _ensure_api_exports(name)
    if name == "state_snapshot":
        return _load_state_snapshot_module()
    raise AttributeError(name)


def __dir__() -> list[str]:  # pragma: no cover - simple helper
    return sorted(set(list(globals()) + list(_API_EXPORTS) + ["state_snapshot"]))


if __name__ + ".state_snapshot" not in _sys.modules:
    _STATE_SNAPSHOT_PROXY = _LazySubmodule(
        __name__ + ".state_snapshot", _load_state_snapshot_module
    )
    _sys.modules[__name__ + ".state_snapshot"] = _STATE_SNAPSHOT_PROXY
