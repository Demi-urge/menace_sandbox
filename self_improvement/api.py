from __future__ import annotations

"""Public API for the self-improvement engine."""

print("🧱 SI-2a: importing init")
from .init import (
    init_self_improvement,
    settings,
    _repo_path,
    _data_dir,
    _atomic_write,
    get_default_synergy_weights,
)

print("🧱 SI-2b: importing orchestration")
from .orchestration import (
    integrate_orphans,
    post_round_orphan_scan,
    self_improvement_cycle,
    start_self_improvement_cycle,
    stop_self_improvement_cycle,
)

print("🧱 SI-2c: importing patch_application")
from .patch_application import generate_patch, apply_patch

print("🧱 SI-2d: importing roi_tracking")
from .roi_tracking import update_alignment_baseline

print("🧱 SI-2e: importing data_stores")
from .data_stores import router, STABLE_WORKFLOWS

print("🧱 SI-2f: importing engine")
from .engine import SelfImprovementEngine

print("🧱 SI-2g: importing learners")
from .learners import (
    SynergyWeightLearner,
    DQNSynergyLearner,
    DoubleDQNSynergyLearner,
    SACSynergyLearner,
    TD3SynergyLearner,
)

print("🧱 SI-2h: importing dashboards")
from .dashboards import (
    SynergyDashboard,
    load_synergy_history,
    synergy_stats,
    synergy_ma,
)

print("🧱 SI-2i: importing registry")
from .registry import ImprovementEngineRegistry, auto_x

print("🧱 SI-2j: importing orchestration_utils")
from .orchestration_utils import benchmark_workflow_variants

print("🧱 SI-2k: importing snapshot_tracker")
from .snapshot_tracker import (
    Snapshot,
    capture as capture_snapshot,
    compute_delta as delta,
    save_checkpoint,
)

print("🧱 SI-2ℹ️: self-improvement API imports completed")

__all__ = [
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
]
