from __future__ import annotations

"""Public API for the self-improvement engine."""

from .init import (
    init_self_improvement,
    settings,
    _repo_path,
    _data_dir,
    _atomic_write,
    get_default_synergy_weights,
)
from .orchestration import (
    integrate_orphans,
    post_round_orphan_scan,
    self_improvement_cycle,
    start_self_improvement_cycle,
    stop_self_improvement_cycle,
)
from .patch_integration import generate_patch
from .telemetry import _update_alignment_baseline
from .data_stores import router, STABLE_WORKFLOWS
from .engine import (
    SelfImprovementEngine,
    SynergyWeightLearner,
    DQNSynergyLearner,
    DoubleDQNSynergyLearner,
    SACSynergyLearner,
    TD3SynergyLearner,
    SynergyDashboard,
    ImprovementEngineRegistry,
)

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
    "_update_alignment_baseline",
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
]
