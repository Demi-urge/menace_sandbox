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
from .patch_application import generate_patch
from .roi_tracking import update_alignment_baseline
from .data_stores import router, STABLE_WORKFLOWS
from .engine import (
    SelfImprovementEngine,
    SACSynergyLearner,
    TD3SynergyLearner,
)
from .learners import (
    SynergyWeightLearner,
    DQNSynergyLearner,
    DoubleDQNSynergyLearner,
)
from .dashboards import (
    SynergyDashboard,
    load_synergy_history,
    synergy_stats,
    synergy_ma,
)
from .registry import ImprovementEngineRegistry, auto_x
from .orchestration_utils import benchmark_workflow_variants

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
]
