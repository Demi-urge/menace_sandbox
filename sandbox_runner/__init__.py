"""Support modules for sandbox_runner wrapper."""
import os
import importlib


_LIGHT_IMPORTS = bool(os.getenv("MENACE_LIGHT_IMPORTS"))
_env_simulate_temporal_trajectory = None

if not _LIGHT_IMPORTS:
    from .environment import (
        simulate_execution_environment,
        generate_sandbox_report,
        run_repo_section_simulations,
        run_workflow_simulations,
        run_scenarios,
        simulate_temporal_trajectory as _env_simulate_temporal_trajectory,
        temporal_trajectory_presets,
        auto_include_modules,
        discover_and_integrate_orphans,
        integrate_new_orphans,
        simulate_full_environment,
        generate_input_stubs,
        record_module_usage,
        SANDBOX_INPUT_STUBS,
        SANDBOX_EXTRA_METRICS,
        SANDBOX_ENV_PRESETS,
        SANDBOX_STUB_STRATEGY,
        _preset_high_latency,
        _preset_resource_strain,
        _preset_chaotic_failure,
    )
    from .meta_logger import _SandboxMetaLogger
else:  # defer heavy imports until needed
    _env_mod = None

    def _load_env() -> None:
        global _env_mod, _env_simulate_temporal_trajectory
        if _env_mod is None:
            _env_mod = importlib.import_module(".environment", __name__)
            for name in (
                "simulate_execution_environment",
                "generate_sandbox_report",
                "run_repo_section_simulations",
                "run_workflow_simulations",
                "run_scenarios",
                "temporal_trajectory_presets",
                "auto_include_modules",
                "discover_and_integrate_orphans",
                "integrate_new_orphans",
                "simulate_full_environment",
                "generate_input_stubs",
                "record_module_usage",
                "SANDBOX_INPUT_STUBS",
                "SANDBOX_EXTRA_METRICS",
                "SANDBOX_ENV_PRESETS",
                "SANDBOX_STUB_STRATEGY",
                "_preset_high_latency",
                "_preset_resource_strain",
                "_preset_chaotic_failure",
            ):
                globals()[name] = getattr(_env_mod, name)
            _env_simulate_temporal_trajectory = getattr(
                _env_mod, "simulate_temporal_trajectory"
            )

    def __getattr__(name: str):  # type: ignore[override]
        if name in {
            "simulate_execution_environment",
            "generate_sandbox_report",
            "run_repo_section_simulations",
            "run_workflow_simulations",
            "run_scenarios",
            "temporal_trajectory_presets",
            "auto_include_modules",
            "discover_and_integrate_orphans",
            "integrate_new_orphans",
            "simulate_full_environment",
            "generate_input_stubs",
            "record_module_usage",
            "SANDBOX_INPUT_STUBS",
            "SANDBOX_EXTRA_METRICS",
            "SANDBOX_ENV_PRESETS",
            "SANDBOX_STUB_STRATEGY",
            "_preset_high_latency",
            "_preset_resource_strain",
            "_preset_chaotic_failure",
            "_sandbox_cycle_runner",
            "_SandboxMetaLogger",
        }:
            if name == "_SandboxMetaLogger":
                from .meta_logger import _SandboxMetaLogger as ml
                globals()["_SandboxMetaLogger"] = ml
                return ml
            _load_env()
            if name == "_sandbox_cycle_runner":
                from .cycle import _sandbox_cycle_runner as cyc
                globals()["_sandbox_cycle_runner"] = cyc
                return cyc
            return globals()[name]
        raise AttributeError(name)


def simulate_temporal_trajectory(
    workflow_id, workflow, tracker=None, foresight_tracker=None
):
    if _env_simulate_temporal_trajectory is None:
        loader = globals().get("_load_env")
        if loader:
            loader()
    return _env_simulate_temporal_trajectory(
        workflow_id,
        workflow,
        tracker=tracker,
        foresight_tracker=foresight_tracker,
    )


if not _LIGHT_IMPORTS:
    from .cycle import _sandbox_cycle_runner

from .resource_tuner import ResourceTuner
if _LIGHT_IMPORTS:
    def _run_sandbox(*_a, **_k):
        raise RuntimeError("CLI disabled in light import mode")

    def rank_scenarios(*_a, **_k):
        raise RuntimeError("CLI disabled in light import mode")

    def main(*_a, **_k):
        raise RuntimeError("CLI disabled in light import mode")
else:
    from .cli import _run_sandbox, rank_scenarios, main
from .metrics_plugins import (
    discover_metrics_plugins,
    load_metrics_plugins,
    collect_plugin_metrics,
)
from .stub_providers import (
    discover_stub_providers,
    load_stub_providers,
)

from .orphan_discovery import discover_orphan_modules, discover_recursive_orphans

__all__ = [
    "simulate_execution_environment",
    "generate_sandbox_report",
    "run_repo_section_simulations",
    "run_workflow_simulations",
    "run_scenarios",  # export scenario runner for external callers
    "simulate_temporal_trajectory",
    "temporal_trajectory_presets",
    "auto_include_modules",
    "discover_and_integrate_orphans",
    "integrate_new_orphans",
    "discover_orphan_modules",
    "discover_recursive_orphans",
    "simulate_full_environment",
    "generate_input_stubs",
    "record_module_usage",
    "SANDBOX_INPUT_STUBS",
    "SANDBOX_EXTRA_METRICS",
    "SANDBOX_ENV_PRESETS",
    "SANDBOX_STUB_STRATEGY",
    "_preset_high_latency",
    "_preset_resource_strain",
    "_preset_chaotic_failure",
    "_sandbox_cycle_runner",
    "_SandboxMetaLogger",
    "_run_sandbox",
    "rank_scenarios",
    "main",
    "discover_metrics_plugins",
    "load_metrics_plugins",
    "collect_plugin_metrics",
    "discover_stub_providers",
    "load_stub_providers",
    "ResourceTuner",
]
