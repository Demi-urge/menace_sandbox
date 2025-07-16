"""Support modules for sandbox_runner wrapper."""
from .environment import (
    simulate_execution_environment,
    generate_sandbox_report,
    run_repo_section_simulations,
    run_workflow_simulations,
    simulate_full_environment,
    generate_input_stubs,
    SANDBOX_INPUT_STUBS,
    SANDBOX_EXTRA_METRICS,
    SANDBOX_ENV_PRESETS,
)
from .cycle import _sandbox_cycle_runner
from .cli import _run_sandbox, rank_scenarios, main
from .metrics_plugins import (
    discover_metrics_plugins,
    load_metrics_plugins,
    collect_plugin_metrics,
)

__all__ = [
    "simulate_execution_environment",
    "generate_sandbox_report",
    "run_repo_section_simulations",
    "run_workflow_simulations",
    "simulate_full_environment",
    "generate_input_stubs",
    "SANDBOX_INPUT_STUBS",
    "SANDBOX_EXTRA_METRICS",
    "SANDBOX_ENV_PRESETS",
    "_sandbox_cycle_runner",
    "_run_sandbox",
    "rank_scenarios",
    "main",
    "discover_metrics_plugins",
    "load_metrics_plugins",
    "collect_plugin_metrics",
]
