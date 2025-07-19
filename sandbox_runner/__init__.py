"""Support modules for sandbox_runner wrapper."""
import os
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
if os.getenv("MENACE_LIGHT_IMPORTS"):
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
    "discover_stub_providers",
    "load_stub_providers",
]
