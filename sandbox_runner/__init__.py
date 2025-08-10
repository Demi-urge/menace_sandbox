"""Support modules for sandbox_runner wrapper."""
import os
import importlib


_LIGHT_IMPORTS = bool(os.getenv("MENACE_LIGHT_IMPORTS"))

if not _LIGHT_IMPORTS:
    from .environment import (
        simulate_execution_environment,
        generate_sandbox_report,
        run_repo_section_simulations,
        run_workflow_simulations,
        auto_include_modules,
        simulate_full_environment,
        generate_input_stubs,
        SANDBOX_INPUT_STUBS,
        SANDBOX_EXTRA_METRICS,
        SANDBOX_ENV_PRESETS,
        SANDBOX_STUB_STRATEGY,
    )
else:  # defer heavy imports until needed
    _env_mod = None

    def _load_env() -> None:
        global _env_mod
        if _env_mod is None:
            _env_mod = importlib.import_module(".environment", __name__)
            for name in (
                "simulate_execution_environment",
                "generate_sandbox_report",
                "run_repo_section_simulations",
                "run_workflow_simulations",
                "auto_include_modules",
                "simulate_full_environment",
                "generate_input_stubs",
                "SANDBOX_INPUT_STUBS",
                "SANDBOX_EXTRA_METRICS",
                "SANDBOX_ENV_PRESETS",
                "SANDBOX_STUB_STRATEGY",
            ):
                globals()[name] = getattr(_env_mod, name)

    def __getattr__(name: str):  # type: ignore[override]
        if name in {
            "simulate_execution_environment",
            "generate_sandbox_report",
            "run_repo_section_simulations",
            "run_workflow_simulations",
            "auto_include_modules",
            "simulate_full_environment",
            "generate_input_stubs",
            "SANDBOX_INPUT_STUBS",
            "SANDBOX_EXTRA_METRICS",
            "SANDBOX_ENV_PRESETS",
            "SANDBOX_STUB_STRATEGY",
            "_sandbox_cycle_runner",
        }:
            _load_env()
            if name == "_sandbox_cycle_runner":
                from .cycle import _sandbox_cycle_runner as cyc
                globals()["_sandbox_cycle_runner"] = cyc
                return cyc
            return globals()[name]
        raise AttributeError(name)

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
    "auto_include_modules",
    "discover_orphan_modules",
    "discover_recursive_orphans",
    "simulate_full_environment",
    "generate_input_stubs",
    "SANDBOX_INPUT_STUBS",
    "SANDBOX_EXTRA_METRICS",
    "SANDBOX_ENV_PRESETS",
    "SANDBOX_STUB_STRATEGY",
    "_sandbox_cycle_runner",
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
