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
from .resource_tuner import ResourceTuner
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

from pathlib import Path
import ast


def _load_discover_func():
    path = Path(__file__).with_name("sandbox_runner.py")
    try:
        src = path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "discover_orphan_modules":
                mod: dict[str, object] = {}
                ast.fix_missing_locations(node)
                code = ast.Module(body=[node], type_ignores=[])
                exec(compile(code, str(path), "exec"), mod)
                return mod.get("discover_orphan_modules")
    except Exception:
        return None
    return None


discover_orphan_modules = _load_discover_func()

__all__ = [
    "simulate_execution_environment",
    "generate_sandbox_report",
    "run_repo_section_simulations",
    "run_workflow_simulations",
    "discover_orphan_modules",
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
    "ResourceTuner",
]
