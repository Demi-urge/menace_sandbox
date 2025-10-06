"""Support modules for sandbox_runner wrapper."""
import importlib

from logging_utils import get_logger
from sandbox_settings import SandboxSettings

settings = SandboxSettings()
_LIGHT_IMPORTS = settings.menace_light_imports

_env_mod = None
_env_simulate_temporal_trajectory = None
_cycle_runner_loader = None

_ENV_EXPORTS = (
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
    "_preset_high_latency",
    "_preset_resource_strain",
    "_preset_chaotic_failure",
)


def _load_env() -> None:
    """Import :mod:`sandbox_runner.environment` and cache key exports."""

    global _env_mod, _env_simulate_temporal_trajectory
    if _env_mod is None:
        _env_mod = importlib.import_module(".environment", __name__)
        for name in _ENV_EXPORTS:
            globals()[name] = getattr(_env_mod, name)
        _env_simulate_temporal_trajectory = getattr(
            _env_mod, "simulate_temporal_trajectory"
        )


def __getattr__(name: str):  # type: ignore[override]
    if name in _ENV_EXPORTS:
        _load_env()
        return globals()[name]
    if name == "_SandboxMetaLogger":
        try:
            from .meta_logger import _SandboxMetaLogger as ml
        except ImportError as exc:  # pragma: no cover - meta logger missing
            get_logger(__name__).warning(
                "sandbox meta logging unavailable: %s", exc
            )
            globals()[name] = None
            return None
        globals()[name] = ml
        return ml
    if name == "_sandbox_cycle_runner":
        return _load_cycle_runner()
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


def _load_cycle_runner():
    """Import and cache :func:`sandbox_runner.cycle._sandbox_cycle_runner`."""

    global _cycle_runner_loader
    if _cycle_runner_loader is None:
        from .cycle import _sandbox_cycle_runner as cyc

        _cycle_runner_loader = cyc
        globals()["_sandbox_cycle_runner"] = cyc
    return _cycle_runner_loader


if not _LIGHT_IMPORTS:
    def _sandbox_cycle_runner(*args, **kwargs):  # type: ignore[override]
        return _load_cycle_runner()(*args, **kwargs)

from .resource_tuner import ResourceTuner  # noqa: E402
from .workflow_sandbox_runner import WorkflowSandboxRunner  # noqa: E402
if _LIGHT_IMPORTS:
    run_tests = TestHarnessResult = None  # type: ignore[assignment]
else:
    from .test_harness import run_tests, TestHarnessResult  # noqa: E402
if _LIGHT_IMPORTS:
    def _run_sandbox(*_a, **_k):
        raise RuntimeError("CLI disabled in light import mode")

    def rank_scenarios(*_a, **_k):
        raise RuntimeError("CLI disabled in light import mode")

    def main(*_a, **_k):
        raise RuntimeError("CLI disabled in light import mode")

    def launch_sandbox(*_a, **_k):
        raise RuntimeError("CLI disabled in light import mode")
else:
    from .cli import _run_sandbox, rank_scenarios, main  # noqa: E402
    from .bootstrap import launch_sandbox  # noqa: E402
from .metrics_plugins import (  # noqa: E402
    discover_metrics_plugins,
    load_metrics_plugins,
    collect_plugin_metrics,
)
from .stub_providers import (  # noqa: E402
    discover_stub_providers,
    load_stub_providers,
)

from .orphan_discovery import discover_orphan_modules, discover_recursive_orphans  # noqa: E402
from .orphan_integration import (  # noqa: E402
    post_round_orphan_scan,
    integrate_and_graph_orphans,
    integrate_orphans,
)  # noqa: E402

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
    "integrate_and_graph_orphans",
    "integrate_orphans",
    "post_round_orphan_scan",
    "simulate_full_environment",
    "generate_input_stubs",
    "record_module_usage",
    "SANDBOX_INPUT_STUBS",
    "SANDBOX_EXTRA_METRICS",
    "SANDBOX_ENV_PRESETS",
    "_preset_high_latency",
    "_preset_resource_strain",
    "_preset_chaotic_failure",
    "_sandbox_cycle_runner",
    "_SandboxMetaLogger",
    "_run_sandbox",
    "rank_scenarios",
    "main",
    "launch_sandbox",
    "discover_metrics_plugins",
    "load_metrics_plugins",
    "collect_plugin_metrics",
    "discover_stub_providers",
    "load_stub_providers",
    "ResourceTuner",
    "WorkflowSandboxRunner",
    "run_tests",
    "TestHarnessResult",
]
