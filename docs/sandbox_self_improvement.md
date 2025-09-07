# Sandbox Self-Improvement

This guide outlines the packages, environment variables and common workflows needed
to run self-improvement cycles inside the sandbox.

## Required packages

- Install core dependencies with `./setup_env.sh` or `pip install -r requirements.txt`.
- `sandbox_runner` (including the ``sandbox_runner.orphan_integration`` module)
  and `quick_fix_engine` provide the execution harness and patch generation.
  These packages are **not** installed automatically; if they are missing the
  self-improvement utilities raise a ``RuntimeError`` with guidance to install
  them manually (e.g. ``pip install sandbox_runner quick_fix_engine``).

## Optional packages

- `pandas`, `psutil` and `prometheus-client` enable dashboards and resource
  metrics.
- `torch` supplies deep reinforcement learning modules.

## Environment variables

- `SANDBOX_REPO_PATH` – path to the local sandbox repository clone.
- `SANDBOX_DATA_DIR` – directory for metrics and state files.
- `SANDBOX_ENV_PRESETS` – comma separated scenario preset files.
- `SANDBOX_STUB_SEED` – seed value for deterministic stub generation.
- `AUTO_TRAIN_INTERVAL`, `SYNERGY_TRAIN_INTERVAL`,
  `ADAPTIVE_ROI_RETRAIN_INTERVAL` – control retraining frequency.
- `ENABLE_META_PLANNER` – require meta-planning support when set to `true`.
- `BASELINE_WINDOW` – number of recent scores used for the moving average baseline.
- `STAGNATION_ITERS` – cycles with no improvement before the baseline resets.
- `DELTA_MARGIN` – minimum positive delta over baseline needed to accept a patch.

## Example workflows

### Run a single cycle from the CLI

```bash
SANDBOX_REPO_PATH=$(pwd) python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)" --runs 1
```

### Trigger a cycle programmatically

```python
from self_improvement.api import SelfImprovementEngine
from model_automation_pipeline import ModelAutomationPipeline
from vector_service import ContextBuilder

engine = SelfImprovementEngine(bot_name="alpha",
                               pipeline=ModelAutomationPipeline(context_builder=ContextBuilder()))
engine.run_cycle()
```

## Adaptive strategy

Recent cycle scores are compared against a moving baseline. The window size is
configurable via `BASELINE_WINDOW`. When ROI deltas fail to beat this baseline
for `STAGNATION_ITERS` iterations or fall short by more than `DELTA_MARGIN`,
the engine escalates its urgency tier to explore more aggressive mutations.

## Troubleshooting

### Missing dependencies

- `ModuleNotFoundError: sandbox_runner` or `quick_fix_engine` – run
  `./setup_env.sh` or install the packages manually.
- `RuntimeError: ffmpeg not found` – install `ffmpeg` and `tesseract` and
  ensure they are on `PATH`.

### Common runtime errors

- Stale containers or overlays: `python -m sandbox_runner.cli --cleanup`.
- Planner required but unavailable: set `ENABLE_META_PLANNER=0` to disable
  meta-planning.

For configuration details see
[`sandbox_self_improvement_configuration.md`](sandbox_self_improvement_configuration.md).
