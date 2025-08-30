# Sandbox Self-Improvement

This guide outlines the packages, environment variables and common workflows needed
to run self-improvement cycles inside the sandbox.

## Required packages

- Install core dependencies with `./setup_env.sh` or `pip install -r requirements.txt`.
- `sandbox_runner` and `quick_fix_engine` provide the execution harness and patch
generation.

## Optional packages

- `pandas`, `psutil` and `prometheus-client` enable dashboards and resource
  metrics.
- `torch` supplies deep reinforcement learning modules.

## Environment variables

- `SANDBOX_REPO_PATH` – path to the local sandbox repository clone.
- `SANDBOX_DATA_DIR` – directory for metrics and state files.
- `SANDBOX_ENV_PRESETS` – comma separated scenario preset files.
- `AUTO_TRAIN_INTERVAL`, `SYNERGY_TRAIN_INTERVAL`,
  `ADAPTIVE_ROI_RETRAIN_INTERVAL` – control retraining frequency.
- `ENABLE_META_PLANNER` – require meta-planning support when set to `true`.

## Example workflows

### Run a single cycle from the CLI

```bash
SANDBOX_REPO_PATH=$(pwd) python sandbox_runner.py --runs 1
```

### Trigger a cycle programmatically

```python
from self_improvement import SelfImprovementEngine
from model_automation_pipeline import ModelAutomationPipeline

engine = SelfImprovementEngine(bot_name="alpha",
                               pipeline=ModelAutomationPipeline())
engine.run_cycle()
```

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
