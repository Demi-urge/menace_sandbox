# Sandbox Settings

`SandboxSettings` provides configuration for sandbox runners. Settings are
loaded from environment variables, an optional `.env` file, and may be
version-controlled by supplying a YAML or JSON configuration file via
`load_sandbox_settings()`.

```python
from sandbox_settings import load_sandbox_settings
settings = load_sandbox_settings("docs/sandbox_config.sample.yaml")
```

## Meta planning defaults
- `meta_entropy_threshold`: `null`

## ROI defaults
- `threshold`: `null`
- `confidence`: `null`
- `ema_alpha`: `0.1`
- `compounding_weight`: `1.0`
- `min_integration_roi`: `0.0`
- `entropy_threshold`: `null`
- `entropy_plateau_threshold`: `null`
- `entropy_plateau_consecutive`: `null`
- `entropy_ceiling_threshold`: `null`
- `entropy_ceiling_consecutive`: `null`

## Synergy defaults
- `threshold`: `null`
- `confidence`: `null`
- `threshold_window`: `null`
- `threshold_weight`: `null`
- `ma_window`: `null`
- `stationarity_confidence`: `null`
- `std_threshold`: `null`
- `variance_confidence`: `null`
- `weight_roi`: `1.0`
- `weight_efficiency`: `1.0`
- `weight_resilience`: `1.0`
- `weight_antifragility`: `1.0`
- `weight_reliability`: `1.0`
- `weight_maintainability`: `1.0`
- `weight_throughput`: `1.0`
- `weights_lr`: `0.1`
- `train_interval`: `10`
- `replay_size`: `100`

## Alignment defaults
- `enable_flagger`: `True`
- `warning_threshold`: `0.5`
- `failure_threshold`: `0.9`
- `improvement_warning_threshold`: `0.5`
- `improvement_failure_threshold`: `0.9`
- `baseline_metrics_path`: `sandbox_metrics.yaml`

See [`sandbox_config.sample.yaml`](sandbox_config.sample.yaml) for a complete
example configuration file.

## Personal configuration examples

Override defaults via constructor arguments or environment variables when
tailoring the sandbox for personal deployments:

```python
from sandbox_settings import SandboxSettings

settings = SandboxSettings(
    sandbox_repo_path="/home/alice/menace_sandbox",
    sandbox_data_dir="/home/alice/.menace",
    synergy_train_interval=20,
)
```

```env
SANDBOX_REPO_PATH=/home/alice/menace_sandbox
SANDBOX_DATA_DIR=/home/alice/.menace
SYNERGY_TRAIN_INTERVAL=20
```

Any field listed above can be overridden in the same manner.
