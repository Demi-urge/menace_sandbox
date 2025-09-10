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
- `meta_planning_interval`: `10`
- `meta_planning_period`: `3600`
- `meta_planning_loop`: `False`
- `enable_meta_planner`: `False` – fail if the optional MetaWorkflowPlanner is missing
- `meta_improvement_threshold`: `0.01`
- `meta_mutation_rate`: `1.0`
- `meta_roi_weight`: `1.0`
- `meta_domain_penalty`: `1.0`
- `meta_entropy_threshold`: `0.2`
- `overfitting_entropy_threshold`: `0.2`

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

## Delta weight defaults
- `roi_weight`: `1.0`
- `pass_rate_weight`: `1.0`
- `momentum_weight`: `1.0`
- `entropy_weight`: `0.1`
- `momentum_weight_scale`: `0.0`
- `entropy_weight_scale`: `0.0`

## Momentum defaults
- `momentum_stagnation_dev_multiplier`: `1.0` – multiplier applied to the
  momentum standard deviation when detecting stagnation.

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
- `batch_size`: `32`
- `gamma`: `0.99`

## Alignment defaults
- `enable_flagger`: `True`
- `warning_threshold`: `0.5`
- `failure_threshold`: `0.9`
- `improvement_warning_threshold`: `0.5`
- `improvement_failure_threshold`: `0.9`
- `baseline_metrics_path`: `sandbox_metrics.yaml`

## Scenario deviation defaults
- `scenario_alert_dev_multiplier`: `1.0` (`SCENARIO_ALERT_DEV_MULTIPLIER`)
- `scenario_patch_dev_multiplier`: `2.0` (`SCENARIO_PATCH_DEV_MULTIPLIER`)
- `scenario_rerun_dev_multiplier`: `3.0` (`SCENARIO_RERUN_DEV_MULTIPLIER`)

## Policy defaults
- `alpha`: `0.5`
- `gamma`: `0.9`
- `epsilon`: `0.1`
- `temperature`: `1.0`
- `exploration`: `epsilon_greedy`

## Risk analysis defaults
- `risk_weight_commit`: `0.2`
- `risk_weight_complexity`: `0.4`
- `risk_weight_failures`: `0.4`

These weights influence the risk score produced by `_analyse_module`.

## Prompt logging defaults
- `prompt_success_log_path`: `sandbox_data/prompt_success_log.jsonl` (`PROMPT_SUCCESS_LOG_PATH`)
- `prompt_failure_log_path`: `sandbox_data/prompt_failure_log.jsonl` (`PROMPT_FAILURE_LOG_PATH`)

## Prompt chunking defaults
- `prompt_chunk_token_threshold`: `3500` (`PROMPT_CHUNK_TOKEN_THRESHOLD`)
- `chunk_summary_cache_dir`: `chunk_summary_cache` (`CHUNK_SUMMARY_CACHE_DIR`,
  `PROMPT_CHUNK_CACHE_DIR` for backward compatibility)

These settings control token limits and caching for code chunk summaries. Lower
token thresholds produce smaller chunks while raising the value preserves more
context. The cache directory can be cleared at any time to force regeneration of
summaries when they become stale or disk space is tight.

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

## Optional service versions

`SandboxSettings.optional_service_versions` controls which optional services are
checked during bootstrap and the minimum versions required. The default ensures
`relevancy_radar` and `quick_fix_engine` are available, but you can override the
mapping via configuration files or the environment:

```yaml
optional_service_versions:
  relevancy_radar: "1.2.0"
  quick_fix_engine: "1.1.0"
```

```bash
export OPTIONAL_SERVICE_VERSIONS='{"relevancy_radar": "1.2.0"}'
```

Unset modules are ignored.

## Required database files

`SandboxSettings.sandbox_required_db_files` lists SQLite files that should be
present inside `sandbox_data_dir`. The bootstrap helper ensures each file
exists, creating empty databases when necessary. By default the sandbox
maintains `metrics.db` and `patch_history.db`. Override the defaults via a
configuration file or the `SANDBOX_REQUIRED_DB_FILES` environment variable:

```yaml
sandbox_required_db_files:
  - metrics.db
  - patch_history.db
```

```bash
export SANDBOX_REQUIRED_DB_FILES='["metrics.db", "custom.db"]'
```

## LLM backends

`SandboxSettings` can dynamically load different language model clients. The
`preferred_llm_backend` field selects the primary backend. Available backends
are defined by the `available_backends` mapping, which associates backend names
with dotted import paths. These entries are registered with the
`llm_registry` module so the router can instantiate them by name.

Register a custom or private adapter by extending the mapping or by using the
helpers in `llm_registry` and selecting it as the preferred backend:

```yaml
preferred_llm_backend: custom
available_backends:
  custom: "my_package.custom_client.CustomClient"
```

Environment variables accept the same configuration using JSON:

```bash
export PREFERRED_LLM_BACKEND=custom
export AVAILABLE_LLM_BACKENDS='{"custom": "my_package.custom_client.CustomClient"}'
```

The selected entry must resolve to a callable that returns an `LLMClient`
instance when invoked without arguments. Alternatively, register a factory
directly in code using `llm_registry.register_backend` or the
`@llm_registry.backend` decorator and simply reference the chosen name in
`preferred_llm_backend`.
