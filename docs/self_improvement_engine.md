# Self-Improvement Engine

`SelfImprovementEngine` runs a model automation pipeline whenever metrics signal the need for an update. The engine can target any bot by supplying a `bot_name` and pipeline instance.

```python
from menace.self_improvement_engine import SelfImprovementEngine, ImprovementEngineRegistry
from menace.model_automation_pipeline import ModelAutomationPipeline

engine = SelfImprovementEngine(bot_name="alpha",
                               pipeline=ModelAutomationPipeline())
registry = ImprovementEngineRegistry()
registry.register_engine("alpha", engine)

# Execute one cycle for all registered engines
registry.run_all_cycles()
```

When a `SelfCodingEngine` is supplied, the engine may patch helper code before running the automation pipeline. See [self_coding_engine.md](self_coding_engine.md) for more information.

Persisting cycle data across runs is possible by providing `state_path` when creating the engine. ROI deltas, an exponential moving average of those deltas (`roi_delta_ema`) and the timestamp of the last cycle are written to this JSON file and reloaded on startup. The smoothing factor for the EMA can be set with `roi_ema_alpha` (default `0.1`).

Each engine may use its own databases, event bus and automation pipeline allowing multiple bots to improve in parallel.

## Recursive inclusion flow

When invoked by the autonomous sandbox the improvement engine participates in
the recursive module discovery used by `SelfTestService`. The sandbox follows
orphan and isolated modules through their import chains and
`discover_recursive_orphans` returns a mapping where each entry records its
importing `parents` and whether it is considered `redundant`. Passing files are
merged into `module_map.json`, making them available for future cycles. The generated
`.env` enables this behaviour by default via `SANDBOX_RECURSIVE_ORPHANS=1`,
`SANDBOX_RECURSIVE_ISOLATED=1` and `SANDBOX_AUTO_INCLUDE_ISOLATED=1`. Disable
recursion with `SANDBOX_RECURSIVE_ORPHANS=0` or `SANDBOX_RECURSIVE_ISOLATED=0`
or by passing the CLI flags `--no-recursive-include` or `--no-recursive-isolated`.
Use `--auto-include-isolated` (or set `SANDBOX_AUTO_INCLUDE_ISOLATED=1`) to force
isolated discovery and `--clean-orphans`/`SANDBOX_CLEAN_ORPHANS=1` to drop
passing entries from `orphan_modules.json` after integration.

### Environment variables controlling recursion

- `SANDBOX_RECURSIVE_ORPHANS` / `SELF_TEST_RECURSIVE_ORPHANS` – follow orphan
  modules and their imports.
- `SANDBOX_RECURSIVE_ISOLATED` / `SELF_TEST_RECURSIVE_ISOLATED` – traverse
  dependencies of isolated modules.
- `SANDBOX_AUTO_INCLUDE_ISOLATED` – force discovery of modules returned by
  `discover_isolated_modules`.
- `SANDBOX_CLEAN_ORPHANS` – prune passing entries from
  `orphan_modules.json` after integration.

### Redundant module handling

Modules returned by `discover_recursive_orphans` include a `redundant` flag
derived from `orphan_analyzer.analyze_redundancy`. These modules are skipped
automatically during test runs so the classification and `parents` information
is captured, but they are never merged into `module_map.json`.

Additional thresholds determine which modules `_test_orphan_modules`
returns after sandbox simulations:

- `SELF_TEST_ROI_THRESHOLD` – minimum total ROI required for a module to
  be accepted (default `0.0`).
- `SELF_TEST_SYNERGY_THRESHOLD` – minimum `synergy_roi` value for
  inclusion when multiple modules run together (default `0.0`).

The engine logs the measured ROI and synergy metrics for each candidate
and only integrates modules exceeding at least one of these thresholds.

`run_autonomous.py` mirrors the `SANDBOX_*` values to the corresponding
`SELF_TEST_*` variables so the improvement engine and `SelfTestService` apply the
same recursion rules. Passing modules are appended to `module_map.json`, and
`try_integrate_into_workflows` updates existing flows so subsequent cycles
exercise the new code. Module identifiers are repository-relative paths to
avoid filename collisions. Tests like `tests/test_run_autonomous_env_vars.py`
and `tests/test_self_test_service_recursive_integration.py` assert this
recursive integration. For a concrete walkthrough see the
[isolated module example](autonomous_sandbox.md#example-isolated-module-discovery-and-integration).

### Example: recursive module inclusion with cleanup

```bash
# create an orphan and helper module
echo 'import helper\n' > orphan.py
echo 'VALUE = 1\n'   > helper.py

# run the self tests recursively and clean passing entries
python -m menace.self_test_service run --recursive-include --clean-orphans

# integrate automatically during the next autonomous cycle
python run_autonomous.py --include-orphans --recursive-include --clean-orphans
```

  Passing files are merged into `module_map.json` and, with `--clean-orphans`
  or `SANDBOX_CLEAN_ORPHANS=1`, removed from `sandbox_data/orphan_modules.json`.
Include `--auto-include-isolated --recursive-isolated` to apply the same flow
to modules that are not referenced anywhere else.

## Reinforcement-learning policy

`SelfImprovementEngine` can optionally be initialised with a `SelfImprovementPolicy`.
The policy learns from previous evolution cycles via Q‑learning and predicts
whether running another cycle is likely to increase ROI. `_should_trigger()`
consults this policy in addition to diagnostic checks and the predicted value
also scales the `energy` passed to the automation pipeline.

In addition to policy predictions, the engine tracks rolling averages of ROI
deltas. A short term mean and an exponential moving average influence the
energy allocated to each cycle. Positive averages boost the energy budget while
negative values reduce it.

Recent sandbox runs record ROI deltas per module via ``sandbox_runner._SandboxMetaLogger``.
Passing the logger to `SelfImprovementEngine` exposes `rankings()` and
`diminishing()` so module trends influence the policy state. Module names are
hashed and persisted to keep the Q‑learning state space bounded.

## Automatic scaling

`ImprovementEngineRegistry.autoscale()` can create or remove engines
automatically based on available resources and the long‑term ROI trend.
Provide it with a `CapitalManagementBot`, a `DataBot` and a factory function
that returns a new `SelfImprovementEngine` for a given name:

```python
def factory(name: str) -> SelfImprovementEngine:
    return SelfImprovementEngine(bot_name=name,
                                 pipeline=ModelAutomationPipeline())

registry = ImprovementEngineRegistry()
registry.register_engine("alpha", factory("alpha"))

# Adjust the number of engines according to ROI and energy
registry.autoscale(capital_bot=my_capital_bot,
                   data_bot=my_data_bot,
                   factory=factory,
                   cost_per_engine=0.05,
                   approval_callback=lambda: True)
```

When the energy score exceeds ``create_energy`` (default ``0.8``) and the ROI
trend is above ``roi_threshold`` the registry ensures projected ROI covers the
``cost_per_engine`` before adding a new one. Optionally ``approval_callback``
can gate risky expansions. If resources dwindle or ROI falls below the
threshold, engines are removed down to ``min_engines``.

## Synergy Weight Learners

The engine keeps a set of synergy weights that modulate how cross‑module
metrics influence policy updates. ``SynergyWeightLearner`` stores seven weights
(``roi``, ``efficiency``, ``resilience``, ``antifragility``, ``reliability``,
``maintainability`` and ``throughput``) in a JSON file. After each cycle
``_update_synergy_weights`` measures the rolling change of the corresponding
synergy metrics and calls ``SynergyWeightLearner.update``:

``synergy_roi`` → ``roi``

``synergy_efficiency`` → ``efficiency``

``synergy_resilience`` → ``resilience``

``synergy_antifragility`` → ``antifragility``

The base learner uses a lightweight actor‑critic policy so weights follow a
learned reinforcement‑learning strategy. ``DQNSynergyLearner`` provides a deeper
Double DQN alternative when PyTorch is available.  ``SACSynergyLearner`` and
``TD3SynergyLearner`` wrap simplified Soft Actor‑Critic and TD3 strategies so
different RL approaches can be selected via ``synergy_learner_cls``.

``DQNSynergyLearner`` extends this process with a small deep Q‑network. When
PyTorch is available the learner defaults to a Double DQN variant with a target
network that periodically syncs from the online model. The synergy deltas form
the state and the ROI‑scaled change acts as the reward for each action.
Predicted Q‑values replace the manual gradient step so weight updates follow the
learned policy. Both the online and target model weights are persisted alongside
the policy file so progress carries over between runs without a cold start.

Both learners emit a warning during initialisation if optional dependencies are
missing. Install ``torch`` for the best performance when using the DQN variant.

## Modifying ``synergy_weights.json``

Synergy weights are persisted in a JSON file. By default the engine stores it at
``sandbox_data/synergy_weights.json``. Provide ``synergy_weights_path`` when
creating ``SelfImprovementEngine`` to choose a different location. The file can
be edited manually or with ``synergy_weight_cli.py`` to set starting values.

The environment variables ``SYNERGY_WEIGHT_ROI``,
``SYNERGY_WEIGHT_EFFICIENCY``, ``SYNERGY_WEIGHT_RESILIENCE`` and
``SYNERGY_WEIGHT_ANTIFRAGILITY`` override the loaded weights at startup. After
each cycle the learner writes back to the JSON file so adjustments persist
between runs.

Example using ``TD3SynergyLearner`` and a custom weight file:

```python
from menace.self_improvement_engine import SelfImprovementEngine, TD3SynergyLearner

engine = SelfImprovementEngine(
    synergy_learner_cls=TD3SynergyLearner,
    synergy_weights_path="synergy_weights.json",
)
engine.run_cycle()
```

## Synergy Predictions and Preset Adaptation

`ROITracker` records synergy metrics whenever multiple modules run together in
the sandbox. `predict_synergy()` forecasts the next `synergy_roi` value while
`predict_synergy_metric()` predicts specific metrics such as
`synergy_efficiency` or `synergy_resilience`. `environment_generator.adapt_presets`
uses these forecasts to adjust resource limits before each run.

```python
from menace.roi_tracker import ROITracker
from menace.environment_generator import adapt_presets

tracker = ROITracker()
tracker.update(0.0, 0.1, metrics={"synergy_efficiency": 0.02})

# Combine recent measurements with the predicted next value
pred_eff = tracker.predict_synergy_metric("efficiency")
presets = [{"CPU_LIMIT": "1", "MEMORY_LIMIT": "256"}]
new_presets = adapt_presets(tracker, presets)
```

When `pred_eff` is positive, CPU and memory limits are lowered because modules
are expected to cooperate efficiently. Negative values have the opposite effect
and increase the limits.

## Synergy Metrics Reference

The sandbox stores `synergy_<name>` entries capturing how the combined run of
multiple modules differs from the average of their individual runs:

- **synergy_roi** – overall ROI delta caused by module interaction.
- **synergy_efficiency** – higher values suggest lower CPU and memory usage when
  modules cooperate.
- **synergy_adaptability** – indicates how well modules adjust to changing
  scenarios; positive values reduce resource limits.
- **synergy_antifragility** – measures benefits from stress. High scores raise
  `THREAT_INTENSITY` while negative ones lower it.
- **synergy_resilience** – reflects tolerance to failures. Positive values
  increase bandwidth limits, negative ones decrease them.
- **synergy_safety_rating** – combined safety performance influencing
  `THREAT_INTENSITY`.
- **synergy_risk_index** – security risk trend; high values bump the
  `SECURITY_LEVEL` preset.
- **synergy_security_score** – cross‑module security score impact.
- **synergy_recovery_time** – how quickly modules recover from errors when run
  together.
- **synergy_shannon_entropy** – diversity of behaviours in the combined run.
- **synergy_flexibility** – ability to handle varied inputs without failures.
- **synergy_energy_consumption** – additional or reduced energy usage.
- **synergy_profitability** – combined profit impact.
- **synergy_revenue** – revenue change compared to solo runs.
- **synergy_projected_lucrativity** – long‑term earning potential of the group.
- **synergy_maintainability** – effect on code maintainability metrics.
- **synergy_code_quality** – aggregated code quality change.
- **synergy_network_latency** – latency difference when modules interact.
- **synergy_throughput** – throughput change for the combined workload.
- **synergy_discrepancy_count** – variation in result consistency across runs.
- **synergy_gpu_usage**, **synergy_cpu_usage**, **synergy_memory_usage** –
  resource usage deltas for GPU, CPU and memory respectively.
- **synergy_long_term_lucrativity** – ROI trend over extended periods.

`ROITracker.reliability()` converts prediction errors into a value between `0`
and `1`. A score near `1` means synergy forecasts closely matched reality while
values near `0` indicate noisy predictions. `synergy_reliability()` is a helper
for the `synergy_roi` series and is used by the sandbox to scale ROI tolerance
during long runs.

## Interpreting synergy metrics

Synergy metrics highlight how modules influence each other when run together. A
positive `synergy_efficiency` means the group consumes fewer resources than its
parts individually while a negative `synergy_roi` shows the interaction reduces
profit.

```python
from menace.roi_tracker import ROITracker
tracker = ROITracker()
forecast = tracker.predict_synergy_metric("efficiency")
if forecast > 0:
    print("Expect lower CPU usage", forecast)
```

## Adjusting synergy weights

Weights stored in `synergy_weights.json` control how much each metric affects the
reinforcement learner. They can be tweaked programmatically:

```python
from menace.self_improvement_engine import SynergyWeightLearner

learner = SynergyWeightLearner(path="synergy_weights.json")
learner.weights["efficiency"] *= 1.5  # favour efficient interactions
learner.save()
```

Alternatively export the file, edit it and import the new values:

```bash
python synergy_weight_cli.py --path synergy_weights.json export --out weights.json
# edit weights.json
python synergy_weight_cli.py --path synergy_weights.json import weights.json
```
