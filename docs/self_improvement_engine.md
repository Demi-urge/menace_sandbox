# Self-Improvement Engine

`SelfImprovementEngine` runs a model automation pipeline whenever metrics signal the need for an update. The engine can target any bot by supplying a `bot_name` and pipeline instance.

```python
import os
from dynamic_path_router import resolve_path
from menace.self_improvement.api import SelfImprovementEngine, ImprovementEngineRegistry
from menace.model_automation_pipeline import ModelAutomationPipeline
from vector_service.context_builder import ContextBuilder

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
engine = SelfImprovementEngine(
    bot_name="alpha",
    pipeline=ModelAutomationPipeline(context_builder=builder),
    state_path=resolve_path(f"{os.getenv('SANDBOX_DATA_DIR', 'sandbox_data')}/alpha_state.json"),
)
registry = ImprovementEngineRegistry()
registry.register_engine("alpha", engine)

# Execute one cycle for all registered engines and inspect the results
results = registry.run_all_cycles(energy=2)
for name, outcome in results.items():
    print(name, outcome.roi.roi_gain)
```

Set `MENACE_ROOT` or `SANDBOX_REPO_PATH` to direct `resolve_path` at a different
checkout. For multi-root environments define `MENACE_ROOTS` or
`SANDBOX_REPO_PATHS` and use `repo_hint` to select a specific root:

```bash
MENACE_ROOTS="/repo/main:/repo/alt" python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py', repo_hint='/repo/alt'))
PY
```

When a `SelfCodingEngine` is supplied, the engine may patch helper code before
running the automation pipeline.  The helper generator retries requests using
`CODEX_RETRY_DELAYS`, simplifies prompts when failures persist and falls back to
queueing or rerouting requests to `CODEX_FALLBACK_MODEL` depending on
`CODEX_FALLBACK_STRATEGY`.  Even in this degraded mode the cycle continues.  See
[self_coding_engine.md](self_coding_engine.md) for more information.

Persisting cycle data across runs is possible by providing `state_path` when creating the engine. ROI deltas, an exponential moving average of those deltas (`roi_delta_ema`) and the timestamp of the last cycle are written to this JSON file and reloaded on startup. The smoothing factor for the EMA can be set with `roi_ema_alpha` (default `0.1`). Recent cycle outcomes are also stored in `success_history` to derive a momentum coefficient that influences future scheduling.

Each engine may use its own databases, event bus and automation pipeline allowing multiple bots to improve in parallel.

## Snapshot Tracking

`SnapshotTracker` records key metrics before and after each cycle so
regressions can be detected and improvements checkpointed.  Set
`ENABLE_SNAPSHOT_TRACKER=0` to disable the feature or adjust the
locations and thresholds via `SandboxSettings`:

* `SNAPSHOT_DIR` and `SNAPSHOT_DIFF_DIR` – where snapshots and diff
  artefacts are written (default `sandbox_data/snapshots` and
  `sandbox_data/diffs`).
* `CHECKPOINT_DIR` – base directory for saved checkpoints.
* `CHECKPOINT_RETENTION` – number of checkpoint directories to keep.
* `ROI_PENALTY_THRESHOLD` / `ENTROPY_PENALTY_THRESHOLD` – ROI drops or
  entropy gains beyond these values penalise the responsible prompt until
  a positive cycle resets the count.

## Optional Dependencies

Some features of the engine depend on optional packages. Missing modules now
cause a ``RuntimeError`` to surface configuration issues:

* `sandbox_runner` – integrates orphaned modules and performs post-round scans.
* `quick_fix_engine` – generates small helper patches.

Ensure these dependencies are installed before enabling the related
functionality.

## Usage Notes

Invoke `run_cycle()` to process a single improvement step or register the engine
with `ImprovementEngineRegistry` and call `run_all_cycles()` to iterate over all
registered bots. The method returns a mapping of engine names to
`AutomationResult` objects so callers can inspect ROI gains. Provide
`state_path` to persist ROI history between runs and set environment variables
like `SANDBOX_ENV_PRESETS` when running inside the sandbox to reuse scenario
presets.

## Workflow evolution

During a cycle the engine can refine workflow definitions through
``WorkflowEvolutionManager``. The manager benchmarks the current sequence with
``CompositeWorkflowScorer``, generates candidate variants via
``WorkflowEvolutionBot`` and records ROI deltas in ``ROIResultsDB``. The
best‑performing variant is promoted when it beats the baseline, with outcomes
logged through ``mutation_logger`` for later analysis.

Calling ``run_all_cycles()`` on an :class:`ImprovementEngineRegistry` simply
invokes each engine's ``run_cycle``, which now delegates to
``WorkflowEvolutionManager.evolve`` after the refactor. No additional wiring is
required to benefit from automatic workflow evolution and gating in batch runs.

## Algorithm Details

`SelfImprovementEngine` orchestrates a short loop during each cycle:

1. inspect recent metrics to decide whether the bot should evolve;
2. optionally apply helper patches through `SelfCodingEngine`;
3. execute the supplied `ModelAutomationPipeline` to retrain and evaluate the
   model;
4. record ROI deltas, update the exponential moving average and write state to
   disk while tracking success history for momentum-based scheduling;
5. consult `_SandboxMetaLogger` to flag modules that hit entropy ceilings or
   exhibit diminishing returns.

## Systemic foresight

Improvement cycles can consult :mod:`workflow_graph` to anticipate how changes
propagate through dependant workflows.  Calling
:meth:`workflow_graph.WorkflowGraph.simulate_impact_wave` before a cycle yields
projected ROI and synergy deltas which can steer scheduling:

```python
from workflow_graph import WorkflowGraph

graph = WorkflowGraph()

def run_with_projection(engine, target_wid: str) -> None:
    projection = graph.simulate_impact_wave(target_wid, 1.0, 0.0)
    # ...use `projection` to prioritise follow-up actions...
    engine.run_cycle()
```

## Alignment flagger integration

After applying a commit the engine invokes `HumanAlignmentFlagger` on the latest
diff. The flagger reports removed docstrings or logging, missing tests, high
complexity blocks and ethics or risk/reward issues. Every report is written to
`sandbox_data/alignment_flags.jsonl` with the commit hash and is also published
on the event bus under the `alignment:flag` topic.

Security AI or developers should tail the JSON Lines file or subscribe to the
event bus to review warnings. Alerts labelled `alignment_warning` are emitted
when scores exceed `ALIGNMENT_WARNING_THRESHOLD`, while entries above
`ALIGNMENT_FAILURE_THRESHOLD` deserve immediate attention. Set
`ENABLE_ALIGNMENT_FLAGGER=0` to disable the check or adjust
`ALIGNMENT_WARNING_THRESHOLD`, `ALIGNMENT_FAILURE_THRESHOLD` and
`ALIGNMENT_BASELINE_METRICS_PATH` to tune its behaviour.

If a metrics snapshot is provided via `ALIGNMENT_BASELINE_METRICS_PATH` the
flagger additionally checks for regressions against those baseline values. A
drop in recorded `tests` or a rise in overall `complexity` will raise
maintainability warnings even when performance indicators such as accuracy or
ROI improve. Clearing the variable disables the baseline comparison.

## Promotion gating via collapse prediction

Before a workflow's new version is promoted, the engine consults
``ForesightTracker.predict_roi_collapse``. The forecast labels trajectories as
**Stable**, **Slow decay**, **Volatile** or **Immediate collapse risk** and
adds a ``brittle`` flag when minor entropy changes cause large ROI drops.
Promotions proceed only when the returned ``risk`` resolves to ``Stable`` or
``Slow decay`` and no brittleness is detected. Workflows flagged as
``Volatile``, ``Immediate collapse risk`` or ``brittle`` remain in evaluation
until their metrics stabilise, preventing short‑lived spikes from being deployed.

```python
risk = engine.foresight_tracker.predict_roi_collapse(wf)
if risk["risk"] in {"Immediate collapse risk", "Volatile"} or risk["brittle"]:
    return  # promotion blocked
deploy(wf)
```

## Adaptive ROI Prediction

`SelfImprovementEngine` can call the `AdaptiveROIPredictor` to estimate the
ROI impact of planned actions. Train or refresh the model with the helper
CLI:

```bash
python -m menace_sandbox.adaptive_roi_cli train
python -m menace_sandbox.adaptive_roi_cli retrain
```

The accompanying databases record additional context for these forecasts.
`evaluation_history.db` includes optional `gpt_feedback` scores,
`gpt_feedback_tokens` and `long_term_delta` columns, while
`roi_events.db` stores prediction `confidence` together with full
`predicted_horizons` and `actual_horizons` arrays. These fields feed new
dataset features like long-term performance deltas and GPT feedback
metrics.

During a cycle the engine feeds feature sequences to the predictor and
receives an ROI estimate alongside a growth classification such as
"exponential" or "marginal". Meaningful forecasts require the optional
`scikit-learn` dependency and a non‑empty history; otherwise a naive
baseline is used, so predictions should be viewed as guidance rather than
hard guarantees.

### Monitoring prediction accuracy

`EvaluationDashboard` can surface detailed health metrics for
`ROITracker` predictions. After populating a tracker with predictions and
outcomes, query the panel to obtain rolling mean absolute error,
accuracy, class counts, a confusion matrix and rolling error trends:

```python
from menace.evaluation_dashboard import EvaluationDashboard
from menace.roi_tracker import ROITracker

tracker = ROITracker()
# ... run cycles that record predictions ...
dashboard = EvaluationDashboard(manager)
stats = dashboard.roi_prediction_panel(tracker)
print(
    "MAE", stats["mae"],
    "accuracy", stats["accuracy"],
    "recent MAE", stats["mae_trend"][-1],
)
```

The returned mappings are convenient for quick visualisations:

```python
import pandas as pd
pd.Series(stats["class_counts"]["actual"]).plot.bar()
pd.DataFrame(stats["confusion_matrix"]).plot.bar()
```

### Confidence weighting and demotion

`ROITracker` tracks per‑workflow mean absolute error and ROI variance and
combines them into a confidence score via ``1 / (1 + mae + variance)``. The
engine multiplies a candidate's risk‑adjusted ROI by this confidence to compute
the final score used for ranking. A threshold ``tau`` (default ``0.5``)
demotes modules whose confidence falls below the cutoff, logging a review
action instead of automatically applying patches. The evaluation dashboard
surfaces these metrics through ``workflow_mae``, ``workflow_variance`` and
``workflow_confidence`` fields.

## Scenario Types

Improvement cycles inherit the sandbox scenario presets. Use profiles such as
`high_latency_api`, `hostile_input`, `user_misuse` or `concurrency_spike` to
exercise bots under different conditions.

## Configuring and Extending Presets

Create custom presets with `environment_cli.py generate` and feed them to the
autonomous run hosting the engine via `--preset-file` or the
`SANDBOX_ENV_PRESETS` variable. Additional profiles can be introduced by
extending `environment_generator.generate_canonical_presets` or by supplying
extra JSON snippets.

## Interpreting Per-Scenario Metrics

The engine records ROI deltas and synergy metrics for each `SCENARIO_NAME` in
the tracker history. Comparing these values highlights which conditions trigger
improvements. For example, rising `roi_delta_ema` during a `concurrency_spike`
run indicates the bot is adapting to thread bursts.

## Entropy delta detection

`ROITracker` records how each patch changes `synergy_shannon_entropy` compared
to the ROI it produced. When the average ROI gain per entropy delta drops below
`ENTROPY_THRESHOLD`, or when the ratios remain below
`ENTROPY_PLATEAU_THRESHOLD` for `ENTROPY_PLATEAU_CONSECUTIVE` samples, the
engine marks the module complete to avoid infinite tweaking.

Watch for log lines like ``modules hitting entropy ceiling`` in debug output or
``sandbox diminishing`` messages in the run log. They indicate that further
changes would add complexity without meaningful returns.

Tune the sensitivity with the ``--entropy-threshold`` flag or the
``ENTROPY_THRESHOLD`` environment variable. Plateau detection is governed by
``--consecutive``/``--entropy-plateau-consecutive`` (``ENTROPY_PLATEAU_CONSECUTIVE``)
and ``ENTROPY_PLATEAU_THRESHOLD``.

## Interpreting plateau logs

During a cycle the sandbox logs a `sandbox diminishing` entry when ROI deltas or
entropy ratios flatten out for a module. These entries list the affected sections
and indicate that the engine has marked them complete, so subsequent cycles will
skip their improvement steps. Adjust `ENTROPY_PLATEAU_THRESHOLD`,
`ENTROPY_PLATEAU_CONSECUTIVE`, `ROI_THRESHOLD` or `SYNERGY_THRESHOLD` if modules
stabilise too early, or clear their history files to force reevaluation.

## Example: running with a custom preset

```bash
python environment_cli.py generate --profiles concurrency_spike --out spike.json
python run_autonomous.py --preset-file spike.json --runs 1
```

## Recursive orphan discovery

### How it works

When invoked by the autonomous sandbox the improvement engine participates in
the recursive module discovery used by `SelfTestService`. The sandbox follows
orphan and isolated modules through their import chains and
`discover_recursive_orphans` returns a mapping where each entry records its
importing `parents` and whether it is considered `redundant`. Passing files are
merged into `module_map.json`, making them available for future cycles. The
generated `.env` enables this behaviour by default via
`SANDBOX_RECURSIVE_ORPHANS=1`, `SANDBOX_RECURSIVE_ISOLATED=1` and
`SANDBOX_AUTO_INCLUDE_ISOLATED=1`. Setting `SANDBOX_AUTO_INCLUDE_ISOLATED=1` or
using `--auto-include-isolated` forces `discover_isolated_modules` to run and
implicitly sets `SANDBOX_DISCOVER_ISOLATED=1` and `SANDBOX_RECURSIVE_ISOLATED=1`
unless they are overridden. `SANDBOX_RECURSIVE_ORPHANS=1` ensures helper chains
found by the orphan walker are also traversed. Disable recursion with
`SANDBOX_RECURSIVE_ORPHANS=0` or `SANDBOX_RECURSIVE_ISOLATED=0` or by passing
the CLI flags `--no-recursive-include` or `--no-recursive-isolated`. Use
`--auto-include-isolated` (or set `SANDBOX_AUTO_INCLUDE_ISOLATED=1`) to force
isolated discovery and `--clean-orphans`/`SANDBOX_CLEAN_ORPHANS=1` to drop
passing entries from `orphan_modules.json` after integration.
All isolated modules are executed before final classification so runtime
behaviour determines whether they are redundant or ready for integration.
Modules labelled `redundant` or `legacy` are now also queued for tests. With
`SANDBOX_TEST_REDUNDANT=1` (the default) or the CLI flag
`--include-redundant`/`--test-redundant`, their import chains are followed
recursively so helper code is validated alongside active modules. Set
`SANDBOX_TEST_REDUNDANT=0` to log them without execution.

Examples:

```bash
# Include redundant modules in the run (default)
python run_autonomous.py --include-redundant

# Skip redundant modules during testing
SANDBOX_TEST_REDUNDANT=0 python run_autonomous.py --discover-orphans
```

During a cycle the sandbox first calls `include_orphan_modules` to load entries
from `sandbox_data/orphan_modules.json`. Each name is expanded by
`discover_recursive_orphans`, and the resulting set is executed by
`SelfTestService`. Passing modules are merged via
`environment.auto_include_modules`, which writes them to
`sandbox_data/module_map.json` and, when recursion is enabled through
`SANDBOX_RECURSIVE_ORPHANS` or `SANDBOX_RECURSIVE_ISOLATED`, also records any
helpers discovered along the way. While scanning, modules tagged as legacy
increment the `orphan_modules_legacy_total` gauge and it is decremented when
those modules are later reclassified or integrated.

Legacy helpers can therefore be folded back into the repository when
`SANDBOX_TEST_REDUNDANT=1` (the default). A minimal cycle might look like:

```bash
# orphan_modules.json lists a helper previously marked as legacy
$ cat sandbox_data/orphan_modules.json
{"old_helper.py": {"classification": "legacy", "parents": [], "redundant": true}}

# The helper is exercised and reintegrated
$ SANDBOX_TEST_REDUNDANT=1 python run_autonomous.py --include-orphans
...
INFO isolated module tests added=['old_helper.py'] legacy=[]
```

After a successful run the entry is pruned from the orphan cache and the
`orphan_modules_legacy_total` gauge decreases to reflect the reintegration.

### Surfacing orphan chains

`discover_recursive_orphans` loads names from
`sandbox_data/orphan_modules.json` and follows their `import` statements
recursively. Each item in the returned mapping lists its `parents`,
revealing the full chain of helpers required for a module to run.

### Validating and integrating candidates

`auto_include_modules` runs the discovered modules through
`SelfTestService` when validation is enabled. Candidates that pass are
written to `sandbox_data/module_map.json` and `try_integrate_into_workflows`
patches them into related workflows so subsequent cycles execute the new
code.

### Monitoring orphan module metrics

Progress through the orphan workflow is exposed via Prometheus gauges:

- `orphan_modules_tested_total`
- `orphan_modules_reintroduced_total`
- `orphan_modules_failed_total`
- `orphan_modules_redundant_total`
- `orphan_modules_legacy_total`

Discovery and integration of isolated modules are tracked separately:

- `isolated_modules_discovered_total`
- `isolated_modules_integrated_total`

Start the exporter with `metrics_exporter.start_metrics_server(8001)` and
configure Prometheus to scrape the port to monitor these values over time.

To run a cycle with orphan discovery enabled, use:

```bash
python run_autonomous.py --include-orphans --recursive-include
```

### Environment variables and CLI flags

- `--recursive-include` / `SANDBOX_RECURSIVE_ORPHANS` /
  `SELF_TEST_RECURSIVE_ORPHANS` – follow orphan modules and their imports.
- `--recursive-isolated` / `SANDBOX_RECURSIVE_ISOLATED` /
  `SELF_TEST_RECURSIVE_ISOLATED` – traverse dependencies of isolated modules.
- `--discover-isolated` / `SANDBOX_DISCOVER_ISOLATED` /
  `SELF_TEST_DISCOVER_ISOLATED` – run `discover_isolated_modules` during scans.
- `--auto-include-isolated` / `SANDBOX_AUTO_INCLUDE_ISOLATED` – force discovery
  of modules returned by `discover_isolated_modules`.
- `--include-redundant` / `--test-redundant` /
  `SANDBOX_TEST_REDUNDANT` / `SELF_TEST_INCLUDE_REDUNDANT` – run tests for
  modules classified as redundant or legacy.
- `--clean-orphans` / `SANDBOX_CLEAN_ORPHANS` – prune passing entries from
  `orphan_modules.json` after integration.
- `SANDBOX_SIDE_EFFECT_THRESHOLD` – side-effect score above which modules are
  tagged as `heavy_side_effects` and skipped during integration (default `10`).

### Classification and metrics storage

`discover_recursive_orphans` writes classification details to
`sandbox_data/orphan_classifications.json` in parallel with the primary orphan
cache `sandbox_data/orphan_modules.json`. The improvement engine logs test
outcomes and integration counts to `sandbox_data/metrics.db` through
`MetricsDB`, powering Prometheus gauges like `orphan_modules_reintroduced_total`.

### Redundant module handling

Modules returned by `discover_recursive_orphans` include a `redundant` flag
derived from `orphan_analyzer.analyze_redundancy`. With
`SANDBOX_TEST_REDUNDANT=1` or the CLI flag `--include-redundant`/`--test-redundant`
these modules are exercised during self tests and their dependency chains are
walked recursively. They remain excluded from `module_map.json`, but their
classification and `parents` information is captured for auditing. Set
`SANDBOX_TEST_REDUNDANT=0` to record them without execution.

Additional thresholds determine which modules `_test_orphan_modules`
returns after sandbox simulations:

- `SELF_TEST_ROI_THRESHOLD` – minimum total ROI required for a module to
  be accepted (default `0.0`).
- `SELF_TEST_SYNERGY_THRESHOLD` – minimum `synergy_roi` value for
  inclusion when multiple modules run together (default `0.0`).

The engine logs the measured ROI and synergy metrics for each candidate
and only integrates modules exceeding at least one of these thresholds.

### Side-effect capture and isolated-module execution

`orphan_analyzer` executes each candidate inside a restricted subprocess that
counts attempted file writes and network connections. The resulting
side-effect score is stored under ``side_effects`` and modules exceeding the
``SANDBOX_SIDE_EFFECT_THRESHOLD`` value are marked ``heavy_side_effects`` and
skipped during automatic workflow integration.

Enable side-effect capture during autonomous runs and combine it with
isolated-module discovery to validate helpers individually:

```bash
# Record side effects for candidate modules
SANDBOX_CAPTURE_SIDE_EFFECTS=1 python run_autonomous.py --include-orphans

# Execute isolated modules and capture their side effects
SANDBOX_CAPTURE_SIDE_EFFECTS=1 python run_autonomous.py --discover-isolated --auto-include-isolated
```

`run_autonomous.py` mirrors the `SANDBOX_*` values to the corresponding
`SELF_TEST_*` variables so the improvement engine and `SelfTestService` apply the
same recursion rules. Passing modules are appended to `module_map.json`, and
`try_integrate_into_workflows` updates existing flows so subsequent cycles
exercise the new code. Modules are indexed and auto-included using
repository-relative paths, so files named the same in different directories
do not collide:

```bash
mkdir -p pkg_a pkg_b
echo 'VALUE=1' > pkg_a/common.py
echo 'VALUE=2' > pkg_b/common.py
python -m menace.self_test_service run pkg_a/common.py pkg_b/common.py --auto-include-isolated
python run_autonomous.py --auto-include-isolated
# module_map.json records pkg_a/common.py and pkg_b/common.py separately
```
Tests like `tests/test_run_autonomous_env_vars.py`
and `tests/test_self_test_service_recursive_integration.py` assert this
recursive integration. For a concrete walkthrough see the example below.

### Example: isolated module with helper

1. **Create the isolated module and helper**:

   ```bash
   echo 'import helper\n' > isolated.py
   echo 'VALUE = 1\n'   > helper.py
   ```

2. **Run self tests to discover and validate the files**:

   ```bash
   python -m menace.self_test_service run isolated.py --auto-include-isolated
   ```

3. **Integrate during the next autonomous cycle**:

   ```bash
   SANDBOX_AUTO_INCLUDE_ISOLATED=1 python run_autonomous.py --auto-include-isolated
   ```

4. **Verify integration** – both files now appear in
   `sandbox_data/module_map.json` and can participate in future runs.

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

### Example: reintroducing a dormant module

```bash
# assume util.py was removed from module_map.json
echo '["util.py"]' > sandbox_data/orphan_modules.json
python run_autonomous.py --include-orphans
# util.py is added back and orphan_modules_reintroduced_total increases
```

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
from vector_service.context_builder import ContextBuilder


def factory(name: str) -> SelfImprovementEngine:
    builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
    return SelfImprovementEngine(
        bot_name=name,
        pipeline=ModelAutomationPipeline(context_builder=builder),
    )

registry = ImprovementEngineRegistry()
registry.register_engine("alpha", factory("alpha"))

# Adjust the number of engines according to ROI and energy
registry.autoscale(capital_bot=my_capital_bot,
                   data_bot=my_data_bot,
                   factory=factory,
                   cost_per_engine=0.05,
                   approval_callback=lambda: True)
```

Energy and ROI thresholds are derived from rolling averages maintained by the
registry.  When the current energy exceeds the mean by
``autoscale_create_dev_multiplier`` standard deviations and the ROI trend
exceeds the mean by ``autoscale_roi_dev_multiplier`` deviations, a new engine is
considered if the projected ROI covers the ``cost_per_engine``.  Deviation
multipliers are configurable via :class:`sandbox_settings.SandboxSettings` so
behaviour adapts to historical performance.  Optionally ``approval_callback``
can gate risky expansions.  If resources dwindle or ROI falls below the lower
bound, engines are removed down to ``min_engines``.

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
from menace.self_improvement.api import SelfImprovementEngine, TD3SynergyLearner

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
from menace.self_improvement.api import SynergyWeightLearner

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
