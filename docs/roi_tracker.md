# ROITracker

`ROITracker` monitors return on investment deltas across self-improvement cycles.
It exposes utilities to record history, forecast the next ROI change and plot
trends.

For end-to-end workflow evaluation and persistent metric storage see the
[CompositeWorkflowScorer](composite_workflow_scorer.md).

For profile based scoring see the [ROI calculator](roi_calculator.md).

## Adaptive Forecasting

The `forecast()` method evaluates multiple ARIMA orders when the optional
`statsmodels` package is available. The model with the lowest AIC/BIC is chosen
and the order is cached until the history changes, avoiding repeated parameter
searches. When ARIMA is unavailable the tracker falls back to a simple linear
regression forecast.

## Weighted Averages

`ROITracker` can apply custom weights to recent ROI deltas when deciding
whether to stop further iterations. Pass the ``weights`` argument to
`ROITracker()` with a list matching the ``window`` size. Values are normalised
and applied to the most recent deltas, giving greater influence to later
entries when desired.

## Scenario scorecards

`generate_scenario_scorecard(workflow_id, scenarios=None)` assembles a
JSON‑serialisable summary of stress test results.  For each scenario preset
(``concurrency_spike``, ``hostile_input`` and others from
`sandbox_runner.environment`) the scorecard records:

* ``roi_delta`` – ROI difference between workflow on/off runs
* ``metrics_delta`` – per‑metric changes against the baseline
* ``synergy_delta`` – differences in synergy metrics

Use the CLI to emit or persist a scorecard:

```bash
python -m menace_sandbox.adaptive_roi_cli scorecard WF_ID \
    --history sandbox_data/roi_history.json \
    --output sandbox_data/scorecard.json
```

When ``--output`` is omitted the scorecard prints to STDOUT.  Providing no
explicit scenarios defaults to the standard presets.

## Retrieval Source ROI

`ROITracker.update()` accepts a `retrieval_metrics` list describing
retrieval outcomes from `UniversalRetriever`. Each entry should include an
`origin_db` label, a `hit` flag, and optional `tokens` count. When ROI deltas
are recorded with these metrics, `roi_by_origin_db()` returns the average ROI
contribution per database. `export_origin_db_roi_csv(path)` writes a CSV report
and `retrieval_bias()` exposes weights that ranking modules can use to favour
high-ROI databases.

## Resource-aware Forecasts

`ROITracker` can incorporate CPU, memory, disk and time usage when predicting the next delta. Pass a `ROIHistoryDB` instance to the constructor and provide the `resources` argument to `update()`. These values act as exogenous variables for ARIMA or as additional regression features.

## Risk-adjusted ROI

`calculate_raroi()` derives a risk-adjusted score that penalises unstable or unsafe workflows. It returns a `(base_roi, raroi, bottlenecks)` tuple so the raw ROI remains visible alongside the adjusted value and low scores provide remediation hints:

```python
from menace_sandbox.roi_tracker import ROITracker

tracker = ROITracker()
tracker.roi_history = [0.2, 0.15, 0.18]
base_roi, raroi, bottlenecks = tracker.calculate_raroi(
    1.2,
    workflow_type="standard",
    metrics={"errors_per_minute": 0.1},
    failing_tests={"security_suite": False},
)
print(base_roi, raroi, bottlenecks)  # raw ROI, risk-adjusted ROI and any bottlenecks
```

### Formula components

```
raroi = base_roi * (1 - catastrophic_risk) * stability_factor * safety_factor
```

* **catastrophic_risk** – product of the estimated `rollback_probability` and the impact severity resolved for the workflow type.
* **stability_factor** – `1 - instability` where `instability` is the standard deviation of recent ROI deltas.
* **safety_factor** – derived by `_safety_factor` from provided metrics and
  penalised when critical suites fail.

### Impact severity and test metrics

Impact severity levels live in `config/impact_severity.yaml` and can be overridden via the `IMPACT_SEVERITY_CONFIG` environment variable. `metrics` may supply values such as `errors_per_minute`, `instability` or a pre-computed `rollback_probability`. Failing test names or boolean mappings can be passed via `failing_tests` so critical suites further reduce the score. In the autonomous sandbox these inputs come from each module's `collect_metrics` hook and self-test reports.

`_safety_factor` expects a metrics mapping that may include scores like
`safety_rating`, `security_score`, `synergy_safety_rating` and
`synergy_security_score`. Penalties are applied for failure counters such as
`hostile_failures`, `misuse_failures` or per-suite entries following the
pattern `<suite>_failures` (for example `security_failures` or
`alignment_failures`).

Higher RAROI values promote a workflow in ranking while lower scores push it
down. Inside the autonomous sandbox the collected `metrics` and
`failing_tests` are passed to `calculate_raroi`, and the resulting score is
combined with raw ROI to rank modules and decide whether the self-improvement
engine should continue iterating or deprioritise a workflow. See the
[RAROI overview](raroi.md) for additional background.

## Borderline workflows

Workflows whose RAROI drops below ``raroi_borderline_threshold`` or whose
confidence score falls under ``confidence_threshold`` are routed to a
[borderline bucket](borderline_bucket.md) rather than being thrown away. The tracker
adds the workflow as a pending candidate and the lightweight bucket records its
RAROI history and latest confidence. Calling ``borderline_bucket.process`` (or
the convenience ``process_borderline_candidates`` wrapper) runs a micro‑pilot
evaluation for each pending candidate. Results above the thresholds trigger
``promote()`` while lower scores call ``terminate()`` so borderline workflows are
either adopted or discarded based on the micro‑pilot outcome. In the autonomous
sandbox these thresholds are configured via ``BORDERLINE_RAROI_THRESHOLD`` and
``BORDERLINE_CONFIDENCE_THRESHOLD``.

Set ``BORDERLINE_RAROI_THRESHOLD`` (and optionally
``BORDERLINE_CONFIDENCE_THRESHOLD``) to control when a workflow is queued and
``MICROPILOT_MODE`` to decide how candidates are handled:

* ``auto`` – immediately run a micro‑pilot when a workflow enters the bucket.
* ``queue`` – only enqueue candidates; call ``process_borderline_candidates``
  later.
* ``off`` – disable the bucket entirely.

Example CLI usage that enables automatic micro‑pilots:

```bash
MICROPILOT_MODE=auto BORDERLINE_RAROI_THRESHOLD=0.1 \
python run_autonomous.py --runs 1
```

## Entropy delta tracking

The constructor also accepts ``entropy_threshold`` which sets the minimum ROI
gain per unit entropy delta before further increases are considered
unproductive. When omitted it falls back to ``tolerance`` to preserve existing
behaviour.

Each prediction records the change in `synergy_shannon_entropy` relative to the complexity delta of the last patch. The ratio for each module is appended to `module_entropy_deltas`. `entropy_delta_history(name)` returns the stored ratios while `entropy_plateau(threshold, consecutive)` identifies modules whose entropy change stays below a threshold for a given number of samples. Inside the autonomous sandbox these flags are controlled by the `ENTROPY_PLATEAU_THRESHOLD` and `ENTROPY_PLATEAU_CONSECUTIVE` variables and cause modules to be marked complete and skipped in later cycles.

Use ``--entropy-threshold``/``ENTROPY_THRESHOLD`` to set the minimum ROI gain
per entropy unit and ``--consecutive``/``ENTROPY_PLATEAU_CONSECUTIVE`` to define
how many low-entropy samples are required before a module is considered
complete.

```python
tracker.record_prediction(0.1, 0.1)
print(tracker.entropy_delta_history("m.py"))
print(tracker.entropy_plateau(0.01, 3))
```

## Adaptive ROI Prediction

Longer term ROI trends can be estimated with the lightweight
`AdaptiveROIPredictor` model.

### Building the dataset

Training requires a feature matrix describing past improvement cycles.
`adaptive_roi_dataset.build_dataset()` assembles this matrix from the
evolution, ROI and evaluation databases:

```python
from menace_sandbox.adaptive_roi_dataset import build_dataset
X, y, growth = build_dataset(
    "evolution_history.db", "roi.db", "evaluation_history.db"
)
```

`load_training_data()` offers a higher level helper that exports the
dataset to CSV. The CLI can refresh this file periodically without
retraining the model:

```bash
python -m menace_sandbox.adaptive_roi_cli refresh --once --output-csv roi_dataset.csv
```

### Training the predictor

The `adaptive_roi_cli` module exposes simple subcommands and can be
executed via ``python -m menace_sandbox.adaptive_roi_cli``:

```bash
python -m menace_sandbox.adaptive_roi_cli train
python -m menace_sandbox.adaptive_roi_cli predict "[[0.1,0.2,0.0,0.0,0.0,0.5]]" --horizon 3
python -m menace_sandbox.adaptive_roi_cli retrain
python -m menace_sandbox.adaptive_roi_cli schedule --once
python -m menace_sandbox.adaptive_roi_cli refresh --once
```

`train` fits a new model on available history, `predict` returns an ROI
forecast sequence and growth type for a JSON encoded feature matrix
(`--horizon` controls the number of steps) and `retrain` updates an
existing model with the latest data. The `schedule` command calls
`load_training_data()` to assemble the latest dataset and retrains the
model at a fixed interval (default one hour). Pass `--interval` to
adjust the cadence or `--once` to run a single cycle, which is useful
for cron jobs. `refresh` performs the dataset rebuild step on a timer
without retraining the model:

```cron
0 * * * * python -m menace_sandbox.adaptive_roi_cli schedule --history sandbox_data/roi_history.json
```

Training stores the most influential input metrics in a companion
``.meta.json`` file under ``selected_features`` whenever the model
exposes ``feature_importances_``. Running `adaptive_roi_cli` with the
``--selected-features`` flag restricts `build_dataset()` to these columns
so you can periodically retrain on a pruned feature set and drop
low-importance metrics from future runs.

The feature matrix produced by `build_dataset()` now also includes
`gpt_feedback_score`, `gpt_feedback_tokens`, `long_term_perf_delta`,
`prediction_confidence`, `predicted_horizon_delta` and
`actual_horizon_delta` columns for richer training signals.

The predictor uses `scikit-learn` when installed and falls back to a naive
baseline if no regression backend or dataset is available, so results
should be treated as coarse guidance rather than exact forecasts.

`ROITracker.evaluate_model()` monitors recent prediction accuracy and,
when error exceeds thresholds, automatically spawns the `schedule` command
to refresh the model in the background.


### CLI workflow

The CLI is designed for quick experimentation and can be driven directly
from the project root. Common flows look like:

```bash
# train a fresh model
python -m menace_sandbox.adaptive_roi_cli train

# run a one-off forecast for a feature matrix
python -m menace_sandbox.adaptive_roi_cli predict "[[0.2,0.4,0.0,0.1]]" --horizon 2

# refresh the existing model with new data
python -m menace_sandbox.adaptive_roi_cli retrain

# periodically rebuild the dataset and retrain
python -m menace_sandbox.adaptive_roi_cli schedule --interval 3600
```

`train` fits an initial predictor, `retrain` performs an incremental
update and `predict` returns both the ROI sequence and growth category
for the supplied features. The `schedule` subcommand is useful for cron
jobs; pass `--once` for a single run or adjust `--interval` to control
the cadence.

### Action planning integration

Predictions can steer the sandbox's decision making. `ActionPlanner`
scales priority weights by the forecasted ROI and a multiplier derived
from the growth classification:

```python
from menace_sandbox.action_planner import ActionPlanner
from menace_sandbox.neuroplasticity import PathwayDB
from menace_sandbox.resource_allocation_optimizer import ROIDB
from vector_service.context_builder import ContextBuilder

planner = ActionPlanner(PathwayDB(), ROIDB(), ContextBuilder(),
                        feature_fn=lambda action: [0.1, 0.2])
planner.update_priorities({"launch_campaign": 1.0, "cleanup": 0.8})
print(planner.get_priority_queue())  # highest ROI and growth first
```

`update_priorities` queries `AdaptiveROIPredictor` for each action and
multiplies the base weight by the predicted ROI and a growth multiplier.
Actions forecast to deliver exponential growth therefore rise to the top
of the queue.

### Self-improvement module integration

Self-improvement loops can consume these predictions directly.
`SelfImprovementEngine` initialises an `AdaptiveROIPredictor` and
`ROITracker` when adaptive ROI prioritisation is enabled. During each
cycle it queries the predictor for the expected ROI and growth class of
candidate actions:

```python
from menace_sandbox.self_improvement import SelfImprovementEngine

engine = SelfImprovementEngine(bot_name="alpha")
roi_seq, growth, *_ = engine.roi_predictor.predict([[0.1, 0.2, 0.0]])
if growth == "exponential":
    print("scale up the improvement plan")
```

Predictions and realised outcomes are recorded in
`engine.roi_tracker`, enabling later retraining with
``adaptive_roi_cli schedule``.

### Growth classification, interpretation and threshold tuning

The predictor labels horizons as ``exponential``, ``linear`` or
``marginal`` based on slope and curvature thresholds.

```python
from menace_sandbox.adaptive_roi_predictor import AdaptiveROIPredictor

pred = AdaptiveROIPredictor()
_, growth, *_ = pred.predict([[0.05, 0.1, 0.0]])
print(growth)  # "linear"
```

- ``exponential`` – ROI accelerates each cycle; consider aggressive scaling.
- ``linear`` – steady improvement; keep the current strategy.
- ``marginal`` – little change; deprioritise the action.

Treat these classes as broad guidance rather than exact guarantees. Start
with the defaults estimated by `calibrate_thresholds()` and adjust using
the `--slope-threshold` and `--curvature-threshold` flags when calling the
CLI. Raising the thresholds yields more conservative classifications while
lowering them makes the model more sensitive to small changes. Monitor
`ROITracker.evaluate_model()` to ensure the thresholds produce accurate
forecasts for your dataset.


## Usage Example
```python
from menace.roi_tracker import ROITracker

tracker = ROITracker()
for i in range(5):
    tracker.update(0.0, float(i))
next_roi, (lo, hi) = tracker.forecast()
```

## Persistence

`save_history()` writes both ROI deltas, per‑module contributions and
prediction accuracy data. JSON files now include `roi_history`, `module_deltas`,
`predicted_roi` and `actual_roi` keys while SQLite stores an additional
`predictions` table. `load_history()` remains backwards compatible with the
previous plain list JSON format.

## Metric Forecasting

Pass a dictionary via `metrics` to `update()` to record arbitrary series such as
loss or security scores. `forecast_metric(name)` predicts the next value for a
metric and `record_metric_prediction()` allows tracking accuracy with
`rolling_mae_metric()`.

`PredictionManager` registers helper bots like `FutureLucrativityBot` and
`FutureProfitabilityBot` when initialised with a `DataBot`. These bots average
recent metric values and provide simple forecasts automatically used by
`ROITracker.predict_all_metrics()` each cycle. Calling this method iterates over
all names in ``metrics_history`` and appends the returned prediction to
``predicted_metrics`` while recording the latest observed value in
``actual_metrics``. The lists are persisted by ``save_history()`` so forecast
accuracy can be evaluated later.


## Reliability Metrics

`record_prediction()` records predicted and actual ROI values. Use
`record_metric_prediction(metric, predicted, actual)` for arbitrary metrics.
`rolling_mae()` and `rolling_mae_metric()` compute a rolling mean absolute error
for ROI and metric forecasts respectively. `reliability()` converts these errors
into a value between `0` and `1` where `1` denotes perfect predictions.
The CLI exposes this via the `reliability` command:

```bash
python -m menace.roi_tracker reliability history.json --window 10
python -m menace.roi_tracker reliability history.json --metric profit
```

### Per-workflow metrics

`record_prediction()` optionally accepts a `workflow_id`. Predictions and
outcomes are stored per workflow in `workflow_predicted_roi` and
`workflow_actual_roi` using a rolling window controlled by the
``workflow_window`` constructor argument. Call
`workflow_mae(workflow_id)` or `workflow_variance(workflow_id)` to inspect the
latest mean absolute error and ROI variance for a given workflow. These values
combine into a per‑workflow confidence score using ``1 / (1 + mae + variance)``
which is clipped to ``[0, 1]`` and stored in ``workflow_confidence_history``.
`SelfImprovementEngine` multiplies risk‑adjusted ROI by this confidence and
defers modules whose confidence falls below a threshold ``tau`` (default
``0.5``). The evaluation dashboard exposes ``workflow_mae``,
``workflow_variance`` and ``workflow_confidence`` fields for visualising these
trends.

### ROI prediction chart

``EvaluationDashboard.roi_prediction_chart()`` provides sequences suitable for
plotting predicted versus actual ROI values. The returned ``labels`` correspond
to the index of each prediction in the tracker history so windowed slices retain
their original offsets:

```python
from menace.evaluation_dashboard import EvaluationDashboard
from menace.roi_tracker import ROITracker

dash = EvaluationDashboard(manager)
tracker = ROITracker()
chart = dash.roi_prediction_chart(tracker)
print(chart["labels"], chart["predicted"], chart["actual"])
```

### ROI prediction events panel

The evaluation dashboard exposes a helper for inspecting persisted
``roi_prediction_events``. ``EvaluationDashboard.roi_prediction_events_panel()``
returns the latest mean absolute error for each forecast horizon, accuracy of
growth class predictions, growth type accuracy and drift metrics recorded by the
adaptive predictor, and recent drift detection flags:

```python
from menace.evaluation_dashboard import EvaluationDashboard
from menace.roi_tracker import ROITracker

dash = EvaluationDashboard(manager)
tracker = ROITracker()
panel = dash.roi_prediction_events_panel(tracker)
print(
    panel["mae_by_horizon"],
    panel["growth_class_accuracy"],
    panel["growth_type_accuracy"],
    panel["drift_metrics"],
    panel["drift_flags"],
)
```

`predict_metric_with_manager(manager, name, features, actual=None)` queries
prediction bots registered with a `PredictionManager` and automatically stores
the prediction via `record_metric_prediction()`. Use the CLI command
`predict-metric` to manually request such a forecast.
## Prediction Integration Example

```python
from menace.roi_tracker import ROITracker
from menace.prediction_manager_bot import PredictionManager

tracker = ROITracker()
manager = PredictionManager()
tracker.update(1.0, 1.1, metrics={"profit": 0.5})
pred = tracker.predict_metric_with_manager(manager, "profit", [0.5], actual=0.45)
print("prediction", pred)
print("MAE", tracker.rolling_mae_metric("profit"))
```
After several updates you can inspect forecast accuracy:
```python
error = tracker.rolling_mae()
metric_error = tracker.rolling_mae_metric("profit")
score = tracker.reliability()
```


## Extended Metrics
Sandbox runs record additional metrics in the tracker:
- `security_score`
- `safety_rating`
- `adaptability`
- `antifragility`
- `shannon_entropy`
- `efficiency`
- `flexibility`
- `projected_lucrativity`
- `profitability`
- `patch_complexity`
- `energy_consumption`
- `resilience`
- `network_latency`
- `throughput`
- `risk_index`
- `maintainability`
- `code_quality`
- `recovery_time`
- `orphan_modules_tested`
- `orphan_modules_passed`
- `orphan_modules_failed`
- `orphan_modules_redundant`

Forecasts for these values help prioritise improvements when ROI alone is stable.

`maintainability` is computed using `radon.metrics.mi_visit` when available and
averages the Maintainability Index of the source files touched in the current
section. When `radon` is missing an AST-based implementation of the standard
Maintainability Index formula is used instead. The `code_quality` metric uses
`pylint`'s global rating for those files.

Example forecasting extended metrics:
```python
from menace.roi_tracker import ROITracker
from menace.prediction_manager_bot import PredictionManager

manager = PredictionManager(data_bot=data_bot)
tracker = ROITracker()
tracker.update(
    0.0,
    0.0,
    metrics={"flexibility": 0.3, "antifragility": 0.7, "shannon_entropy": 0.8},
)
preds = tracker.predict_all_metrics(manager)
print(preds["flexibility"], preds["antifragility"], preds["shannon_entropy"])
```

## Synergy Metric Forecasting

When a sandbox session finishes it runs all modified sections together.
The differences between this combined run and the average of the individual
section runs are stored under `synergy_roi` and `synergy_<metric>` keys. Values
above zero mean the modules work better together while negative values indicate
interference.

Recent updates introduced additional synergy metrics capturing how entropy,
flexibility and energy consumption change when modules interact:

- `synergy_shannon_entropy`
- `synergy_flexibility`
- `synergy_energy_consumption`
- `synergy_profitability`
- `synergy_revenue`
- `synergy_efficiency`
- `synergy_antifragility`
- `synergy_resilience`
- `synergy_projected_lucrativity`
- `synergy_adaptability`
- `synergy_safety_rating`
- `synergy_security_score`
- `synergy_maintainability`
- `synergy_code_quality`
- `synergy_network_latency`
- `synergy_throughput`
- `synergy_risk_index`
- `synergy_recovery_time`

`ROITracker.predict_synergy()` predicts the next `synergy_roi` entry and
`predict_synergy_metric(name)` forecasts any other synergy metric. The function
accepts the plain metric name so both `"security_score"` and
`"synergy_security_score"` are valid arguments.
`predict_synergy_profitability()`, `predict_synergy_revenue()` and
`predict_synergy_projected_lucrativity()` are thin wrappers around
`predict_synergy_metric()` for these common metrics.
Additional helpers like `predict_synergy_maintainability()`,
`predict_synergy_adaptability()`, `predict_synergy_code_quality()`,
`predict_synergy_safety_rating()` and `predict_synergy_risk_index()` call the
same method with the respective metric names for convenience.

Example usage:

```python
from menace.roi_tracker import ROITracker

tracker = ROITracker()
tracker.update(
    0.0,
    0.1,
    metrics={
        "security_score": 0.8,
        "shannon_entropy": 0.5,
        "synergy_security_score": 0.02,
        "synergy_shannon_entropy": -0.05,
        "synergy_roi": 0.01,
    },
)

sec_delta = tracker.predict_synergy_metric("security_score")
roi_delta = tracker.predict_synergy()
rel = tracker.reliability(metric="synergy_security_score")
print(sec_delta, roi_delta, rel)
```

Synergy predictions help decide whether combining sections is beneficial and how
large the effect may be during the next iteration. The reliability value ranges
from `0` to `1` and indicates how well previous synergy forecasts matched the
actual results.

Example computing synergy reliability:

```python
tracker.record_metric_prediction("synergy_roi", 0.04, 0.02)
tracker.record_metric_prediction("synergy_roi", 0.05, 0.03)
print(tracker.synergy_reliability())
```

## Custom Metrics

`sandbox_runner.py` reads `sandbox_metrics.yaml` (or JSON) from the repository
root. The file can contain an `extra_metrics` mapping or a list of metric names.
Entries are merged into the metrics dictionary supplied to `ROITracker.update`
each cycle.

```yaml
extra_metrics:
  code_quality: 0.0
  maintainability: 0.0
  stale_containers_removed: 0.0
  stale_vms_removed: 0.0
  runtime_vms_removed: 0.0
  cleanup_failures: 0.0
  force_kills: 0.0
```

The environment itself tracks operational counters. `collect_metrics` exposes
cleanup statistics such as `cleanup_disk` or `cleanup_lifetime`. The additional
fields `stale_containers_removed`, `stale_vms_removed` and
`runtime_vms_removed` report how many stale Docker containers or VM overlays were
purged by `purge_leftovers` or the background cleanup worker. The last metric
only counts files removed while the application is running. `cleanup_failures`
counts failed attempts to stop or delete containers while `force_kills`
increments whenever the CLI fallback forcibly removes a lingering container.

## Prediction-enabled Metrics Plugins

Plugins can also call `PredictionManager` to compute forecasts for custom
metrics. The sandbox runner exposes the loaded `PredictionManager` instance so
plugins may store it in a module level variable on load. Predictions returned
from `collect_metrics` are recorded like any other metric value.

Example plugin collecting a CPU delta and predicting the next value:

```python
# plugins/cpu_predict.py
from menace.prediction_manager_bot import PredictionManager

manager: PredictionManager | None = None
tracker: ROITracker | None = None

def register(pm: PredictionManager, tr: ROITracker | None = None) -> None:
    global manager, tracker
    manager = pm
    tracker = tr

def collect_metrics(prev_roi: float, roi: float, resources: dict | None) -> dict:
    cpu = resources.get("cpu", 0.0) if resources else 0.0
    pred = 0.0
    if manager:
        # predict the next cpu delta using registered bots
        pred = manager.registry[next(iter(manager.registry))].bot.predict_metric(
            "cpu_delta", [cpu]
        )
        if tracker:
            tracker.record_metric_prediction("cpu_delta", pred, cpu - 50.0)
    return {"cpu_delta": cpu - 50.0, "cpu_delta_pred": pred}
```

Register the plugin manager inside your sandbox run:

```python
manager = PredictionManager(data_bot=data_bot)
plugins = load_metrics_plugins("plugins")
for plugin in plugins:
    if hasattr(plugin, "register"):
        plugin.register(manager, tracker)
```

During each cycle the predicted value is written alongside the actual metric and
stored by `ROITracker`.

### Synergy Prediction Plugin

The repository also ships with `plugins/synergy_predict.py` which uses
`PredictionManager` to forecast `synergy_roi` and any other
`synergy_<metric>` entries present in the tracker. Predictions are automatically
recorded via `ROITracker.record_metric_prediction` so their accuracy can be
analysed later.

Register the plugin the same way as other prediction-enabled plugins:

```python
manager = PredictionManager(data_bot=data_bot)
plugins = load_metrics_plugins("plugins")
for plugin in plugins:
    if hasattr(plugin, "register"):
        plugin.register(manager, tracker)
```

## Metrics Dashboard

Launch a small Flask dashboard to visualise ROI trends and forecast accuracy:

```bash
python -m menace.metrics_dashboard --file roi_history.json --port 8002
```

Open the printed address in your browser. Visit `/plots/predictions.png` to see
a PNG with predicted and actual series for ROI and all recorded metrics. Lines
that closely overlap indicate accurate forecasts while large gaps highlight poor
predictions.

## Forecast Accuracy

Visualise predictions with the metrics dashboard or print rolling errors from
the command line:

```bash
python -m menace.metrics_dashboard --file roi_history.json --port 8002
python -m menace.roi_tracker reliability roi_history.json --metric security_score
```

## Command Line Interface

The module exposes a small CLI for quick inspection:

```bash
python -m menace.roi_tracker forecast history.json
# Predicted ROI: 6.0 (CI 5.8 - 6.2)

python -m menace.roi_tracker rank history.json
# b.py 1.0
# a.py 0.5

python -m menace.roi_tracker reliability history.json
# MAE: 0.75
python -m menace.roi_tracker reliability history.json --metric profit
# MAE: 0.10

python -m menace.roi_tracker predict-metric history.json projected_lucrativity
# Predicted projected_lucrativity: 1.23
```

