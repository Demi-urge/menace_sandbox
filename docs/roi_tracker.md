# ROITracker

`ROITracker` monitors return on investment deltas across self-improvement cycles.
It exposes utilities to record history, forecast the next ROI change and plot
trends.

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

## Resource-aware Forecasts

`ROITracker` can incorporate CPU, memory, disk and time usage when predicting the next delta. Pass a `ROIHistoryDB` instance to the constructor and provide the `resources` argument to `update()`. These values act as exogenous variables for ARIMA or as additional regression features.


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

`maintainability` is computed using `radon.metrics.mi_visit` and averages the
Maintainability Index of the source files touched in the current section. The
`code_quality` metric uses `pylint`'s global rating for those files.

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

