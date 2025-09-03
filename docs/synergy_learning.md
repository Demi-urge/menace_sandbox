# Synergy learning

This page explains how synergy weights are learned and how predictions feed into ROI calculations. Synergy history is persisted in `synergy_history.db` under the sandbox data directory. Resolve the path dynamically to respect environment overrides:

```python
import os
from dynamic_path_router import resolve_path
history = resolve_path(f"{os.getenv('SANDBOX_DATA_DIR', 'sandbox_data')}/synergy_history.db")
```

If a legacy `synergy_history.json` exists it is migrated automatically at startup.

## SynergyWeightLearner

`SynergyWeightLearner` maintains seven weights controlling how synergy metrics influence policy updates:

``roi``
``efficiency``
``resilience``
``antifragility``
``reliability``
``maintainability``
``throughput``

The learner stores these values in `synergy_weights.json`. When the file is missing the initial weights are pulled from the metrics snapshot referenced by `ALIGNMENT_BASELINE_METRICS_PATH` or, if that is unavailable, from the active `SandboxSettings` configuration. A hard-coded set of unit weights is only used as a last resort. Each cycle `SelfImprovementEngine._update_synergy_weights` computes the latest deltas for `synergy_<name>` metrics and calls `SynergyWeightLearner.update`. The default implementation uses an actor–critic policy to nudge weights toward positive ROI changes. A `DQNSynergyLearner` subclass provides a deeper Double DQN variant when PyTorch is available. `DoubleDQNSynergyLearner` and `TD3SynergyLearner` expose alternative strategies when you set `SYNERGY_LEARNER` to `double_dqn` or `td3`.

Weights can be edited manually or with `synergy_weight_cli.py`. Environment variables such as `SYNERGY_WEIGHT_ROI` or `SYNERGY_WEIGHT_EFFICIENCY` override the loaded values at startup. The learning rate is controlled by `SYNERGY_WEIGHTS_LR`. To switch reinforcement-learning strategy set `SYNERGY_LEARNER` to `double_dqn` or `td3` before running the CLI.

## ARIMASynergyPredictor

`ARIMASynergyPredictor` fits an ARIMA model to a synergy metric series and predicts the next value. `ROITracker.predict_synergy` and `predict_synergy_metric` consult this predictor when the environment variable `SANDBOX_SYNERGY_MODEL=arima` is set and enough history is available. Otherwise a simpler exponential moving average is used.

## ROI feedback loop

During ROI calculation `SelfImprovementEngine` adds the weighted `synergy_roi` delta to the profit figure and adjusts the energy score using `synergy_efficiency`, `synergy_resilience` and `synergy_antifragility`. This means synergy metrics directly impact ROI as the weights change. Improving prediction accuracy through models such as `ARIMASynergyPredictor` therefore helps the engine react earlier to beneficial or harmful interactions.

## Tuning examples

Personal deployments can start with custom weights and learning rates:

```dotenv
# .env
SYNERGY_WEIGHT_ROI=1.2
SYNERGY_WEIGHT_EFFICIENCY=0.8
SYNERGY_WEIGHT_RESILIENCE=1.0
SYNERGY_WEIGHTS_LR=0.05
```

Create a minimal `synergy_weights.json`:

```json
{"roi": 1.2, "efficiency": 0.8, "resilience": 1.0,
 "antifragility": 1.0, "reliability": 1.0,
 "maintainability": 1.0, "throughput": 1.0}
```

Then run:

```bash
python synergy_weight_cli.py --path synergy_weights.json show
```

After a few sessions record synergy history and train the weights:

```bash
python synergy_weight_cli.py --path synergy_weights.json train sandbox_data/synergy_history.db
```

The updated file persists between runs and influences ROI calculations automatically.

## Customising weights with `synergy_weight_cli.py`

The helper CLI directly edits `synergy_weights.json` and records each change in
`sandbox_data/synergy_weights.log`. Typical steps are:

1. **Show the current weights** to verify the starting values:

   ```bash
   python synergy_weight_cli.py --path synergy_weights.json show
   ```

2. **Export** them for manual tweaking:

   ```bash
   python synergy_weight_cli.py --path synergy_weights.json export --out weights.json
   ```

   Edit `weights.json` and adjust each value between `0.0` and `10.0`.

3. **Import** the edited file:

   ```bash
   python synergy_weight_cli.py --path synergy_weights.json import weights.json
   ```

4. **Train** from recorded history whenever you have new synergy metrics:

   ```bash
   python synergy_weight_cli.py --path synergy_weights.json train sandbox_data/synergy_history.db
   ```

5. **Reset** to defaults if experimentation goes wrong:

   ```bash
   python synergy_weight_cli.py --path synergy_weights.json reset
   ```

6. **Review the change log** or produce a plot of all updates:

   ```bash
   python synergy_weight_cli.py history --log sandbox_data/synergy_weights.log --plot
   ```

These commands modify the file in place so subsequent sandbox runs pick up the
new weights automatically.

## Visualising weight history

`sandbox_dashboard.py` reads `sandbox_data/synergy_weights.log` and exposes the
recorded values at `/weights`. When you open the dashboard root page it shows a
Chart.js graph of how each weight has changed over time. Use this view to track
training progress alongside ROI metrics.

## Weight update walkthrough

1. Each iteration the ROI tracker stores the latest synergy metrics in
   `synergy_history.db`:

   ```json
   [
     {"synergy_roi": 0.01, "synergy_efficiency": 0.02},
     {"synergy_roi": 0.03, "synergy_efficiency": 0.05}
   ]
   ```

2. `SelfImprovementEngine._update_synergy_weights` computes the delta for each
   metric and invokes `SynergyWeightLearner.update(roi_delta, deltas)`. The
   learner adjusts the values stored in `synergy_weights.json`:

   ```json
   {"roi": 1.05, "efficiency": 0.95, "resilience": 1.0,
    "antifragility": 1.0, "reliability": 1.0,
    "maintainability": 1.0, "throughput": 1.0}
   ```

3. During the next cycle `_weighted_synergy_adjustment` multiplies the current
   metric deltas by these weights. Positive ROI deltas increase the
   corresponding weights while negative values reduce them, causing the system
   to favour metrics that historically improved ROI.

## Interpreting SynergyExporter metrics

`SynergyExporter` reads the same `synergy_history.db` file and exposes the
most recent entry as Prometheus gauge metrics. Each key becomes a metric of the
same name, for example `synergy_roi`, `synergy_efficiency` and
`synergy_resilience`. Higher values indicate improvements relative to the
previous cycle while negative values show a decline.

Enable the exporter with `EXPORT_SYNERGY_METRICS=1` or by calling
`start_synergy_exporter()`:

```python
from menace.synergy_exporter import start_synergy_exporter

start_synergy_exporter(history_file="sandbox_data/synergy_history.db", port=8003)
```

The module also provides a small command line interface:

```bash
python -m menace.synergy_exporter --history-file sandbox_data/synergy_history.db
```

The exporter listens on port `8003` by default. Pass `--port` to change it and
use `--interval` to adjust the polling frequency:

```bash
python -m menace.synergy_exporter --history-file sandbox_data/synergy_history.db --port 8003
```

`run_autonomous.py` starts the exporter automatically when
`EXPORT_SYNERGY_METRICS=1`. Invoke the command above on its own when you only
need to expose the metrics or debug outside the sandbox loop.

A minimal Prometheus configuration to scrape the exporter looks like this:

```yaml
scrape_configs:
  - job_name: 'synergy'
    static_configs:
      - targets: ['localhost:8003']
```

Collected metrics appear as plain values:

```
synergy_roi 0.05
synergy_efficiency 0.01
```

Use these metrics together with the current weights to understand how synergy
factors contribute to overall ROI.

## Background training and exporter monitoring

Set `AUTO_TRAIN_SYNERGY=1` to start a background trainer that periodically updates
`synergy_weights.json` from the data stored in `synergy_history.db`. When this
environment variable is enabled, `SelfImprovementEngine` launches the task in
the background. The update interval defaults to 600 seconds and can be adjusted
with `AUTO_TRAIN_INTERVAL`.

When `EXPORT_SYNERGY_METRICS=1` is enabled the exporter is monitored and
automatically restarted if its health check fails. Control how frequently the
monitor checks the exporter with `SYNERGY_EXPORTER_CHECK_INTERVAL`.

Failed training attempts increment the `synergy_trainer_failures_total` gauge.
The monitor exposes two additional gauges reflecting automatic restarts:
`synergy_exporter_restarts_total` and `synergy_trainer_restarts_total`. Set
`SANDBOX_CENTRAL_LOGGING=1` to forward exporter and trainer logs to the audit
trail specified by `AUDIT_LOG_PATH` (or Kafka when `KAFKA_HOSTS` is defined).
If either restart counter grows beyond `SYNERGY_ALERT_THRESHOLD` (default 5) the
monitor calls `alert_dispatcher.dispatch_alert` so repeated failures trigger an
operator notification.

Every successful weight save increments the `synergy_weight_updates_total`
gauge. If an update fails inside `_update_synergy_weights` or when running
`synergy_weight_cli.train_from_history`, the
`synergy_weight_update_failures_total` gauge increases. The CLI,
`SelfImprovementEngine` and `SynergyAutoTrainer` dispatch an alert for every
failed update which also increments the
`synergy_weight_update_alerts_total` gauge.

## Standalone auto trainer

Run `synergy_auto_trainer.py` directly to update weights from the history
database without launching the full sandbox:

```bash
python -m menace.synergy_auto_trainer --history-file sandbox_data/synergy_history.db --weights-file sandbox_data/synergy_weights.json
```

By default it trains continuously every 600 seconds. Use `--interval` to adjust
the delay or `--run-once` to perform a single update and exit. The trainer
records the last processed history entry in `last_ts.json` (configurable with
`--progress-file`) and resumes from there on the next run so repeated restarts
only train on new entries. When running `run_autonomous.py` the same trainer is
started automatically if `AUTO_TRAIN_SYNERGY=1`; invoke it manually only when
you want to retrain outside the autonomous loop.

## Docker Compose setup

`docker-compose.synergy.yml` runs the exporter and trainer as separate
containers. Start both services with:

```bash
docker compose -f docker-compose.synergy.yml up
```

The **synergy_exporter** exposes metrics from
`sandbox_data/synergy_history.db` on `http://localhost:8003/metrics` while the
**synergy_trainer** periodically updates
`sandbox_data/synergy_weights.json`. The compose file mounts `./sandbox_data`
into each container so history and weights persist between runs. Environment
variables defined in `.env` control the update interval and exporter port.

Stop the services with:

```bash
docker compose -f docker-compose.synergy.yml down
```
Failed training attempts no longer stop the background process. The trainer logs
the error and retries after the configured interval so progress in
`last_ts.json` (or the specified `--progress-file`) is preserved.
If `synergy_weight_cli.py` exits with a non-zero code the trainer raises
`SynergyWeightCliError`, writes the error to the log and continues with the next
cycle.

## Running exporter and auto trainer together

The exporter and trainer are typically enabled at the same time so that
`synergy_history.db` is continuously exported while the weights update in the
background. Add the following variables to your `.env` and launch the sandbox
normally:

```dotenv
AUTO_TRAIN_SYNERGY=1
AUTO_TRAIN_INTERVAL=600
EXPORT_SYNERGY_METRICS=1
SYNERGY_METRICS_PORT=8003
SYNERGY_EXPORTER_CHECK_INTERVAL=10
```

```bash
python run_autonomous.py
```

Metrics are then available on
`http://localhost:${SYNERGY_METRICS_PORT}/metrics` and
`synergy_weights.json` refreshes every `${AUTO_TRAIN_INTERVAL}` seconds.

### Troubleshooting

- **Port in use** – change `SYNERGY_METRICS_PORT` or stop the conflicting
  service.
- **Weights not updating** – verify `AUTO_TRAIN_SYNERGY=1` and check write
  permissions for `synergy_weights.json`.
- **Exporter unreachable** – ensure the logs show "Synergy metrics exporter
  running" and that
  `curl http://localhost:${SYNERGY_METRICS_PORT}/health` returns
  `{"status": "ok"}`.

## Synergy weights and SelfDebuggerSandbox scoring

`SelfDebuggerSandbox` evaluates patches using a composite score derived from
recent metrics. When the sandbox runs as part of `SelfImprovementEngine` these
metrics are multiplied by the current synergy weights prior to scoring. Higher
weights therefore increase the influence of the corresponding metrics on patch
ranking. For example, a large `synergy_reliability` weight boosts patches that
improve reliability metrics while a negative weight reduces their impact.

The sandbox loads the values from `synergy_weights.json` unless overrides are
provided via environment variables such as `SYNERGY_WEIGHT_ROI`. You can also
pass a custom mapping to `_composite_score` when integrating the sandbox into
other tooling.
