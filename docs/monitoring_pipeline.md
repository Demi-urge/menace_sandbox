# Monitoring Pipeline

Menace exposes metrics, logs and errors so you can keep track of every bot.

## Prometheus Metrics

`DataBot` and `metrics_exporter.start_metrics_server()` publish CPU, memory and
ROI gauges. Set `METRICS_PORT` before launching `run_autonomous.py` or
`synergy_tools.py` to start the exporter automatically. Point Prometheus at the
metrics server and visualise the gauges with Grafana. The exporter now exposes
`roi_threshold_gauge` and `synergy_threshold_gauge` so dashboards can track the
current ROI and synergy convergence thresholds. Two additional gauges help
monitor preset adaptation:

- `roi_forecast` – predicted ROI for the next sandbox cycle
- `synergy_adaptation_actions_total{action="<name>"}` – counter of how often a
  specific adaptation action executed
- `roi_forecast_failures_total` – number of exceptions raised while computing the ROI forecast
- `synergy_forecast_failures_total` – number of exceptions raised while predicting synergy

Add the metrics server to your Prometheus configuration and query these gauges
to inspect upcoming ROI and the behaviour of the preset RL agent.
For a local run you can start the exporter with:

```bash
METRICS_PORT=8001 python run_autonomous.py
curl http://localhost:8001/metrics | grep roi_forecast
```

## Log Aggregation

`OperationalMonitoringBot` can forward metrics and anomaly records to either
Elasticsearch or Splunk. Initialise the bot with an `ESIndex` or `SplunkHEC`
instance:

```python
from menace.operational_monitor_bot import OperationalMonitoringBot
from menace.database_steward_bot import ESIndex
from menace.splunk_logger import SplunkHEC

monitor = OperationalMonitoringBot(es=ESIndex(), splunk=SplunkHEC(token="token"))
```

Dashboards in Kibana or Splunk can then search recent events and trends.
Set `SANDBOX_CENTRAL_LOGGING=1` to forward logs from sandbox components to the
configured audit trail while running `run_autonomous.py` or `synergy_tools.py`.

## Sentry Alerts

Install `sentry-sdk` and create a `SentryClient` to capture exceptions from
`ErrorLogger`:

```python
from menace.sentry_client import SentryClient
from menace.error_logger import ErrorLogger
from vector_service import ContextBuilder

sentry = SentryClient("https://public@o0.ingest.sentry.io/0")
logger = ErrorLogger(sentry=sentry, context_builder=ContextBuilder())
```

## Watchdog Runbooks

`Watchdog` evaluates failure trends using `ErrorDB`, `ROIDB` and `MetricsDB`.
When consecutive crashes, ROI decline and downtime all exceed thresholds it
escalates through an `EscalationProtocol` with a JSON runbook. `compile_dossier()`
writes the runbook to `/tmp/watchdog_runbook_<epoch>.json` and the path is attached
to the alert. The unique runbook ID returned by the protocol is logged for traceability.

```python
from menace.watchdog import Watchdog, Notifier
from vector_service import ContextBuilder

notifier = Notifier(slack_webhook="https://hooks.slack.com/services/TOKEN")
builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
watchdog = Watchdog(
    err_db, roi_db, metrics_db, notifier=notifier, context_builder=builder
)
watchdog.check()
```
`Watchdog` requires a `ContextBuilder` supplied via the `context_builder` argument.
The `Watchdog` logs the generated runbook ID whenever an escalation occurs so
operators can easily cross-reference alerts with archived runbooks.

The runbook helps operators triage issues quickly.

`compile_dossier()` also queries `KnowledgeGraph.root_causes()` for any bots
seen in recent telemetry. These dependencies are stored under a new
`"dependencies"` key in the JSON file:

```json
{
  "dependencies": {
    "Alpha": ["error:42", "model:1"],
    "Beta": []
  }
}
```

Combining Prometheus metrics, centralised logs, Sentry alerts and Watchdog
runbooks provides a complete picture of Menace health.

## Synergy Restart Alerts

`ExporterMonitor` and `AutoTrainerMonitor` keep the synergy exporter and auto
trainer running in the background. Each restart increments
`synergy_exporter_restarts_total` or `synergy_trainer_restarts_total`. When a
service exceeds `SYNERGY_ALERT_THRESHOLD` (default 5) the monitor invokes
`alert_dispatcher.dispatch_alert` with the restart count so operators can
investigate persistent crashes.

## Cascading Error Forecasts

`ErrorBot.predict_errors()` now calls `KnowledgeGraph.cascading_effects()` for
any bot flagged by the `ErrorForecaster`. The resulting node chains are stored
in `ErrorBot.last_forecast_chains` and returned as strings in the prediction
list. Operators can review cascades such as `bot:A -> model:1 -> code:foo` to
understand likely downstream failures and pre-empt remediation steps.

## Container Creation Alerts

`environment.py` dispatches an alert via `alert_dispatcher` whenever repeated
container launches fail. Each dispatched alert increments the
`container_creation_alerts_total{image="<name>"}` gauge so the affected image
can be monitored. Start exporting the gauge with
`metrics_exporter.start_metrics_server` and add the port to your Prometheus
configuration:

```python
from menace.metrics_exporter import start_metrics_server

start_metrics_server(8001)  # exposes http://localhost:8001/metrics
```

```yaml
scrape_configs:
  - job_name: 'menace'
    static_configs:
      - targets: ['localhost:8001']
```

## Synergy Weight Update Alerts

`synergy_weight_cli`, `SelfImprovementEngine` and `SynergyAutoTrainer` dispatch a
`synergy_weight_update_failure` alert whenever weight training fails.
`synergy_weight_update_alerts_total` increments with every alert and
`SYNERGY_WEIGHT_ALERT_THRESHOLD` (default `5`) determines when consecutive
failures trigger the alert from the auto trainer. Export the gauge using the
same metrics server so Prometheus can scrape the failure count.

## Central Alert Forwarding

Set `SANDBOX_CENTRAL_LOGGING=1` to forward alerts emitted by
`self_test_service`, `synergy_auto_trainer` and `environment.py` through
`alert_dispatcher`. The environment variable enables central logging so the
dispatcher writes to the configured audit trail or Kafka topic in addition to
standard output. When running `run_autonomous.py` or `synergy_tools.py` the
variable defaults to `1`, so disable it explicitly with `SANDBOX_CENTRAL_LOGGING=0`
if forwarding is not desired.

## Metrics Dashboard

`metrics_dashboard.MetricsDashboard` exposes a simple JSON API with the latest
gauge values. The endpoint `/metrics` now includes container creation and
synergy training gauges alongside ROI metrics:

* `container_creation_success_total`
* `container_creation_failures_total`
* `container_creation_alerts_total`
* `synergy_trainer_iterations`
* `synergy_trainer_failures_total`
* `synergy_weight_update_failures_total`
* `roi_forecast_failures_total`
* `synergy_forecast_failures_total`
* `roi_threshold_gauge`
* `synergy_threshold_gauge`
* `roi_forecast`
* `synergy_adaptation_actions_total{action}`

Start the dashboard and query the metrics:

```bash
python -m menace.metrics_dashboard --port 8002
curl http://localhost:8002/metrics
```
Example output:
```
roi_forecast 0.82
synergy_adaptation_actions_total{action="use_history"} 3
```

## High-Risk Module Predictions

`ErrorOntologyDashboard` exposes a prediction API for modules likely to fail.
The endpoint `/predicted_modules` uses
`ErrorClusterPredictor.predict_high_risk_modules` to rank modules by risk and
returns a JSON object with `labels` and `rank` arrays.

Start the dashboard and query the predictions:

```bash
python -m menace.error_ontology_dashboard --port 8003
curl http://localhost:8003/predicted_modules
```

The response helps operators anticipate which modules may require attention.
