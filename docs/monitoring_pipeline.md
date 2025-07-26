# Monitoring Pipeline

Menace exposes metrics, logs and errors so you can keep track of every bot.

## Prometheus Metrics

`DataBot` and `metrics_exporter.start_metrics_server()` publish CPU, memory and
ROI gauges. Point Prometheus at the metrics server and visualise them with
Grafana.

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

## Sentry Alerts

Install `sentry-sdk` and create a `SentryClient` to capture exceptions from
`ErrorLogger`:

```python
from menace.sentry_client import SentryClient
from menace.error_logger import ErrorLogger

sentry = SentryClient("https://public@o0.ingest.sentry.io/0")
logger = ErrorLogger(sentry=sentry)
```

## Watchdog Runbooks

`Watchdog` evaluates failure trends using `ErrorDB`, `ROIDB` and `MetricsDB`.
When consecutive crashes, ROI decline and downtime all exceed thresholds it
escalates through an `EscalationProtocol` with a JSON runbook. `compile_dossier()`
writes the runbook to `/tmp/watchdog_runbook_<epoch>.json` and the path is attached
to the alert. The unique runbook ID returned by the protocol is logged for traceability.

```python
from menace.watchdog import Watchdog, Notifier

notifier = Notifier(slack_webhook="https://hooks.slack.com/services/TOKEN")
watchdog = Watchdog(err_db, roi_db, metrics_db, notifier=notifier)
watchdog.check()
```

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
