# Metrics Dashboard

Menace stores statistics in `MetricsDB` and records evolution cycles in
`EvolutionHistoryDB`. To follow long-term trends you can expose these
metrics to a Prometheus server and visualise them using Grafana or a
simple Flask dashboard.

## Prometheus Export

Start the metrics server so Prometheus can scrape the gauges:

```python
from menace.metrics_exporter import start_metrics_server

start_metrics_server(8001)  # exposes http://localhost:8001
```

Add the port to your Prometheus `scrape_configs` and the gauges will
appear under names such as `learning_cv_score` and
`evolution_cycle_count`. Workflow benchmarking exposes
`workflow_cpu_time_seconds`, `workflow_memory_usage_mb`,
`workflow_network_bytes`, latency percentiles and disk I/O gauges for
real-time dashboards. The median latency is exported via
`workflow_latency_median_seconds`.

## Dashboard Options

* **Grafana** – configure Prometheus as a data source and create panels
  for the exported gauges. Trends like `long_term_roi_trend()` can be
  plotted alongside efficiency or error rates.
* **Flask** – for lighter setups you can run a small Flask app that
  queries the Prometheus HTTP API and renders charts with a library such
  as Chart.js. The :class:`menace.monitoring_dashboard.MonitoringDashboard`
  helper exposes metrics, evolution history and error summaries in a
  single view and can email periodic reports or write them to disk.

## Tracking Evolution Cycles

`EvolutionHistoryDB` stores `patch_id`, `workflow_id` and
`trending_topic` for each cycle. Include these fields in your dashboard
so you can correlate metric jumps with particular patches, workflows or
trending topics.

## Synergy Metrics

When synergy runs are enabled additional metrics such as
`synergy_profitability`, `synergy_revenue` and `synergy_projected_lucrativity` are stored in the
ROI history. The dashboard exposes these values via their metric names so you
can fetch tables or include them in the prediction plot.

```bash
python -m menace.metrics_dashboard --file roi_history.json --port 8002
# Visit http://localhost:8002/metrics/synergy_profitability
# Visit http://localhost:8002/metrics/synergy_revenue
# Visit http://localhost:8002/metrics/synergy_projected_lucrativity
```

The endpoints return JSON data with recorded values and prediction series which
can be embedded in custom dashboards.


### Synergy Prometheus Exporter

Set `EXPORT_SYNERGY_METRICS=1` when running `run_autonomous.py` to expose the
latest values from `synergy_history.db` as Prometheus gauges. Legacy JSON files
are migrated automatically. The exporter
listens on `SYNERGY_METRICS_PORT` (default 8003).

```bash
EXPORT_SYNERGY_METRICS=1 SYNERGY_METRICS_PORT=8003 python run_autonomous.py
```

