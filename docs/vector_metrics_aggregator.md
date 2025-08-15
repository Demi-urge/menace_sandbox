# Vector Metrics Aggregator

`vector_metrics_aggregator.py` summarises entries stored in
`VectorMetricsDB` and produces heatmap‑ready data files. The aggregator
is intended to run on a schedule (hourly or daily) and its output can be
served by the existing `metrics_dashboard.py`.

## Command Line Usage

Aggregate the last hour of data and write JSON/CSV files (named
`vector_metrics_heatmap_hourly.json`/`.csv` by default):

```bash
python -m menace.vector_metrics_aggregator --period hourly
```

Daily aggregates produce files such as
`vector_metrics_heatmap_daily.json`:

```bash
python -m menace.vector_metrics_aggregator --period daily
```

Custom database or output locations may be supplied with `--db`,
`--json` and `--csv` arguments. When omitted, filenames include the
aggregation period so hourly and daily jobs keep separate outputs.

## Cron Scheduling

Run the aggregator every hour and every night at midnight using cron:

```cron
0 * * * * python -m menace.vector_metrics_aggregator --period hourly
0 0 * * * python -m menace.vector_metrics_aggregator --period daily
```

Each run writes `vector_metrics_heatmap_<period>.json` and
`vector_metrics_heatmap_<period>.csv`.

## Dashboard Endpoint

`metrics_dashboard.py` exposes the aggregated data at
`/vector_heatmap/<period>`. Start the dashboard and fetch the JSON for
use in front‑end heatmaps:

```bash
python -m menace.metrics_dashboard --port 8002
curl http://localhost:8002/vector_heatmap/hourly
```

The endpoint simply returns the latest contents of
`vector_metrics_heatmap_<period>.json`.
