# Vector Metrics Aggregator

`vector_metrics_aggregator.py` summarises entries stored in
`VectorMetricsDB` and produces heatmap‑ready data files. The aggregator
is intended to run on a schedule (hourly or daily) and its output can be
served by the existing `metrics_dashboard.py`.

## Command Line Usage

Aggregate the last hour of data and write JSON/CSV files:

```bash
python -m menace.vector_metrics_aggregator --period hourly
```

Daily aggregates can be produced with:

```bash
python -m menace.vector_metrics_aggregator --period daily
```

Custom database or output locations may be supplied with `--db`,
`--json` and `--csv` arguments.

## Cron Scheduling

Run the aggregator every hour and every night at midnight using cron:

```cron
0 * * * * python -m menace.vector_metrics_aggregator --period hourly
0 0 * * * python -m menace.vector_metrics_aggregator --period daily
```

The files `vector_metrics_heatmap.json` and
`vector_metrics_heatmap.csv` will be overwritten on each run.

## Dashboard Endpoint

`metrics_dashboard.py` exposes the aggregated data at
`/vector_heatmap`. Start the dashboard and fetch the JSON for use in
front‑end heatmaps:

```bash
python -m menace.metrics_dashboard --port 8002
curl http://localhost:8002/vector_heatmap
```

The endpoint simply returns the latest contents of
`vector_metrics_heatmap.json`.
