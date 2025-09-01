# Self-Improvement Quickstart

This guide covers the basics for running the autonomous sandbox in
self-improvement mode.

## Required packages

- Install core dependencies with `pip install -r requirements.txt` or run
  `./setup_env.sh`.
- `sandbox_runner` (including the `sandbox_runner.orphan_integration`
  module) and `quick_fix_engine` provide the execution harness and patch
  generation. Install them explicitly when missing:
  `pip install sandbox_runner quick_fix_engine`.
- Optional: `prometheus-client` exposes metrics via HTTP.

## Launch `start_autonomous_sandbox.py`

Start the sandbox with an optional log level:

```bash
python start_autonomous_sandbox.py --log-level INFO
```

The wrapper forwards control to `sandbox_runner.bootstrap.launch_sandbox` and
exits with a non-zero status when initialisation fails.

## Environment variables

| Variable | Description | Default |
| --- | --- | --- |
| `SANDBOX_REPO_PATH` | repository root used during self-improvement | current working directory |
| `SANDBOX_DATA_DIR` | directory for metrics and state files | `sandbox_data` |
| `SANDBOX_ENV_PRESETS` | comma separated scenario preset files | unset |
| `AUTO_TRAIN_INTERVAL` | seconds between sandbox training passes | `600` |
| `SYNERGY_TRAIN_INTERVAL` | steps between synergy weight updates | `10` |
| `ADAPTIVE_ROI_RETRAIN_INTERVAL` | cycles between adaptive ROI model retraining | `20` |
| `METRICS_PORT` | start metrics server on this port when set | unset (disabled) |

## Monitoring metrics and logs

Set `METRICS_PORT` to expose Prometheus gauges via
`metrics_exporter.start_metrics_server`. Metrics are available at
`http://localhost:<port>/` and can be scraped by Prometheus or viewed with a
browser.

Runtime logs are written under the `logs/` directory and streamed to stdout.
Increase verbosity with `--log-level DEBUG` when launching the sandbox and use
`tail -f logs/*.log` to monitor activity. ROI, synergy and error messages help
identify regressions during self-improvement cycles.

