# Autonomous Sandbox

This guide describes the prerequisites and environment variables used when
running the fully autonomous sandbox.

## First-time setup

Follow these steps when launching the sandbox for the first time:

1. **Clone the repository** and switch into it:

   ```bash
   git clone https://example.com/menace_sandbox.git
   cd menace_sandbox
   ```

2. **Install system dependencies**. The required packages are listed below.
   On Debian-based systems you can run:

   ```bash
   sudo apt install ffmpeg tesseract-ocr chromium-browser qemu-system-x86
   ```

3. **Run the personal setup script** which installs Python packages,
   verifies optional tools and creates a `.env` file:

   ```bash
   scripts/setup_personal.sh
   ```

   The script wraps `setup_env.sh` and `setup_dependencies.py` and fails if
   `ffmpeg`, `tesseract`, `qemu-system-x86_64` or `docker` are missing.
   When used, you can skip steps 4–7 below.

4. **Install Python packages**. Use the helper script which installs
   everything from `requirements.txt` and sets up a development environment:

   ```bash
   ./setup_env.sh
   ```

5. **Install optional packages** used by reinforcement learning and the
   metrics dashboard:

   ```bash
   scripts/setup_optional.sh
   ```

6. **Bootstrap the environment** to verify optional dependencies and
   install anything missing:

   ```bash
   python scripts/bootstrap_env.py
   ```

7. **Create a `.env` file** with sensible defaults. The easiest way is:

   ```bash
   python -c 'import auto_env_setup as a; a.ensure_env()'
   ```

   Edit the resulting `.env` to add missing API keys.

8. **Start the visual agent** in one terminal and the autonomous loop in
   another as described below.

## System packages

The sandbox relies on several system utilities in addition to the Python
dependencies listed in `pyproject.toml`:

- `ffmpeg` – audio extraction for video clips
- `tesseract-ocr` – OCR based bots
- `chromium-browser` or any Chromium based browser for web automation
- `qemu-system-x86` – optional virtualization backend used for cross platform presets

Most of these packages are installed automatically when using the provided
Dockerfile. On bare metal they must be installed manually via your package
manager.

After the system tools are in place install the Python requirements via
`./setup_env.sh` or `pip install -r requirements.txt`.

### Package checklist

1. **Install base utilities** – make sure `python3`, `pip` and `git` are
   available on your system.
2. **Install required packages** – use your package manager to install all
   system tools listed above. On Debian/Ubuntu this command covers them:

   ```bash
   sudo apt install ffmpeg tesseract-ocr chromium-browser qemu-system-x86
   ```

3. **Install Python dependencies** – run the provided helper script to create
   a virtual environment and install all packages from `requirements.txt`:

   ```bash
   ./setup_env.sh
   ```

4. **Install optional extras** – reinforcement learning and dashboard features
   rely on additional libraries. They can be set up with:

   ```bash
   scripts/setup_optional.sh
   ```

5. **Verify the environment** – execute the bootstrap script which checks that
   all optional tools are available and installs anything missing:

   ```bash
   python scripts/bootstrap_env.py
   ```

   If any dependency fails to install, run the command again or consult the
   troubleshooting section below.

## Recommended environment variables

`auto_env_setup.ensure_env()` generates a `.env` file with sensible defaults. The following variables are particularly relevant for the autonomous workflow:

- `AUTO_SANDBOX=1` – run the sandbox on first launch
- `SANDBOX_CYCLES=5` – number of self‑improvement iterations
- `SANDBOX_ROI_TOLERANCE=0.01` – ROI delta required to stop early
- `RUN_CYCLES=0` – unlimited run cycles for the orchestrator
- `AUTO_DASHBOARD_PORT=8001` – start the metrics dashboard
- `METRICS_PORT=8001` – start the internal metrics exporter (same as `--metrics-port`)
- `EXPORT_SYNERGY_METRICS=1` – enable the Synergy Prometheus exporter
- `SYNERGY_METRICS_PORT=8003` – port for the exporter
- `SELF_TEST_METRICS_PORT=8004` – port exposing self‑test metrics
- `VISUAL_AGENT_TOKEN=<secret>` – authentication token for `menace_visual_agent_2.py`
 - `VISUAL_AGENT_AUTOSTART=1` – automatically launch the visual agent when missing
 - `VISUAL_AGENT_AUTO_RECOVER=1` – enable automatic queue recovery (set to `0` to disable)
- `VISUAL_AGENT_TOKEN_ROTATE` – new token value used to restart the
  visual agent between sandbox runs
- `VISUAL_AGENT_SSL_CERT` – optional path to an SSL certificate for HTTPS
- `VISUAL_AGENT_SSL_KEY` – optional path to the corresponding private key
- `ROI_THRESHOLD` – override the diminishing ROI threshold
- `ROI_CONFIDENCE` – t-test confidence when flagging modules
- `SYNERGY_THRESHOLD` – fixed synergy convergence threshold
- `SYNERGY_THRESHOLD_WINDOW` – samples used for adaptive synergy threshold
- `SYNERGY_THRESHOLD_WEIGHT` – exponential weight for threshold calculation
- `SYNERGY_CONFIDENCE` – confidence level for synergy convergence checks
- `SANDBOX_PRESET_RL_PATH` – path to the RL policy used for preset adaptation
- `SANDBOX_PRESET_RL_STRATEGY` – reinforcement learning algorithm
- `SANDBOX_ADAPTIVE_AGENT_PATH` – path to the adaptive RL agent state
- `SANDBOX_ADAPTIVE_AGENT_STRATEGY` – algorithm for the adaptive agent

Additional API keys such as `OPENAI_API_KEY` may be added to the same `.env` file.

## Launch sequence

1. Start the visual agent service:

   ```bash
   python menace_visual_agent_2.py
   ```

   The server listens on the port defined by `MENACE_AGENT_PORT` (default `8001`). Ensure `VISUAL_AGENT_TOKEN` is exported before running the command.

2. In a separate terminal start the autonomous loop:

   ```bash
   python run_autonomous.py
   ```

   Alternatively run `scripts/launch_personal.py` to start the visual agent and
   autonomous loop sequentially.

  `run_autonomous.py` spawns a background `VisualAgentMonitor` which
  polls the visual agent's `/health` endpoint. If the service stops
  responding the monitor restarts it and posts to `/recover` so queued
  tasks resume automatically. A built in queue watchdog thread also
  verifies database integrity and restarts the worker if it crashes.
  Manual recovery is rarely required and the behaviour is controlled by
  `VISUAL_AGENT_AUTO_RECOVER=1`.

   Use `--metrics-port` or `METRICS_PORT` to expose Prometheus metrics from the sandbox.

   The script verifies system dependencies, creates default presets using `environment_generator` and invokes the sandbox runner. The metrics dashboard is available at `http://localhost:${AUTO_DASHBOARD_PORT}` once started.
   If the previous run terminated unexpectedly you can append `--recover` to reload the last recorded ROI and synergy histories.
   Use `--preset-debug` to enable verbose logging of preset adaptation. Combine it with `--debug-log-file <path>` to write these logs to a file for later inspection.
   Use `--forecast-log <path>` to store ROI forecasts and threshold values for each run.
   Use `--dynamic-workflows` to build temporary workflows from module groups when the workflow database is empty. Control the clustering with `--module-algorithm`, `--module-threshold` and `--module-semantic` which mirror the options of `discover_module_groups`.

   Example:

   ```bash
   python run_autonomous.py --dynamic-workflows \
     --module-semantic --module-threshold 0.25
   ```

## Visual agent queueing

`menace_visual_agent_2.py` accepts concurrent requests but `/run` always
responds with HTTP `202` and a task id. Submitted tasks are appended to
`SANDBOX_DATA_DIR/visual_agent_queue.db` and processed sequentially. Poll
`/status/<id>` to monitor progress. The persistent queue avoids race conditions
and survives restarts.
The queue is stored in a SQLite database so tasks persist across restarts. A matching `visual_agent_state.json` file records the status of each job and when the last task completed. Both are loaded on startup so unfinished work continues automatically.

Use `menace_visual_agent_2.py --resume` to process any pending
entries without launching the HTTP service. Stale lock and PID files are cleaned
automatically; run `menace_visual_agent_2.py --cleanup` if the service did not
shut down cleanly.

### Crash recovery

By default the agent requeues tasks marked as `running` on startup. Use
`--no-auto-recover` or set `VISUAL_AGENT_AUTO_RECOVER=0` to disable this
behaviour. The queue database is verified on startup; a corrupt file is moved
aside as ``visual_agent_queue.db.corrupt.<timestamp>`` and rebuilt from
``visual_agent_state.json`` when possible. The client also writes failed
requests to `visual_agent_client_queue.jsonl` and retries them periodically.
Database errors encountered while processing tasks now trigger this recovery
automatically, so manual ``--recover-queue`` is rarely needed. The commands
below are mainly useful for troubleshooting when automatic recovery fails.
Additional CLI helpers simplify manual repairs:
If the service refuses to start because of a stale lock or PID file use `python menace_visual_agent_2.py --cleanup` first.

```bash
python menace_visual_agent_2.py --flush-queue       # drop all queued tasks
python menace_visual_agent_2.py --recover-queue     # reload queue from disk
python menace_visual_agent_2.py --repair-running    # mark running tasks queued
python menace_visual_agent_2.py --resume            # process queue headlessly
python menace_visual_agent_2.py --cleanup           # remove stale lock/PID
```

To recover manually after an unexpected shutdown:
```bash
python menace_visual_agent_2.py --cleanup
python menace_visual_agent_2.py --resume
python menace_visual_agent_2.py --repair-running --recover-queue
python run_autonomous.py --recover
```

## Local run essentials

The sandbox reads several paths and authentication tokens from environment variables. These defaults are suitable for personal deployments and can be overridden in your `.env`:

- `VISUAL_AGENT_TOKEN` – shared secret for the visual agent service.
- `VISUAL_AGENT_TOKEN_ROTATE` – when set, the new token used to restart the
  visual agent between runs.
- `VISUAL_AGENT_SSL_CERT` – path to the TLS certificate served by
  `menace_visual_agent_2.py` when running over HTTPS.
- `VISUAL_AGENT_SSL_KEY` – path to the private key for the certificate.
- `SANDBOX_DATA_DIR` – directory where ROI history, presets and patch records are stored. Defaults to `sandbox_data`.
 - `SANDBOX_AUTO_MAP` – when set to `1` the sandbox builds or
   refreshes `module_map.json` on startup. The legacy `SANDBOX_AUTODISCOVER_MODULES`
   variable is still recognised with a warning.
 - `SANDBOX_REFRESH_MODULE_MAP` – force regeneration even when the map already
   exists.
  When either condition is met the repository is analysed with
  `build_module_map` and the resulting clusters are stored in the file.
   `ModuleIndexDB` loads these assignments so ROI and synergy metrics aggregate
   by module group.
- `PATCH_SCORE_BACKEND_URL` – optional remote backend for patch scores. Supports `http://`, `https://` or `s3://bucket/prefix` URLs.
- `DATABASE_URL` – connection string for the primary database. Defaults to `sqlite:///menace.db`.
- `BOT_DB_PATH` – location of the bot registry database, default `bots.db`.
- `BOT_PERFORMANCE_DB` – path for the performance history database, default `bot_performance_history.db`.
- `MAINTENANCE_DB` – SQLite database used for maintenance logs, default `maintenance.db`.

Generate the module grouping manually with:

```bash
python scripts/generate_module_map.py
```

### Example `.env`

```dotenv
VISUAL_AGENT_TOKEN=my-secret-token
#VISUAL_AGENT_TOKEN_ROTATE=new-secret
VISUAL_AGENT_SSL_CERT=/path/to/cert.pem
VISUAL_AGENT_SSL_KEY=/path/to/key.pem
SANDBOX_DATA_DIR=~/menace_data
PATCH_SCORE_BACKEND_URL=http://example.com/api/scores
DATABASE_URL=sqlite:///menace.db
BOT_DB_PATH=bots.db
BOT_PERFORMANCE_DB=bot_performance_history.db
MAINTENANCE_DB=maintenance.db
OPENAI_API_KEY=sk-xxxxx
```

## Running locally step by step

The following sequence shows how to launch the sandbox on a personal machine.
It assumes the repository is already cloned and all dependencies from
`requirements.txt` are installed.

1. **Generate a `.env` file** or update an existing one using
   `auto_env_setup.ensure_env()`:

   ```bash
   python -c 'import auto_env_setup as a; a.ensure_env()'
   ```

   Review the generated file and add any missing API keys.

2. **Start the visual agent** and leave it running:

   ```bash
   python menace_visual_agent_2.py
   ```

3. **Launch the autonomous loop** in a second terminal:

   ```bash
   python run_autonomous.py
   ```

   To start both components sequentially from a single shell you can run:

   ```bash
   python menace_visual_agent_2.py &
   python run_autonomous.py
   ```

   Terminate the backgrounded visual agent when you are done.

## Logging

Application logs are written to `stdout` by default. Set `SANDBOX_JSON_LOGS=1`
to output logs as single line JSON objects which simplifies collection by log
aggregators. When running under `run_autonomous.py` the same environment
variable enables JSON formatted logs for the sandbox runner as well.

Set `SANDBOX_CENTRAL_LOGGING=1` to forward logs from `self_test_service`,
`synergy_auto_trainer` and `synergy_monitor` to the audit trail defined by
`AUDIT_LOG_PATH`. When launching with `run_autonomous.py` or `synergy_tools.py`
the variable defaults to `1`, so set `SANDBOX_CENTRAL_LOGGING=0` to disable the
forwarding. If `KAFKA_HOSTS` is set, logs are published to Kafka instead via
`KafkaMetaLogger`.

Use `--log-level LEVEL` when running `run_autonomous.py` to change the console
verbosity. The flag falls back to the `SANDBOX_LOG_LEVEL` environment variable
(or `LOG_LEVEL`) when omitted. Pass `--verbose` to enable full debug output
regardless of the configured log level. Setting `SANDBOX_DEBUG=1` (or
`SANDBOX_VERBOSE=1`) has the same effect when `setup_logging()` is invoked
without an explicit level.

Long running services rotate their own log files such as
`service_supervisor.py` which keeps up to three 1&nbsp;MB archives. Rotate or
clean old logs periodically when persisting them on disk.

During each sandbox iteration the runner logs where presets originated
("static file", "history adaptation" or "RL agent"), the next ROI prediction and
its confidence interval, and the computed synergy threshold. Synergy convergence
checks also log the maximum absolute EMA and the t‑test confidence so the
associated p‑values are visible when troubleshooting.

After each iteration the thresholds are appended to
`sandbox_data/threshold_log.jsonl`. Each line contains a JSON object with the
keys `timestamp`, `run`, `roi_threshold`, `synergy_threshold` and `converged`.

### Preset adaptation debug logs

Use `--preset-debug` when launching `run_autonomous.py` to log every preset
adjustment decision. The flag sets the `PRESET_DEBUG` environment variable so
`environment_generator` emits detailed messages each time a parameter changes.
Combine it with `--debug-log-file` to write these messages to a separate file
for later inspection:

```bash
python run_autonomous.py --preset-debug \
    --debug-log-file sandbox_data/preset_debug.log
```

The file is appended to across runs and contains the raw debug statements from
`environment_generator`. Without `--debug-log-file` the extra logs appear only
on the console.

Pass `--forecast-log <path>` to append ROI forecast values and the calculated
thresholds for each run. The file is written in JSON lines format so it can be
processed programmatically.

## Synergy metrics exporter

Set `EXPORT_SYNERGY_METRICS=1` when launching `run_autonomous.py` to start the
`SynergyExporter`. It reads `synergy_history.db` (migrating any legacy JSON
file) and exposes the latest
values on `http://localhost:${SYNERGY_METRICS_PORT}/metrics` (default port
`8003`) for Prometheus scraping. See
[`synergy_learning.md`](synergy_learning.md#interpreting-synergyexporter-metrics)
for an explanation of the exported metrics and how the synergy weights influence
ROI.

Alternatively start the exporter directly from the command line:

```bash
python -m menace.synergy_exporter --history-file sandbox_data/synergy_history.db
```

The exporter listens on port `8003` by default. Pass `--port` to change it:

```bash
python -m menace.synergy_exporter --history-file sandbox_data/synergy_history.db --port 8003
```

Use `--interval` to change how often the history file is scanned for updates.

### Synergy dashboard

The history can also be viewed in a small dashboard. Start it with

```bash
python -m menace.self_improvement_engine synergy-dashboard --file sandbox_data/synergy_history.db
```

Use `--wsgi gunicorn` or `--wsgi uvicorn` to serve the dashboard via Gunicorn or Uvicorn instead of the Flask development server.

### Synergy auto trainer

Set `AUTO_TRAIN_SYNERGY=1` when invoking `run_autonomous.py` to update
`synergy_weights.json` automatically from `synergy_history.db`. To train the
weights without running the full sandbox execute the trainer directly:

```bash
python -m menace.synergy_auto_trainer --history-file sandbox_data/synergy_history.db --weights-file sandbox_data/synergy_weights.json
```

Use this manual invocation to refresh the weights once or run the trainer in
isolation. Pass `--interval` or `--run-once` to control how often it updates.

When the Prometheus exporter is running the trainer also publishes two gauges:

- `synergy_trainer_iterations` – number of training cycles completed
- `synergy_trainer_last_id` – ID of the last history row processed


### Enabling auto trainer with exporter

Follow these steps to run the background trainer and the metrics exporter side
by side:

1. **Set the required environment variables**. Both components are enabled via
   flags when launching `run_autonomous.py`:

   ```bash
   export AUTO_TRAIN_SYNERGY=1
   export EXPORT_SYNERGY_METRICS=1
   ```

   Adjust `AUTO_TRAIN_INTERVAL` or `SYNERGY_METRICS_PORT` if the defaults are not
   suitable.

2. **Start the sandbox** normally:

   ```bash
   python run_autonomous.py
   ```

   The exporter listens on `http://localhost:${SYNERGY_METRICS_PORT}/metrics` and
   the trainer updates `synergy_weights.json` at the configured interval.

3. **Verify that both services are running**. The sandbox logs report when the
   exporter is ready and each time the trainer updates the weights. Point a
   Prometheus instance at the exporter URL to record the metrics.

### Configuring SynergyAutoTrainer and SynergyExporter

For production deployments both services usually run alongside the sandbox.
Export the following variables before launching `run_autonomous.py`:

```bash
export AUTO_TRAIN_SYNERGY=1
export AUTO_TRAIN_INTERVAL=600
export EXPORT_SYNERGY_METRICS=1
export SYNERGY_METRICS_PORT=8003
export SYNERGY_EXPORTER_CHECK_INTERVAL=10
python run_autonomous.py
```

The settings above refresh `synergy_weights.json` every ten minutes and expose
the latest metrics on `http://localhost:8003/metrics`. You can run the tools
manually when debugging:

```bash
python -m menace.synergy_auto_trainer --history-file /var/menace/synergy_history.db \
    --weights-file /var/menace/synergy_weights.json --interval 600
python -m menace.synergy_exporter --history-file /var/menace/synergy_history.db \
    --port 8003 --interval 5
```

**Recommended variables**

- `AUTO_TRAIN_SYNERGY=1` – enable the background trainer.
- `AUTO_TRAIN_INTERVAL=600` – training frequency in seconds.
- `EXPORT_SYNERGY_METRICS=1` – expose metrics for Prometheus.
- `SYNERGY_METRICS_PORT=8003` – exporter HTTP port.
- `SYNERGY_EXPORTER_CHECK_INTERVAL=10` – health check interval.

### Example `.env` for synergy services

```dotenv
AUTO_TRAIN_SYNERGY=1
AUTO_TRAIN_INTERVAL=600
EXPORT_SYNERGY_METRICS=1
SYNERGY_METRICS_PORT=8003
SYNERGY_EXPORTER_CHECK_INTERVAL=10
```

### Launching `synergy_tools.py` locally

Use the `synergy_tools.py` helper to run the exporter and trainer on a personal
machine without starting the full sandbox. Set the relevant environment
variables to enable each service:

```bash
export AUTO_TRAIN_SYNERGY=1
export EXPORT_SYNERGY_METRICS=1
export AUTO_TRAIN_INTERVAL=600
export SYNERGY_METRICS_PORT=8003
python synergy_tools.py --sandbox-data-dir sandbox_data
```

`EXPORT_SYNERGY_METRICS` starts the Prometheus exporter while
`AUTO_TRAIN_SYNERGY` enables periodic weight training.
Adjust `AUTO_TRAIN_INTERVAL` or `SYNERGY_METRICS_PORT` to fit your setup.
Press <kbd>Ctrl+C</kbd> to stop both services. The same environment variables are
respected as when running `run_autonomous.py`.

#### Visual agent concurrency

Requests to `/run` always return HTTP `202` with a task id. The agent processes
one job at a time from `visual_agent_queue.db`, so you can submit tasks
concurrently and poll `/status/<id>` until each job completes.

### Troubleshooting synergy services

- **Port already in use** – adjust `SYNERGY_METRICS_PORT` or use `--port` when
  starting the exporter. `netstat -tlnp` helps identify conflicting processes.
- **History file not found** – ensure the path passed to `--history-file` exists
  and is writable.
- **Weights never update** – verify that `AUTO_TRAIN_SYNERGY=1` is set and that
  the trainer can write its progress file.
- **Exporter stale** – confirm the exporter process is running and that
  `http://localhost:${SYNERGY_METRICS_PORT}/health` returns `{"status": "ok"}`.

### Advanced synergy learning

The default learner uses a lightweight actor–critic strategy. To enable deeper
reinforcement learning you can instantiate `SelfImprovementEngine` with
`DQNSynergyLearner`:

```python
from menace.self_improvement_engine import SelfImprovementEngine, DQNSynergyLearner

engine = SelfImprovementEngine(
    synergy_learner_cls=DQNSynergyLearner,
    synergy_weights_path="sandbox_data/synergy_weights.json",
)
engine.run_cycle()
```

This variant relies on PyTorch and persists the Q‑network weights alongside the
JSON file. See [synergy_learning.md](synergy_learning.md) for background on how
the learner adjusts the metrics.

## Docker usage

Build the sandbox image and run it inside a container using the helper scripts:

```bash
scripts/build_sandbox_image.sh
scripts/run_sandbox_container.sh
```

The run script mounts `sandbox_data/` for persistent metrics and loads
environment variables from `.env` so the container behaves the same as a
local installation.

## Troubleshooting

- **Missing dependencies** – run `./setup_env.sh` again to ensure all Python
  packages are installed. On bare metal verify that `ffmpeg` and `tesseract`
  are present in your `$PATH`.
- **Visual agent queue stalled** – tasks are appended to ``visual_agent_queue.db``
  and processed sequentially. The service automatically cleans stale locks and
  requeues running tasks on startup. Run ``menace_visual_agent_2.py --cleanup``
  or ``--repair-running`` if the queue appears stuck.
- **Dashboard not loading** – confirm that `AUTO_DASHBOARD_PORT` is free and no
  firewall blocks the connection. The dashboard starts automatically once the
  sandbox loop begins.
- **Patch score backend unreachable** – verify that `PATCH_SCORE_BACKEND_URL`
  points to a reachable HTTP or S3 endpoint. Check network connectivity and
  credentials when using S3. The sandbox falls back to local storage if the
  backend cannot be contacted.
- **Tests fail** – ensure all packages listed in the checklist are installed and
  rerun `./setup_env.sh` to reinstall the Python environment. Some tests rely on
  optional tools such as `ffmpeg` or Docker. Execute `pytest -x` to stop on the
  first failure and inspect the output for missing dependencies.
- **Self tests interrupted** – `SelfTestService` saves its progress to
  `sandbox_data/self_test_state.json` (configurable via `SELF_TEST_STATE`).
  Restarting the sandbox automatically resumes any incomplete test run.
- **Authentication errors** – HTTP 401 responses from the visual agent usually
  indicate an invalid token. Confirm that `VISUAL_AGENT_TOKEN` matches the
  secret configured in `.env` and restart the service when the token changes.
- **Synergy exporter not running** – ensure `EXPORT_SYNERGY_METRICS=1` is set
  and that `SYNERGY_METRICS_PORT` is free. Successful startup logs the message
  "Synergy metrics exporter running" together with the chosen port.
- **Exporter endpoint unreachable** – `curl http://localhost:${SYNERGY_METRICS_PORT}/health`
  should return `{"status": "ok"}`. If the sandbox keeps restarting the exporter
  consider raising `SYNERGY_EXPORTER_CHECK_INTERVAL`.
- **Missing synergy metrics** – verify that `synergy_history.db` exists and that
  `/metrics` on `SYNERGY_METRICS_PORT` exposes the expected gauges. When
  `AUTO_TRAIN_SYNERGY=1` additional trainer metrics appear alongside the
  exporter gauges.
- **Self‑test metrics not updating** – check that `SELF_TEST_INTERVAL` is set to
  a positive value. The gauges `self_test_passed_total`,
  `self_test_failed_total`, `self_test_average_runtime_seconds` and
  `self_test_average_coverage` are available on
  `http://localhost:${AUTO_DASHBOARD_PORT}/metrics`. The service keeps its
  progress in `SELF_TEST_STATE`.
