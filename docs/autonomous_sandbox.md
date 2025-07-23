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

3. **Install Python packages**. Use the helper script which installs
   everything from `requirements.txt` and sets up a development environment:

   ```bash
   ./setup_env.sh
   ```

4. **Bootstrap the environment** to verify optional dependencies and
   install anything missing:

   ```bash
   python scripts/bootstrap_env.py
   ```

5. **Create a `.env` file** with sensible defaults. The easiest way is:

   ```bash
   python -c 'import auto_env_setup as a; a.ensure_env()'
   ```

   Edit the resulting `.env` to add missing API keys.

6. **Start the visual agent** in one terminal and the autonomous loop in
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

## Recommended environment variables

`auto_env_setup.ensure_env()` generates a `.env` file with sensible defaults. The following variables are particularly relevant for the autonomous workflow:

- `AUTO_SANDBOX=1` – run the sandbox on first launch
- `SANDBOX_CYCLES=5` – number of self‑improvement iterations
- `SANDBOX_ROI_TOLERANCE=0.01` – ROI delta required to stop early
- `RUN_CYCLES=0` – unlimited run cycles for the orchestrator
- `AUTO_DASHBOARD_PORT=8001` – start the metrics dashboard
- `VISUAL_AGENT_TOKEN=<secret>` – authentication token for `menace_visual_agent_2.py`
- `VISUAL_AGENT_AUTOSTART=1` – automatically launch the visual agent when missing
- `VISUAL_AGENT_SSL_CERT` – optional path to an SSL certificate for HTTPS
- `VISUAL_AGENT_SSL_KEY` – optional path to the corresponding private key

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

   The script verifies system dependencies, creates default presets using `environment_generator` and invokes the sandbox runner. The metrics dashboard is available at `http://localhost:${AUTO_DASHBOARD_PORT}` once started.
   If the previous run terminated unexpectedly you can append `--recover` to reload the last recorded ROI and synergy histories.

## Visual agent queueing

`menace_visual_agent_2.py` only processes **one connection at a time**. When the
`/run` endpoint returns HTTP `409` the agent is busy with another task. Keep a
local queue of requests and retry them once `/status` reports `{"active": false}`.
The `/status` response also includes the length of the internal queue so you can
monitor progress. This behaviour avoids race conditions in the underlying visual
pipeline.

## Local run essentials

The sandbox reads several paths and authentication tokens from environment variables. These defaults are suitable for personal deployments and can be overridden in your `.env`:

- `VISUAL_AGENT_TOKEN` – shared secret for the visual agent service.
- `VISUAL_AGENT_SSL_CERT` – path to the TLS certificate served by
  `menace_visual_agent_2.py` when running over HTTPS.
- `VISUAL_AGENT_SSL_KEY` – path to the private key for the certificate.
- `SANDBOX_DATA_DIR` – directory where ROI history, presets and patch records are stored. Defaults to `sandbox_data`.
- `PATCH_SCORE_BACKEND_URL` – optional remote backend for patch scores. Supports `http://`, `https://` or `s3://bucket/prefix` URLs.
- `DATABASE_URL` – connection string for the primary database. Defaults to `sqlite:///menace.db`.
- `BOT_DB_PATH` – location of the bot registry database, default `bots.db`.
- `BOT_PERFORMANCE_DB` – path for the performance history database, default `bot_performance_history.db`.
- `MAINTENANCE_DB` – SQLite database used for maintenance logs, default `maintenance.db`.

### Example `.env`

```dotenv
VISUAL_AGENT_TOKEN=my-secret-token
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

Long running services rotate their own log files such as
`service_supervisor.py` which keeps up to three 1&nbsp;MB archives. Rotate or
clean old logs periodically when persisting them on disk.

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
- **Visual agent returns 409** – the service only accepts one request at a time.
  Wait until `/status` shows `{"active": false}` or queue the job for later.
- **Dashboard not loading** – confirm that `AUTO_DASHBOARD_PORT` is free and no
  firewall blocks the connection. The dashboard starts automatically once the
  sandbox loop begins.
