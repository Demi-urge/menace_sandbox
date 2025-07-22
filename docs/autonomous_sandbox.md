# Autonomous Sandbox

This guide describes the prerequisites and environment variables used when running the fully autonomous sandbox.

## System packages

The sandbox relies on several system utilities in addition to the Python dependencies listed in `pyproject.toml`:

- `ffmpeg` – audio extraction for video clips
- `tesseract-ocr` – OCR based bots
- `chromium-browser` or any Chromium based browser for web automation
- `qemu-system-x86` – optional virtualization backend used for cross platform presets

Most of these packages are installed automatically when using the provided Dockerfile. On bare metal they must be installed manually via your package manager.

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

   The script verifies system dependencies, creates default presets using `environment_generator` and invokes the sandbox runner. The metrics dashboard is available at `http://localhost:${AUTO_DASHBOARD_PORT}` once started.

## Local run essentials

The sandbox reads several paths and authentication tokens from environment variables. These defaults are suitable for personal deployments and can be overridden in your `.env`:

- `VISUAL_AGENT_TOKEN` – shared secret for the visual agent service.
- `VISUAL_AGENT_SSL_CERT` – path to the TLS certificate served by
  `menace_visual_agent_2.py` when running over HTTPS.
- `VISUAL_AGENT_SSL_KEY` – path to the private key for the certificate.
- `SANDBOX_DATA_DIR` – directory where ROI history, presets and patch records are stored. Defaults to `sandbox_data`.
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
DATABASE_URL=sqlite:///menace.db
BOT_DB_PATH=bots.db
BOT_PERFORMANCE_DB=bot_performance_history.db
MAINTENANCE_DB=maintenance.db
```

## Docker usage

Build the sandbox image and run it inside a container using the helper scripts:

```bash
scripts/build_sandbox_image.sh
scripts/run_sandbox_container.sh
```

The run script mounts `sandbox_data/` for persistent metrics and loads
environment variables from `.env` so the container behaves the same as a
local installation.
