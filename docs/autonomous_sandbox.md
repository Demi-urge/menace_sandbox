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

