# Sandbox Environment Setup

This guide outlines the base configuration required to run the Menace sandbox.

## Required environment variables

The sandbox expects a few variables to be set before launch:

- `SANDBOX_REPO_PATH` – path to the repository root.
- `SANDBOX_DATA_DIR` – directory for runtime state and metrics.
- `DATABASE_URL` – database connection string.

Run `auto_env_setup.ensure_env()` to generate a `.env` file with these variables when
missing. Values may also be supplied via the shell environment.

## Dynamic path routing

Use `dynamic_path_router.resolve_path` for all repository file lookups so paths
remain portable. The resolver infers the project root from Git metadata and can
be overridden by setting `MENACE_ROOT` or `SANDBOX_REPO_PATH`. Combine it with
`SANDBOX_DATA_DIR` when accessing runtime artifacts:

```python
import os
from dynamic_path_router import resolve_path
data_dir = resolve_path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data"))
```

## Prompt logging variables

These optional variables control where prompt execution results are stored:

- `PROMPT_SUCCESS_LOG_PATH` – path for successful prompt logs (default `sandbox_data/prompt_success_log.jsonl`).
- `PROMPT_FAILURE_LOG_PATH` – path for failed prompt logs (default `sandbox_data/prompt_failure_log.jsonl`).

## Container execution variables

- `CONTAINER_SNIPPET_PATH` – path inside the container where snippets are
  executed (default `/code/snippet.py`).

## Optional dependencies

Some features rely on additional tools. Missing components degrade
functionality but do not prevent basic operation.

- **System tools**: `ffmpeg`, `tesseract`, `qemu-system-x86_64`
- **Python packages**: `matplotlib`, `statsmodels`, `uvicorn`, `fastapi`,
  `sklearn`, `httpx`

## Directory layout

A typical checkout contains the following directories:

```
menace_sandbox/
├── configs/         # preset configuration files
├── logs/            # runtime logs
├── sandbox_data/    # metrics, databases and generated presets
└── scripts/         # helper scripts
```

## Minimal `SandboxSettings` example

Secrets should be sourced through `SecretsManager` to avoid committing tokens to
version control:

```python
from secrets_manager import SecretsManager
from sandbox_settings import SandboxSettings
from dynamic_path_router import resolve_path
import os

secrets = SecretsManager()
settings = SandboxSettings(
    sandbox_repo_path=resolve_path("."),
    sandbox_data_dir=resolve_path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data")),
)
```

## Initialization order

1. `auto_env_setup.ensure_env()` – create or load the `.env` file.
2. `initialize_autonomous_sandbox(settings)` – verify dependencies and prepare
   services.
3. `start_self_improvement_cycle(workflows)` – launch the optimisation loop.

## Troubleshooting

- **Missing dependencies** – run `./setup_env.sh` or install packages listed in
  `requirements.txt`. Ensure `ffmpeg` and `tesseract` are available on
  `$PATH`.
- **Misconfiguration** – delete the generated `.env` and rerun
  `auto_env_setup.ensure_env()`. Verify that `SANDBOX_REPO_PATH` and
  `SANDBOX_DATA_DIR` point to valid locations.
- **Secrets not found** – confirm `secrets.json` exists or call
  `SecretsManager.set()` to seed required entries.
