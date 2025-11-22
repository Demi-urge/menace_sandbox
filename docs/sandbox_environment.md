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

## Temporary manager example for maintenance scripts

Ad-hoc maintenance often requires running scripts like `bootstrap_self_coding.py` or bespoke vacuum jobs directly from a shell where no `SelfCodingManager` has been initialised yet. Wrap `ModelAutomationPipeline` creation in `coding_bot_interface.prepare_pipeline_for_bootstrap` so helpers receive a sentinel manager until the real manager is ready:

```python
from coding_bot_interface import prepare_pipeline_for_bootstrap
from model_automation_pipeline import ModelAutomationPipeline
from self_coding_manager import SelfCodingManager
from vector_service.context_builder import ContextBuilder
from bot_registry import BotRegistry
from data_bot import DataBot

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
registry = BotRegistry()
data_bot = DataBot(start_server=False)

pipeline, promote_manager = prepare_pipeline_for_bootstrap(
    pipeline_cls=ModelAutomationPipeline,
    context_builder=builder,
    bot_registry=registry,
    data_bot=data_bot,
)

# Default progress heartbeats during pipeline bootstrap can be tuned via
# PREPARE_PIPELINE_PROGRESS_HEARTBEAT to keep long-running steps alive even when
# the underlying helper is quiet.

# ...run maintenance scripts that expect pipeline.manager to exist...

manager = SelfCodingManager(pipeline=pipeline, bot_registry=registry, data_bot=data_bot)
promote_manager(manager)
```

The sentinel manager returned during construction prevents nested helpers from trying to bootstrap themselves (which would trigger the "re-entrant initialisation depth" warning) while still letting manual scripts inspect or patch the pipeline before the concrete manager is promoted.

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
