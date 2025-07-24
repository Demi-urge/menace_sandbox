- Revenue tracking and monetisation helpers
- **Profit density evaluation** keeping only the most lucrative clips
- Intelligent culling for clips, accounts and topics
- Scoutboard topic log with historical performance
- Prediction engine for emerging topics
- Dynamic account redistribution based on profit density
- Automated reinvestment of profits via `investment_engine.AutoReinvestmentBot` with a predictive spend engine ([docs/auto_reinvestment.md](docs/auto_reinvestment.md))
- Bottleneck detection via performance monitors ([docs/bottleneck_detection.md](docs/bottleneck_detection.md))
- Adaptive energy score guiding resource allocation ([docs/energy_score.md](docs/energy_score.md))
- Central Menace orchestrator coordinating all stages and hierarchical oversight
- Self improvement engine automatically runs the workflow on the Menace model when metrics degrade
- Self-coding manager applies patches then deploys via the automation pipeline
- Evolution orchestrator coordinating self improvement and structural evolution
- System evolution manager runs GA-driven structural updates ([docs/system_evolution_manager.md](docs/system_evolution_manager.md))
- Experiment manager for automated A/B testing of bot variants
- Neuroplasticity tracking via PathwayDB ([docs/neuroplasticity.md](docs/neuroplasticity.md))
- Atomic write mirroring with `TransactionManager`
- Remote database replication via `DatabaseRouter(remote_url=...)`
- Change Data Capture events published to `UnifiedEventBus`
- Optional RabbitMQ integration via `UnifiedEventBus(rabbitmq_host=...)`
- Schema migrations managed through Alembic
- Long-term metrics dashboards with Prometheus ([docs/metrics_dashboard.md](docs/metrics_dashboard.md))
- Workflow benchmarking metrics exported to Prometheus with automatic early stopping when improvements level off. Additional gauges track CPU time, memory usage, network and disk I/O with statistical significance tests.
- `metrics_exporter` tries to install `prometheus_client` during bootstrap and
  serves a fallback HTTP endpoint if the install fails
- Centralised logging via Elasticsearch or Splunk and optional Sentry alerts ([docs/monitoring_pipeline.md](docs/monitoring_pipeline.md))
- Manual safe mode and override flags for risk-free operation
- Daily and weekly budget enforcement via `CapitalManagementBot` ([docs/capital_management.md](docs/capital_management.md))
- Military grade error management helpers ([docs/military_error_handling.md](docs/military_error_handling.md))
- Optional systemd unit for auto-start on boot (`systemd/menace.service`)
- Dependency updates also rebuild container images
- Offline caches for trend scanning and LLM prompts
- Improved summarisation fallback for `TextResearchBot` using NLTK and TF-IDF
- Candidate matcher now falls back to a built-in TF-IDF implementation when
  `scikit-learn` is unavailable. The fallback maintains its own corpus for IDF
  weighting and computes cosine similarity directly.
- Continuous chaos testing with automatic rollback
- Self-hosted model evaluation service for autonomous redeploys
- Distributed benchmarking via `UnifiedEventBus` ([docs/distributed_benchmarking.md](docs/distributed_benchmarking.md))
- Continuous infrastructure auto-provisioning with `EnvironmentBootstrapper`
- Hands-free OS and container updates with staged rollbacks via `UnifiedUpdateService`
- Continuous compliance auditing with `ComplianceChecker`
- Telemetry-driven debugging loop using `DebugLoopService`
- Self-deploying bootstraps across remote hosts
- Automatic environment setup via `auto_env_setup.ensure_env`
- Docker-based dependency provisioning with `ExternalDependencyProvisioner`
- Provisioning failures publish `dependency:provision_failed` events and are retried automatically
- Periodic backups restored through `DisasterRecovery`
- Distributed service supervision with `ClusterServiceSupervisor`
- Instant workflow diversification triggered by microtrend signals
- Automated secret rotation handled by `SecretsManager`
 - Automatic API key retrieval via `auto_env_setup.interactive_setup` without prompts
- Background `SecretRotationService` runs when `AUTO_ROTATE_SECRETS=1`\
  and rotates the comma separated names from `ROTATE_SECRET_NAMES` at the
  interval defined by `SECRET_ROTATION_INTERVAL`
- Automated review of flagged bots via `AutomatedReviewer`
- Self-provisioning of missing packages through `SystemProvisioner`
- Distributed rollback verification via `RollbackValidator`
- ROI-driven autoscaling with `ROIScalingPolicy`
- ROI history forecasting via `ROITracker` ([docs/roi_tracker.md](docs/roi_tracker.md))
- Synergy-aware environment presets adapt CPU, memory, bandwidth and threat levels ([docs/environment_generator.md](docs/environment_generator.md))
- Sandboxed self-debugging using `SelfDebuggerSandbox` (invoked by `launch_menace_bots.py` after the test run)
- Comprehensive build pipeline in `launch_menace_bots.py` that plans,
  develops, tests and scales bots before deployment
- Automated implementation pipeline turning tasks into runnable bots ([docs/implementation_pipeline.md](docs/implementation_pipeline.md))
- Models repository workflow with visual agents ([docs/models_repo_workflow.md](docs/models_repo_workflow.md))
- Retirement of underperforming models by `ModelPerformanceMonitor`
- External dependency monitoring and failover via `DependencyWatchdog`
- Hardware discovery sets `NUM_GPUS` and `GPU_AVAILABLE` automatically
- Cross-platform service installation via `service_installer.py`
- Running `menace_master.py` as root installs the service automatically
- Background updates handled by `UnifiedUpdateService` even without the supervisor
- Automatic first-run sandbox improving the codebase before live execution

See [docs/quickstart.md](docs/quickstart.md) for a Quickstart guide on launching the sandbox.
Detailed environment notes are available in [docs/autonomous_sandbox.md](docs/autonomous_sandbox.md).
A new `menace` CLI wraps common workflows so you no longer need to remember individual scripts.

## Self-Optimisation Loop

1. **Monitor metrics** – `DataBot` tracks ROI, errors and energy scores and
   exposes `long_term_roi_trend()` to measure how ROI changes over time.
2. **Trigger improvement** – when metrics degrade, `EvolutionOrchestrator`
   decides between running `SelfImprovementEngine` or invoking the
   `SystemEvolutionManager`.
3. **Self-coding** – during improvement cycles the `SelfCodingManager`
   applies patches and redeploys updated bots.
4. **Predictive guidance** – `EvolutionPredictor` forecasts ROI impact for
   each action and influences the orchestrator's decisions.
5. **Strategic planning** – `StrategicPlanner` refines long term objectives and
   allocates resources automatically.
5. **Structural updates** – if a structural change is required,
   `WorkflowEvolutionBot` proposes new sequences which are tested via the
   experiment manager.
6. **Deployment** – successful bots are redeployed and metrics are logged back
   into `PathwayDB`, closing the loop.

## Installation

Run the setup helper to install the required system tools and Python
dependencies.  When ``MENACE_OFFLINE_INSTALL=1`` is set the script
installs packages from ``MENACE_WHEEL_DIR`` instead of contacting PyPI.

```bash
scripts/setup_autonomous.sh
```

For a completely automated setup that also runs the tests, bootstraps the
environment and generates presets run:

```bash
python scripts/autonomous_setup.py
```
The helper auto-discovers available GPUs and network interfaces and stores the
information in ``hardware.json``.

After the base requirements are installed you can bootstrap optional
dependencies and verify your configuration with:

```bash
python scripts/bootstrap_env.py
```

## Getting Started

Launch the autonomous sandbox with the default environment presets:

```bash
scripts/start_autonomous.sh
```

The helper creates a `.env` file with safe defaults via `auto_env_setup.ensure_env` and then
starts `run_autonomous.py`.

To automatically start a local visual agent before running the sandbox use:

```bash
python scripts/launch_personal.py
```
Alternatively start the agent on demand and run the sandbox with:
```bash
python scripts/run_personal_sandbox.py
```

To run the sandbox directly with metrics visualisation, use the convenience
script which installs dependencies via `setup_env.sh` before launching the
dashboard:

```bash
scripts/start_sandbox.sh
```

To visualise synergy metrics separately, run:

```bash
python -m menace.self_improvement_engine synergy-dashboard --wsgi flask
```

Replace `flask` with `gunicorn` or `uvicorn` to use a different server.

### Required dependencies

The following Python packages must be available. They are installed
automatically when building the Docker image or running ``pip install -e .``:

```
moviepy
pytube
selenium
pyautogui
pytesseract
requests
opencv-python
numpy
filelock
undetected-chromedriver
selenium-stealth
scikit-learn
beautifulsoup4
scipy
PyPDF2
gensim
SpeechRecognition
gTTS
SQLAlchemy
alembic
psycopg2-binary
pandas
fuzzywuzzy[speedup]
marshmallow
celery
pyzmq
pika
psutil
risky
networkx
pulp
Flask
PyYAML
GitPython
pymongo
elasticsearch
Faker
docker
boto3
prometheus-client
deap
simpy
matplotlib
Jinja2
playwright
fastapi
pydantic
redis
sentence-transformers
hdbscan
annoy
libcst
kafka-python
pyspark
stable-baselines3
torch
sentry-sdk
```

For convenience, a `setup_env.sh` script installs dependencies and pytest. To
prepare a fresh environment for running the test suite in one step run:

```bash
./setup_env.sh && scripts/setup_tests.sh
```

Afterwards execute `pytest` (optionally passing specific test files) as usual.

### Hardware tests

The `tests/hardware` suite uses stubbed serial and GPIO interfaces. These tests
are skipped unless `MENACE_HARDWARE=1` is set:

```bash
MENACE_HARDWARE=1 pytest tests/hardware
```

### Optional dependencies

Some features such as anomaly detection and the `ErrorForecaster` make use of
additional libraries. **pandas** and **PyTorch** are now installed by default,
yet the modules still implement fallbacks so functionality remains intact even
if those libraries are unavailable.

### Cloud deployment

Set `DATABASE_URL` to point to your cloud database (for example
`postgresql://user:pass@host/db`). `TERRAFORM_DIR` may reference a directory with
Terraform manifests used during deployment.  When `AUTOSCALER_ENDPOINT` is
defined, scaling actions will POST to that HTTP service. Set
`AUTOSCALER_PROVIDER=kubernetes` or `swarm` to manage a Kubernetes deployment or
Docker Swarm service instead of spawning local processes. Optional
`K8S_DEPLOYMENT` and `SWARM_SERVICE` can override the target names.

The production container also requires several system utilities that are not
installed by default. Install **ffmpeg** for audio extraction, **tesseract-ocr**
for OCR based bots and a Chromium browser for web automation. The provided
`Dockerfile` now installs these packages automatically.

### Building containers

DeploymentBot automatically builds Docker images from this repository's
`Dockerfile` whenever you deploy. Manual image creation is optional, but can be
performed with:

```bash
docker build -t <bot>:latest .
```

### Docker Compose setup

Use the provided `docker-compose.yml` to launch the autonomous stack:

```bash
docker compose up --build
```

QEMU images can be mounted by placing them in `./qemu_images` and adding:

```
VM_SETTINGS={"windows_image":"/vm-images/windows.qcow2","macos_image":"/vm-images/macos.qcow2","memory":"4G"}
```

to your `.env` file. `run_autonomous.py` reads this configuration automatically.

### Container health checks

The provided `Dockerfile` now defines a `HEALTHCHECK` that executes
`python -m menace.startup_health_check`. This validates essential files and
configuration before the container is marked healthy.


### Production configuration

Set `MENACE_MODE=production` and provide a PostgreSQL `DATABASE_URL` for
production deployments. The application will exit if a SQLite URL is used in
production mode.

### RabbitMQ event bus

To share events across machines, run a RabbitMQ server and provide its host to
`UnifiedEventBus`:

```python
from menace.unified_event_bus import UnifiedEventBus
bus = UnifiedEventBus(rabbitmq_host="localhost")
```

All producers and consumers now operate transparently with either the in-memory
or networked bus.


### Automated review flow

Calling ``UnifiedEventBus.flag_for_review(bot_id)`` immediately invokes the
configured ``AutomatedReviewer``. When ``severity="critical"`` the reviewer
delegates to ``AutoEscalationManager`` and disables the bot.

## Environment Configuration

Application settings are read from environment variables. ``auto_env_setup.ensure_env``
creates a ``.env`` file on first run so Menace can start without manual
configuration. The file contains keys defined in ``env_config.py`` such as
``DATABASE_URL`` and various API credentials like ``OPENAI_API_KEY`` or
``SERP_API_KEY``. Additional service specific keys (for example
``STRIPE_API_KEY`` or ``REDDIT_CLIENT_SECRET``) can be added to the same file.
Set ``MENACE_ENV_FILE`` to load variables from a different path or call
``auto_env_setup.ensure_env("custom.env")`` to generate it elsewhere.

### Environment variables

The deployment helpers use the following variables:

- `CLUSTER_HOSTS` – initial nodes managed by `ClusterServiceSupervisor`.
- `NEW_HOSTS` – comma separated list of freshly provisioned nodes. When set,
  `Autoscaler` bootstraps them via `EnvironmentBootstrapper` and
  `ClusterServiceSupervisor` starts supervisors on each host. The variable is
  cleared after processing.
- `FAILOVER_HOSTS` – fallback hosts started when the primary ones fail.
- `REMOTE_HOSTS` – hosts receiving the bootstrap script during startup.
- `AUTOSCALER_PROVIDER` – scaling backend: `local`, `kubernetes` or `swarm`.
- `K8S_DEPLOYMENT` – target deployment when using the Kubernetes provider.
- `SWARM_SERVICE` – target service when using the Docker Swarm provider.
- `PROMPT_TEMPLATES_PATH` – path to `prompt_templates.v1.json` when running

  outside the repository.
- `METRICS_PORT` – start the internal metrics exporter on this port.
- `AUTO_DASHBOARD_PORT` – start the metrics dashboard automatically on this port.
- `SANDBOX_RESOURCE_DB` – path to a `ROIHistoryDB` for resource-aware forecasts.
- `SANDBOX_BRAINSTORM_INTERVAL` – request GPT-4 brainstorming ideas every N cycles.
- `SANDBOX_BRAINSTORM_RETRIES` – consecutive low-ROI cycles before brainstorming.
- `SANDBOX_OFFLINE_SUGGESTIONS` – enable heuristic patches when GPT is unavailable.
- `SANDBOX_SUGGESTION_CACHE` – JSON file with cached suggestions.
- `PATCH_SCORE_BACKEND_URL` – optional patch score storage. Use `file://path` to
  save scores locally.
- `SANDBOX_PRESET_RL_STRATEGY` – RL algorithm used by `AdaptivePresetAgent` (default `q_learning`).
- `OPENAI_FALLBACK_MODEL` – fallback GPT model name.
- `OPENAI_FALLBACK_ATTEMPTS` – retry count for OpenAI calls.
- `OPENAI_RETRY_DELAY` – delay in seconds between OpenAI retries.


`auto_env_setup.ensure_env` writes sensible defaults when these variables are
missing. Notable defaults include:

- `DATABASE_URL=sqlite:///menace.db`
- `MODELS=demo`
- `MODELS_REPO_URL=https://github.com/Demi-urge/models`
- `MODELS_REPO_PUSH_URL=`
- `SLEEP_SECONDS=0` (run cycles without waiting)
- `AUTO_BOOTSTRAP=1`
- `AUTO_UPDATE=1`
- `UPDATE_INTERVAL=86400`
- `OVERRIDE_UPDATE_INTERVAL=600`
- `AUTO_BACKUP=0`
- `AUTO_SANDBOX=1`
- `SANDBOX_ROI_TOLERANCE=0.01`
- `SANDBOX_CYCLES=5`
- `SANDBOX_DATA_DIR=sandbox_data`
- `RUN_CYCLES=0`
- `RUN_UNTIL=`
- `METRICS_PORT=8001`
- `AUTO_DASHBOARD_PORT=`
- `SANDBOX_BRAINSTORM_INTERVAL=0`
- `SANDBOX_BRAINSTORM_RETRIES=3`
- `SANDBOX_OFFLINE_SUGGESTIONS=0`
- `SANDBOX_SUGGESTION_CACHE=`
- `PATCH_SCORE_BACKEND_URL=`
- `OPENAI_FALLBACK_MODEL=gpt-4o-mini`
- `OPENAI_FALLBACK_ATTEMPTS=3`
- `OPENAI_RETRY_DELAY=1.0`
- `AD_API_URL=` (unset or empty disables ad network integration)

The sentinel file used to detect the initial launch defaults to `.menace_first_run` and can be overridden with the `MENACE_FIRST_RUN_FILE` environment variable.
The models repository defaults to `https://github.com/Demi-urge/models` and can be overridden with the `MODELS_REPO_URL` environment variable.
When `MODELS_REPO_PUSH_URL` is set, `clone_to_new_repo` pushes the clone to a remote origin named after the model ID under this base URL.
Set `MENACE_SANDBOX=1` (or pass `--sandbox`) to run the sandbox manually even after the first run has completed.

Override any value via the command line using ``menace_master.py --env VAR=VALUE``
or by editing the generated ``.env`` file.

``auto_env_setup.interactive_setup`` obtains missing API keys automatically via
``SecretsManager`` and optional vault providers. It loads defaults from the file
referenced by ``MENACE_DEFAULTS_FILE`` and only prompts for values that remain
unset. Answers can be pre-filled with environment variables named
``MENACE_SETUP_<KEY>`` (e.g. ``MENACE_SETUP_OPENAI_API_KEY``).

### First-run sandbox

When Menace launches for the first time it clones itself into a temporary
directory and performs a full cycle under ``SelfDebuggerSandbox`` before any
long‑running services start. All persistent stores are redirected to temporary
paths so the sandbox run does not pollute production data. During this phase
``DATABASE_URL``, ``BOT_DB_PATH``, ``BOT_PERFORMANCE_DB`` and ``MAINTENANCE_DB``
point into the sandbox directory and the event bus uses an ephemeral database.
This check is only
executed when the environment variable ``AUTO_SANDBOX`` is enabled (the default
``1``) and the sentinel file does not yet exist. ``menace_master.py`` first runs
``run_once`` to verify the setup, then iterates through ``SANDBOX_CYCLES``
self‑improvement loops. The loop stops early when the change in ROI between
iterations is below ``SANDBOX_ROI_TOLERANCE`` (defaults to ``0.01``). After the
sandbox completes successfully, Menace writes the sentinel file so subsequent
launches skip this phase. The file defaults to ``.menace_first_run`` in the
repository root and its location can be changed via ``MENACE_FIRST_RUN_FILE``.
Delete the file to force the sandbox to run again. The sandbox copies
``improvement_policy.pkl`` and ``patch_history.db`` from ``SANDBOX_DATA_DIR``
before starting and writes the updated versions back when finishing so the
learning state persists between runs.
Sandbox execution helpers live in ``sandbox_runner.py`` with the main entry point
``_run_sandbox`` used by ``menace_master.py``.

For a completely autonomous optimisation loop run:

```
python run_autonomous.py
```

The wrapper verifies that Docker, QEMU and the required Python packages are
available, generates default presets via ``environment_generator`` and then
invokes the ``full-autonomous-run`` loop from ``sandbox_runner.py``. Final
module rankings and the last metric values are printed on completion. Should
``_sandbox_main`` crash during a cycle the ``SandboxRecoveryManager`` wraps the
function and automatically restarts it so ROI history and intermediate data
persist across failures.

When you want to reproduce a specific scenario pass preset files or JSON strings
directly to ``sandbox_runner.py`` via the ``run-complete`` subcommand:

```
python sandbox_runner.py run-complete presets.json --max-iterations 1
```

The command forwards the presets to ``full_autonomous_run`` and starts the
dashboard when ``--dashboard-port`` (or ``AUTO_DASHBOARD_PORT``) is supplied.

When ``--auto-thresholds`` is enabled the loop recomputes the ROI and synergy
thresholds each iteration. The thresholds are derived from the rolling standard
deviation of recent metrics so manual ``--roi-threshold`` or
``--synergy-threshold`` values are unnecessary.

To execute multiple runs sequentially and launch the metrics dashboard run:

```
python run_autonomous.py --runs 2 --preset-count 2 --dashboard-port 8002

# or simply set AUTO_DASHBOARD_PORT=8002
```

To optimise across distinct scenarios supply preset files explicitly. The
runner cycles through them when ``--preset-file`` is repeated:

```
python run_autonomous.py --runs 3 \
  --preset-file presets/dev.json \
  --preset-file presets/prod.json \
  --preset-file presets/chaos.json
```

This prints messages such as ``Starting autonomous run 1/2`` followed by the
standard module rankings once each run finishes. Metrics from all runs are
written to ``sandbox_data/roi_history.json`` so they can be aggregated later.

### Advanced sandbox commands

- ``--auto-thresholds`` recomputes ROI and synergy thresholds every cycle so
  manual ``--roi-threshold`` and ``--synergy-threshold`` values are optional.
  Fine‑tune synergy convergence with ``--synergy-threshold-window`` and
  ``--synergy-threshold-weight``.
- ``menace_visual_agent_2.py --recover-queue`` reloads tasks from the persisted
  queue after a restart. Use ``--flush-queue`` to drop stalled entries.
- ``menace_visual_agent_2.py --auto-recover`` also restores queued tasks on
  startup and records metrics in ``sandbox_data/visual_agent_recovery.json``.
- ``menace_visual_agent_2.py --cleanup`` removes stale lock and PID files then exits.
- Inspect sandbox restart metrics via ``sandbox_recovery_manager.py --file
  sandbox_data/recovery.json``.

Troubleshooting tips:

- Run ``./setup_env.sh && scripts/setup_tests.sh`` when tests fail to ensure
  all dependencies are installed.
- Delete ``sandbox_data/recovery.json`` if ``SandboxRecoveryManager`` keeps
  restarting unexpectedly.
- If synergy metrics diverge wildly, verify that ``synergy_history.db`` is
  writable and consider adjusting ``--synergy-threshold-weight``.
- HTTP 401 errors from the visual agent usually mean ``VISUAL_AGENT_TOKEN`` is
  missing or mismatched. Confirm the token matches the secret configured in
  ``menace_visual_agent_2.py``.
QEMU must be installed separately for cross-platform tests. Place your QCOW2 files in `qemu_images` and reference them via `VM_SETTINGS` so presets with `OS_TYPE` `windows` or `macos` boot automatically.

### Maintenance logs and audit signing

- `MAINTENANCE_DB` – path for the SQLite maintenance log database. Defaults to
  `maintenance.db` when unset.
- `MAINTENANCE_DB_URL` – optional PostgreSQL connection string used instead of
  SQLite. The bot falls back to SQLite if the connection fails.
- `AUDIT_PRIVKEY` – base64-encoded Ed25519 private key used to sign audit
  entries. Both raw 32-byte keys and DER-encoded keys are supported. The
  following command generates a DER-encoded key that can be used directly:

```bash
openssl genpkey -algorithm Ed25519 | \
  openssl pkey -outform DER | base64 -w 0
```

### Mandatory environment variables

Generate the initial environment file by calling ``auto_env_setup.ensure_env()``.
The function writes a ``.env`` file containing the keys below so Menace can
start unattended.  At minimum the following variables must be defined:

- ``MAINTENANCE_DISCORD_WEBHOOK`` or ``MAINTENANCE_DISCORD_WEBHOOKS`` – Discord
  webhook(s) for maintenance notifications. Set to the webhook URL
  ``https://discordapp.com/api/webhooks/1386593502602723348/I--gtkFFi0m9ZCToGCCAP0PswtKA0m7UYsp3UQSqZU0qDGkbk-1W8zQS8g_zuCBdCcXf`` to
  receive alerts.
- ``WEEKLY_METRICS_WEBHOOK`` – Discord webhook used by ``WeeklyMetricsBot``.
  Defaults to ``https://discord.com/api/webhooks/PLACEHOLDER`` when unset.
- ``DATABASE_URL`` – connection string for the persistent database. When
  ``MENACE_MODE=production`` this must point to a production-ready database
  rather than the default SQLite file.
- ``CELERY_BROKER_URL`` – message broker used by asynchronous maintenance tasks.
  Required when running with Celery in production.
- ``OPENAI_API_KEY`` – API key enabling language model features used by
  ``BotDevelopmentBot`` and other helpers.
- ``STRIPE_API_KEY`` – required for revenue tracking and monetisation helpers.
- ``VISUAL_AGENT_TOKEN`` – shared secret used by ``menace_visual_agent_2.py`` for authentication.
- ``SANDBOX_REPO_PATH`` – path to the local sandbox repository clone processed during self-improvement cycles.
- ``SANDBOX_DATA_DIR`` – directory storing ROI history, presets and patch metrics.

## Usage

Most bots can be executed individually. The project also exposes a convenience
entry point that launches the service supervisor managing all background
services:

```bash
menace-master
```

Running `menace_master.py` with `USE_SUPERVISOR=1` will launch the same
supervisor so everything can be started via a single command.

`menace_master` operates entirely headlessly.  The optional Tkinter interface
provided by `menace_gui` is useful for manual inspection but is not required for
the autonomous workflow.

The supervisor keeps these services running and restarts them automatically.
It also runs the `SelfEvaluationService` which combines microtrend detection
with workflow cloning so new trends immediately spawn variant bots.
The orchestrator repeats the automation workflow until you stop it with
``Ctrl+C``. Set the ``SLEEP_SECONDS`` environment variable to delay the next
cycle (use ``0`` for continuous execution). Target models can be supplied via
``MODELS`` as a comma separated list. Use ``RUN_CYCLES`` to stop after a fixed
number of cycles or ``RUN_UNTIL`` to stop at a Unix timestamp.

### Persistent service

Install Menace as a background service using `service_installer.py`:

```bash
python service_installer.py
```

Running `menace_master.py` performs the same installation automatically. With
root privileges a system-wide service is installed while without elevation it
falls back to a user level systemd unit on Linux/macOS or a Task Scheduler entry
on Windows.

On Linux/macOS this installs a systemd unit. When running without root the unit
is placed under `~/.config/systemd/user`. On Windows it registers a service via
`sc create` when elevated or creates a scheduled task otherwise. Pass
`--orchestrator k8s` or `swarm` to generate
Kubernetes or Docker Swarm manifests:

```bash
python service_installer.py --orchestrator k8s  # writes menace-deployment.yaml
# kubectl apply -f menace-deployment.yaml

python service_installer.py --orchestrator swarm  # writes docker-compose.yml
# docker stack deploy -c docker-compose.yml menace
```

### Visual agent

`menace_visual_agent_2.py` exposes a FastAPI service used during
development. The service must always run with a single worker:

```python
uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT, workers=1)
```
Only one connection is processed at a time. If `/run` returns `409` queue the
request and retry once `/status` reports the agent idle.

The server enforces this single-connection policy via a global lock. Any
additional `/run` request made while a job is active immediately receives HTTP
409.  The orchestrator automatically serialises requests by wrapping the client
in a ``VisualAgentJobQueue`` so multiple workflows can safely share the agent.

Set the environment variable ``VISUAL_TOKEN_REFRESH_CMD`` to a shell command returning a fresh token. ``VisualAgentClient`` runs the command automatically when authentication fails.

``VISUAL_AGENT_TOKEN`` **must** be set and is hashed for comparison. Requests may provide a ``Bearer`` token via the ``Authorization`` header or the legacy ``x-token`` field.

Set ``VISUAL_AGENT_AUTO_RECOVER=1`` to launch the service with
``--auto-recover`` so queued tasks persist across restarts.

``menace_visual_agent_2.py`` automatically sets ``pytesseract.tesseract_cmd``
based on the host OS. Override the path with ``TESSERACT_CMD`` if needed.

### Topic discovery and clip acquisition

Two new bots automate topic research and raw clip gathering:

``clipped/topic_discovery.py`` scrapes trending sources such as YouTube,
TikTok, X, Google Trends, ProductHunt, GitHub and Reddit to build ``data/topics/topics.json``. The
``TopicScraper`` class exposes ``update_topics()`` which pulls the latest
keywords while always injecting ``Balolos`` as a seed topic.

``clipped/clip_downloader.py`` reads those topics and downloads top videos for
each keyword using ``yt-dlp`` into ``data/raw_clips``. Invoke
``ClipDownloader().download_clips()`` to fetch new material.

### Proxy management

Use the proxy manager CLI to acquire or update proxies. When run without
arguments it prints an available proxy:

```bash
python -m clipped.proxy_manager --file proxies.json
```

You can also release or mark a proxy as failed:

```bash
python -m clipped.proxy_manager --release 1.1.1.1:80
python -m clipped.proxy_manager --fail 1.1.1.1:80
```

### Clipping videos

Run the clipper to process downloaded videos into short clips. When invoked
without arguments, it uses the `videos` directory for input and writes clips to
`output_clips`. Both folders are created automatically if missing:

```bash
python -m clipped.clipper
```

The command reads videos from the `videos` directory and writes short clips
to `output_clips`. Both folders are created automatically if they don't
already exist.

### Synergy weight management

Synergy weights influence how the self‑improvement engine balances ROI,
efficiency, resilience and antifragility. Use the helper CLI to inspect or
persist these values:

```bash
python synergy_weight_cli.py show
python synergy_weight_cli.py export --out weights.json
python synergy_weight_cli.py import weights.json
python synergy_weight_cli.py reset
python synergy_weight_cli.py history --plot
```

See [docs/synergy_learning.md](docs/synergy_learning.md#customising-weights-with-synergy_weight_cli.py)
for a full walkthrough and
[the autonomous sandbox guide](docs/autonomous_sandbox.md#advanced-synergy-learning)
for an example using the deeper `DQNSynergyLearner`.

### Synergy metrics exporter

Set `EXPORT_SYNERGY_METRICS=1` when running `run_autonomous.py` to expose the
latest values from `synergy_history.db` via `SynergyExporter`. Legacy JSON files
are migrated automatically. The exporter
listens on `SYNERGY_METRICS_PORT` (default `8003`). Visit
`http://localhost:8003/metrics` (or your chosen port) to view the Prometheus
gauges.

### Dependency provisioning


External services defined in `docker-compose.yml` are now started
automatically by `menace_master` when they are missing.  The
`LocalInfrastructureProvisioner` generates a compose file with RabbitMQ,
Postgres and Vault when none exists and provisioning is retried
automatically.  Failures publish a `dependency:provision_failed` event on the
`UnifiedEventBus`.  The `scripts/provision_dependencies.py` helper remains
available for manual execution but is no longer required.

Configure endpoints via environment variables, for example:

```bash
export DEPENDENCY_ENDPOINTS="redis=http://localhost:6379,rabbit=http://localhost:15672"
export DEPENDENCY_BACKUPS="redis=http://backup:6379"
```
The watchdog checks each endpoint periodically (``WATCHDOG_INTERVAL`` seconds)
and switches to the backup URL when the primary becomes unavailable.
Press ``Ctrl+C`` to interrupt the script; any containers launched during
provisioning are shut down automatically.

### Error telemetry

`ErrorBot` now records detailed telemetry for every exception. An `ErrorLogger` middleware wraps bot entry points, capturing tracebacks, Codex API payloads and shell exit codes. Each incident is stored in a new `telemetry` table with fields for `task_id`, `bot_id`, `error_type`, `stack_trace`, `root_module`, `timestamp`, `resolution_status`, `patch_id` and `deploy_id`. A cascading classifier first applies regex rules to the stack trace and then optional SBERT similarity matching to assign semantic tags (e.g. "Runtime/Reference"). This structured taxonomy enables Menace to analyse spikes after dependency updates and preload remediation prompts.

### Bottleneck Detection Bots

Performance isn’t a luxury; it’s a multiplier on ROI. Each critical function is decorated with an `@perf_monitor` that records wall-clock time via `time.perf_counter` and stores `(function_signature, runtime_ms, cpu_pct, mem_mb)` into `PerfDB`. A nightly cron triggers `bottleneck_scanner.py`, ranking the P95 latencies per module. When a spike exceeds a configurable threshold, the scanner opens a Git issue via API, tagging the responsible bot and auto-assigning the Enhancement Bot.

Technically, this layer relies on `psutil` for cross-platform metrics and `sqlite3` for zero-setup storage. For heavier loads, export to Prometheus + Grafana so you can watch Menace’s pulse in real time. The scanner feeds its findings into the Resource Allocation Optimizer, ensuring slow code is either optimized or throttled.

### Meta-Logging & Replay Training

All inputs, outputs and decisions flow into Kafka topics with the prefix `menace.events.*`. A nightly Spark job (`ReplayTrainer`) aggregates sequences of `error` → `fix` → `success` and trains a lightweight gradient-boosted tree to predict failure likelihood from prompt features and context tokens. The resulting model updates the Prompt Rewriter so history informs future decisions.

For storage economy, raw JSON logs older than the retention window are compacted to Parquet on S3 while novel failures remain in PostgreSQL for quick access. The pipeline requires Kafka (or Redpanda) and a Spark environment like Databricks.

### Neuroplasticity & PathwayDB

Menace logs each workflow into `PathwayDB` and computes a **myelination score**.
Highly myelinated pathways trigger memory preloading, boost resource allocation
and raise the planning trust weight. See [docs/neuroplasticity.md](docs/neuroplasticity.md)
for details.

### Learning Engine

`LearningEngine` trains a logistic-regression model from `PathwayDB` metadata and
memory embeddings. Predictions can guide the self-improvement cycle to focus on
promising workflows. See [docs/learning_engine.md](docs/learning_engine.md).

`ActionLearningEngine` can also leverage Stable-Baselines3. When installed you can
select algorithms like `SAC` or `TD3` and pass hyperparameters through the
constructor. See [docs/action_learning_engine.md](docs/action_learning_engine.md)
for an example.

- `SelfLearningCoordinator` listens for `transactions:new` events so financial
  payouts influence model updates.
- New cross-query helpers `bot_roi_stats()` and `rank_bots()` aggregate ROI and
  CPU usage per bot for training and monitoring.

### Captcha Handling Pipeline

When automation hits a CAPTCHA challenge, the pipeline pauses instead of failing. `CaptchaDetector` scans page HTML (and optionally OCR on screenshots) for challenge markers. `CaptchaManager` snapshots the page to S3/MinIO, marks the job `BLOCKED` in Redis and then automatically attempts to solve the image using the configured anti-CAPTCHA service. Failures are retried with exponential backoff and a fallback Tesseract OCR pass is attempted until a token is obtained. The `CaptchaPipeline` replays the Playwright HAR with the solved token to continue the script seamlessly. Logged images accumulate a dataset for future machine‑learning solvers.

If the environment variable `ANTICAPTCHA_API_KEY` is set, snapshots are solved remotely; otherwise only local OCR is used. Either way, the retry strategy means most CAPTCHAs are handled automatically without human intervention.

- Database connectivity and event interactions (docs/connectivity.md)
- Bot heartbeat monitoring via `BotRegistry.record_heartbeat` ([docs/connectivity.md#bot-heartbeat-tracking](docs/connectivity.md#bot-heartbeat-tracking))

### ROI Trend Analysis

`DataBot.long_term_roi_trend()` compares early and late ROI to reveal long-term
performance drift. A positive number means ROI improved over the sampled
period, while a negative value indicates decline.

```python
from menace.data_bot import DataBot, MetricsDB

data_bot = DataBot(MetricsDB())
trend = data_bot.long_term_roi_trend(limit=100)
print(f"ROI trend: {trend:.2f}")
```

### Workflow ranking with BotCreationBot

`BotCreationBot` can take hints from `WorkflowEvolutionBot` and
`TrendingScraper`. Suggested sequences are ranked first and then reordered using
trending product names scraped from multiple sources:

```python
from menace.bot_creation_bot import BotCreationBot
from menace.workflow_evolution_bot import WorkflowEvolutionBot
from menace.trending_scraper import TrendingScraper

creator = BotCreationBot(
    workflow_bot=WorkflowEvolutionBot(),
    trending_scraper=TrendingScraper(),
)
# tasks is a list of PlanningTask objects
creator.create_bots(tasks)
```

### Safe mode and overrides

`SelfServiceOverride` automatically toggles safe mode when ROI or error
metrics exceed the configured thresholds. Manual environment variables
`MENACE_SAFE` and `EVOLUTION_PAUSED` no longer need to be set.

Enable safe mode when ROI drops by more than **10%**, error rates exceed **25%**
or the energy score falls under **0.3**. These limits match the defaults used by
the core evolution services. When triggered, `AutoRollbackService` automatically
reverts the most recent commit via `git revert` and keeps safe mode active until
metrics recover.


### Automated environment bootstrap
Run `EnvironmentBootstrapper().bootstrap()` to install required packages, verify
essential OS commands, check remote dependencies, apply database migrations and
provision infrastructure automatically.

### Configuration discovery
`DefaultConfigManager` now auto-generates missing configuration values and
persists them to `.env` so the system can start unattended. `ConfigDiscovery`
additionally inspects Terraform directories and host lists to set
`TERRAFORM_DIR`, `CLUSTER_HOSTS` and `REMOTE_HOSTS` automatically.

### Autoscaling and self-healing
`Autoscaler` integrates with `PredictiveResourceAllocator` for dynamic scaling, while `SelfHealingOrchestrator` redeploys crashed bots and triggers automatic rollbacks when failures persist.

### Monitoring dashboard
`MetricsDashboard` exposes Prometheus metrics via a Flask endpoint for easy visualization.

Run `start_metrics_server(8001)` before executing `benchmark_registered_workflows()`
to publish metrics that Grafana can chart in real time. The dashboard examples in
[docs/metrics_dashboard.md](docs/metrics_dashboard.md) show how to track ROI,
resource usage and latency statistics like the new median latency gauge over
time.

### New autonomous helpers
- `VaultSecretProvider` fetches and caches secrets from an optional remote vault.
- `EnvironmentRestorationService` periodically re-applies the bootstrap process after crashes.
- `SelfTestService` executes the full test suite on a schedule for early failure detection. A new `run-scheduled` CLI command runs tests repeatedly inside Docker/Podman.
- `BotTestingBot` now uses `BotTestingSettings` for configurable runs and database retries.
- `ServiceSupervisor` now logs restarts to `restart.log` for persistent auditing.
- `ClusterServiceSupervisor` supports failover hosts via `FAILOVER_HOSTS`.
- `DependencyUpdateService` can verify container builds on a remote host defined by `DEP_VERIFY_HOST`.
- `UnifiedUpdateService` performs staged rollouts when `NODES` and `ROLLOUT_BATCH_SIZE` are set.
- `SupervisorWatchdog` restarts the ServiceSupervisor if it stops unexpectedly.
- `Watchdog` logs restart and heartbeat failures, publishing errors to the `UnifiedEventBus`.
- `ConfigDiscovery` can monitor for configuration drift when
  `CONFIG_DISCOVERY_INTERVAL` is set.
- `ClusterServiceSupervisor` accepts a `CLUSTER_HEALTH_CHECK_CMD` for deeper
  remote health validation.
- `ComplianceAuditService` runs continuous security and compliance audits.

## Legal Notice

See [LEGAL.md](LEGAL.md) for the full legal terms. In short, this project may
only be used for lawful and ethical activities. The authors do not condone
malicious or unlawful behaviour. The software is provided **as-is** without any
warranties, and the maintainers accept no liability for damages arising from its
use.

