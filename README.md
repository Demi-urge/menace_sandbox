See [docs/sandbox_environment.md](docs/sandbox_environment.md) for required environment variables, optional dependencies and directory layout.

## Dynamic path routing

Scripts and data files are located with `dynamic_path_router.resolve_path` so
shell commands remain portable across forked layouts or nested repositories.
Avoid embedding path literals; always resolve file locations at runtime. The
resolver honours the `SANDBOX_REPO_PATH` environment variable when set and
otherwise falls back to Git metadata or a `.git` directory search.

For details on runtime resolution, caching behaviour and migration tips see
[docs/dynamic_path_router.md](docs/dynamic_path_router.md).

```bash
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)" --help
```

Environment variables influence path resolution. Setting `MENACE_ROOT` or
`SANDBOX_REPO_PATH` overrides the repository root:

```bash
SANDBOX_REPO_PATH=/alt/clone python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('configs/foresight_templates.yaml'))
PY
```

Multi-root configurations are supported via `MENACE_ROOTS` or
`SANDBOX_REPO_PATHS` (use the platform path separator). Supply a hint to target a
specific root:

```bash
MENACE_ROOTS="/repo/main:/repo/experiments" python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py', repo_hint='/repo/experiments'))
PY
```

Combine `SANDBOX_DATA_DIR` with `resolve_path` to locate runtime data files:

```python
import os
from dynamic_path_router import resolve_path
roi_history = resolve_path(f"{os.getenv('SANDBOX_DATA_DIR', 'sandbox_data')}/roi_history.json")
```

The router enables experimenting with alternative directory structures without
breaking tooling, and nested clones can share the same lookup logic.
When adding new scripts or documentation, resolve paths with
`dynamic_path_router.resolve_path` rather than hard‑coding file locations.

## ContextBuilder integration

`ContextBuilder` collects relevant snippets from the local vector databases so
prompts stay grounded in recent history. Instantiate it with the standard
SQLite files and thread the instance into any bot or helper that assembles
prompts. Calls to `_build_prompt`, `build_prompt`, `generate_patch` and similar
functions must explicitly receive this `context_builder`.

```python
from vector_service.context_builder import ContextBuilder

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
```

### BotDevelopmentBot

```python
from bot_development_bot import BotDevelopmentBot, BotSpec
from self_coding_engine import SelfCodingEngine
from menace_memory_manager import MenaceMemoryManager

memory_mgr = MenaceMemoryManager()
engine = SelfCodingEngine("code.db", memory_mgr, context_builder=builder)
bot = BotDevelopmentBot(context_builder=builder, engine=engine)
bot._build_prompt(BotSpec(name="demo", purpose="test"), context_builder=builder)
```

BotDevelopmentBot now routes generation through the local `SelfCodingEngine`.
All prompts are processed on your machine and never sent to external
services. Supply your own engine by instantiating
`SelfCodingEngine` and passing it via the ``engine`` argument. Runtime
behaviour can be tuned with the ``SELF_CODING_INTERVAL``,
``SELF_CODING_ROI_DROP`` and ``SELF_CODING_ERROR_INCREASE`` environment
variables.

### AutomatedReviewer

```python
from automated_reviewer import AutomatedReviewer
from vector_service.context_builder import ContextBuilder

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
reviewer = AutomatedReviewer(context_builder=builder)
reviewer.handle({"bot_id": "1", "severity": "critical"})
```

### QuickFixEngine

```python
from quick_fix_engine import QuickFixEngine, generate_patch
from error_bot import ErrorDB
from self_coding_engine import SelfCodingEngine
from model_automation_pipeline import ModelAutomationPipeline
from data_bot import DataBot
from bot_registry import BotRegistry
from self_coding_manager import SelfCodingManager
from vector_service.context_builder import ContextBuilder

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
manager = SelfCodingManager(
    SelfCodingEngine(),
    ModelAutomationPipeline(),
    data_bot=DataBot(),
    bot_registry=BotRegistry(),
)
engine = QuickFixEngine(ErrorDB(), manager, context_builder=builder)
generate_patch("sandbox_runner", context_builder=builder)
```

If the optional `QuickFixEngine` dependency is unavailable, initialise
`SelfCodingManager` with `skip_quick_fix_validation=True` to bypass quick-fix
validation. The manager will emit a warning and continue running without
pre-validation until the dependency is installed.

Custom commands can be provided for testing and repository cloning:

```python
from pathlib import Path
from self_coding_manager import PatchApprovalPolicy

policy = PatchApprovalPolicy(test_command=["pytest", "-q"])
manager.approval_policy = policy
manager.run_patch(Path("sandbox_runner.py"), "tweak", clone_command=["git", "clone", "--depth", "1"])
```

The default test command (`pytest -q`) can be overridden globally via the
`SELF_CODING_TEST_COMMAND` environment variable or
`SandboxSettings.self_coding_test_command`.  Per-bot commands are specified
with a `test_command` entry in `config/self_coding_thresholds.yaml` or within
`SandboxSettings.bot_thresholds`.

### SelfCodingEngine

```python
from self_coding_engine import SelfCodingEngine
from code_database import CodeDB
from menace_memory_manager import MenaceMemoryManager
from vector_service.context_builder import ContextBuilder

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
engine = SelfCodingEngine(
    CodeDB("code.db"),
    MenaceMemoryManager("mem.db"),
    context_builder=builder,
)
```

The builder queries `bots.db`, `code.db`, `errors.db`, and `workflows.db` and
compresses retrieved snippets before embedding them in prompts.

### Constructor propagation and linting

Always initialise `ContextBuilder` with all four standard databases and thread
the same instance through every component that assembles prompts.  Each
constructor should accept a `context_builder` keyword and pass it to child
helpers:

```python
from vector_service.context_builder import ContextBuilder

class Planner:
    def __init__(self, *, context_builder: ContextBuilder) -> None:
        self.context_builder = context_builder

class PromptingBot:
    def __init__(self, planner: Planner, *, context_builder: ContextBuilder) -> None:
        self.planner = planner
        self.context_builder = context_builder

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
planner = Planner(context_builder=builder)
bot = PromptingBot(planner, context_builder=builder)
```

Violations are detected by `scripts/check_context_builder_usage.py`, which
flags any `_build_prompt`, `build_prompt`, `generate_patch` or `.generate()`
call on `LLMClient`-like instances missing `context_builder`. Functions that
invoke `ContextBuilder.build` while allowing a `None` default or fallback
builder are reported as well. Variables assigned from these classes are
tracked and aliases such as `llm` or `model` are heuristically inspected:

```bash
python scripts/check_context_builder_usage.py
```

Example output when the argument is omitted:

```
service_supervisor.py:289 -> ContextBuilder() missing bots.db, code.db, errors.db, workflows.db
```

### Troubleshooting

Validation failures such as `ContextBuilder validation failed` usually mean one
of the database paths is missing or unreadable.  Confirm that `bots.db`,
`code.db`, `errors.db` and `workflows.db` exist and run
`builder.validate()` or `builder.refresh_db_weights()` to verify connectivity.

## Self-Improvement Sandbox Setup

`initialize_autonomous_sandbox()` now creates a `.env` file on first run and
populates it with minimal defaults so the sandbox can start without manual
configuration. The generated file includes stub values for critical settings:

- `DATABASE_URL` defaults to `sqlite:///menace.db`
- `MODELS` resolves to the bundled `micro_models` directory when set to `demo`
 - Self-coding settings such as `SELF_CODING_INTERVAL` are included with
   sensible defaults so the local `SelfCodingEngine` can generate code without
   any external API keys
- `SANDBOX_DATA_DIR` defaults to `sandbox_data` (use `resolve_path` when
  referencing files under this directory)
- `SANDBOX_LOG_LEVEL` defaults to `INFO` (use `--log-level` to override)
- `PROMPT_CHUNK_TOKEN_THRESHOLD` and `CHUNK_SUMMARY_CACHE_DIR` control code
  chunking token limits and caching for large file summaries (see
  [prompt chunking docs](docs/prompt_chunking.md))

### Billing router

The project depends on the official Stripe SDK (`stripe` Python package) for
billing features. `stripe_billing_router` is the **sole payment interface** for
all billing and
monetisation features.  The router owns the Stripe API keys, resolves per‑bot
identifiers, supports region overrides and strategy hooks, and aborts if keys or
routing rules are missing.  The routing configuration
(`config/stripe_billing_router.yaml`) includes an `account_id` per route to
enforce the destination Stripe account; when omitted the master account is used.
Each charge, subscription, refund or checkout session is logged to both the
structured `billing_logger` and the persistent `billing_log_db` for audit and
reconciliation.

```python
from stripe_billing_router import (
    charge,
    create_customer,
    create_subscription,
    refund,
    create_checkout_session,
    get_balance,
)

charge("finance:finance_router_bot", 12.5)
create_customer("finance:finance_router_bot", {"email": "bot@example.com"})
```

Never call the Stripe SDK directly; use the helpers provided by
`stripe_billing_router` such as `charge`, `create_customer`,
`create_subscription`, `refund`, `create_checkout_session` and
`get_balance`. See [docs/billing_router.md](docs/billing_router.md) for
extension points.

### Billing monitoring

`stripe_watchdog` audits recent Stripe activity against local billing logs and
ROI projections. If the optional ``menace_sanity_layer`` dependency is missing,
its feedback stubs emit a single warning and then raise
``SanityLayerUnavailableError`` on further use. Set
``MENACE_SANITY_OPTIONAL=1`` in development to log a critical alert instead of
raising.

### Chunking pipeline

Large modules are split and summarised before prompting so the LLM never sees
more than the configured token budget. The pipeline consists of:

1. `chunk_file` – walks a module and produces token‑bounded chunks.
2. `summarize_code` – generates a short natural‑language blurb for each chunk.
3. `get_chunk_summaries` – uses a disk-backed cache to persist chunk summaries
   and automatically invalidates them when the source file changes.

`PROMPT_CHUNK_TOKEN_THRESHOLD` sets the maximum tokens per chunk and
`CHUNK_SUMMARY_CACHE_DIR` relocates the cache directory. See
[chunked prompting](docs/chunked_prompting.md) for configuration details.

Bootstrap verifies these variables before launching and raises a clear error if
any required value is missing or the model path does not exist.

Install the auxiliary packages for self-improvement ahead of time:

```
make install-self-improvement-deps
```

`verify_dependencies()` reports missing or mismatched packages.  When the
`AUTO_INSTALL_DEPENDENCIES` setting or ``auto_install`` flag is enabled it will
attempt to install requirements automatically, falling back to the same error
message if installation fails.

- Revenue tracking and monetisation helpers
- **Profit density evaluation** keeping only the most lucrative clips
- Intelligent culling for clips, accounts and topics
- Scoutboard topic log with historical performance
- Prediction engine for emerging topics
- Dynamic account redistribution based on profit density
- Automated reinvestment of profits via `investment_engine.AutoReinvestmentBot` with a predictive spend engine ([docs/auto_reinvestment.md](docs/auto_reinvestment.md))
- Bottleneck detection via performance monitors ([docs/bottleneck_detection.md](docs/bottleneck_detection.md))
- Adaptive energy score guiding resource allocation ([docs/energy_score.md](docs/energy_score.md))
- Central Menace orchestrator coordinating all stages and hierarchical oversight (requires an explicit `ContextBuilder` instance)
- Self improvement engine automatically runs the workflow on the Menace model when metrics degrade
- Optional meta planner integrates structural evolution; set `ENABLE_META_PLANNER=true` to require its presence
- ROI foresight with `ForesightTracker.predict_roi_collapse`, projecting trends,
  classifying risk (Stable, Slow decay, Volatile, Immediate collapse risk) and
  flagging brittle workflows. Baseline curves live in
  `resolve_path('configs/foresight_templates.yaml')` with keys `profiles`, `trajectories`,
  `entropy_profiles`, `risk_profiles`, `entropy_trajectories` and
  `risk_trajectories`
- Self-coding manager applies patches then deploys via the automation pipeline
- Command-line interface for setup, retrieval and scaffolding (see
  [docs/cli.md](docs/cli.md)). Semantic search falls back to full-text
  search when needed and cached results expire after one hour. Scaffolding
  commands expose hooks for router registration and migrations.
- Workflow synthesizer CLI lists unresolved inputs for candidates and
  summarises evaluation outcomes with score components.
- Memory-aware GPT wrapper injects prior feedback and fixes into prompts when
  calling ChatGPT. Callers pass a module/action key so retrieved context and the
  resulting interaction are logged with standard tags via
  `memory_logging.log_with_tags`. See
  [docs/autonomous_sandbox.md#gpt-interaction-tags](docs/autonomous_sandbox.md#gpt-interaction-tags)
  for details.
- Evolution orchestrator coordinating self improvement and structural evolution
- System evolution manager runs GA-driven structural updates ([docs/system_evolution_manager.md](docs/system_evolution_manager.md))
- Workflow evolution manager benchmarks variant sequences and promotes higher-ROI versions with diminishing-returns gating ([docs/workflow_evolution.md](docs/workflow_evolution.md)). Similar variants are compared via the [Workflow Synergy Comparator](docs/workflow_synergy_comparator.md) before merging
- Experiment manager for automated A/B testing of bot variants
- Neuroplasticity tracking via PathwayDB ([docs/neuroplasticity.md](docs/neuroplasticity.md))
- Atomic write mirroring with `TransactionManager`
- Remote database replication via `DatabaseRouter(remote_url=...)`
- Shared/local SQLite routing via `DBRouter` ([docs/db_router.md](docs/db_router.md))
  - Table access metrics can be flushed to the telemetry backend via
    `router.get_access_counts(flush=True)`.
  - Set `DB_ROUTER_METRICS_INTERVAL` to automatically report counts at the
    specified interval (seconds). Deployment configs now enable this by default.
  - Run `python table_usage_cli.py --flush` to view shared vs. local usage per
    Menace ID.
  - Audit logs captured via `audit.log_db_access` include the menace ID. Entries
    default to `logs/shared_db_access.log` (override with `DB_ROUTER_AUDIT_LOG` or
    via the function's `log_path` parameter) and can also be inserted into the
    `db_access_audit` table by passing `log_to_db=True`. Summarise logs for
    metrics or anomaly detection with
    `analysis/db_router_log_analysis.py`.
  - Read helpers accept a `scope` parameter (`"local"`, `"global"`, `"all"`)
    to filter records by menace ID, replacing `include_cross_instance` and
    `all_instances` flags. `global` retrieves entries from other Menace instances,
    while `all` removes filtering entirely. For example:

    ```python
    from workflow_summary_db import WorkflowSummaryDB

    db = WorkflowSummaryDB()
    db.get_summary(1, scope="local")   # -> WorkflowSummary for current menace
    db.get_summary(1, scope="global")  # -> records from other menace instances
    db.get_summary(1, scope="all")     # -> records from all menaces
    ```
    Scoping can also be applied manually when building SQL queries:

    ```python
    from scope_utils import Scope, build_scope_clause, apply_scope

    clause, params = build_scope_clause("bots", Scope.GLOBAL, "alpha")
    sql = apply_scope("SELECT * FROM bots", clause)
    # sql == "SELECT * FROM bots WHERE bots.source_menace_id <> ?"
    # params == ["alpha"]
    ```
  - Shared tables include a `content_hash` column storing a SHA256 hash of JSON-encoded core fields.
    A new `insert_if_unique` utility computes this hash and skips inserts when the value already exists,
    suppressing cross-instance duplicates. Existing databases will auto-add the column on startup,
    or can be updated manually with:

    ```sql
    ALTER TABLE bots ADD COLUMN content_hash TEXT;
    CREATE UNIQUE INDEX idx_bots_content_hash ON bots(content_hash);
    ALTER TABLE workflows ADD COLUMN content_hash TEXT;
    CREATE UNIQUE INDEX idx_workflows_content_hash ON workflows(content_hash);
    ALTER TABLE enhancements ADD COLUMN content_hash TEXT;
    CREATE UNIQUE INDEX idx_enhancements_content_hash ON enhancements(content_hash);
    ALTER TABLE errors ADD COLUMN content_hash TEXT;
    CREATE UNIQUE INDEX idx_errors_content_hash ON errors(content_hash);
    ```
- Change Data Capture events published to `UnifiedEventBus`
- Vector embedding search via `EmbeddableDBMixin` for bots, workflows,
  errors, enhancements and research items. Each database stores its own
  embedding fields and can use either a FAISS or Annoy backend for
  similarity queries. See [docs/embedding_system.md](docs/embedding_system.md)
  for configuration details and backfilling instructions.
- Lightweight intent clustering for modules with semantic query support
  ([docs/intent_clusterer.md](docs/intent_clusterer.md))
- `EmbeddingBackfill` and `EmbeddingScheduler` now handle `bots`, `workflows`,
  `enhancements` and `errors` tables directly.  Operators can schedule
  periodic backfills by setting `EMBEDDING_SCHEDULER_SOURCES` to a comma
  separated list (e.g. `bots,workflows,enhancements,errors`).
- Unified cross-database search through `UniversalRetriever`.  The
  `retrieve(query, top_k=10, link_multiplier=1.1)` API returns
  `ResultBundle` objects with `origin_db`, record metadata, a final score and a
  reason backed by normalised metrics (error frequency, ROI uplift, workflow
  usage and bot deployment) with optional relation-based boosting.
- Service layer wrappers in `vector_service` expose `Retriever`, `ContextBuilder`,
  `PatchLogger` and `EmbeddingBackfill` with structured logging and metrics.
  Other modules should interact with embeddings through this layer rather than
  accessing databases or retrievers directly. See
  [docs/vector_service.md](docs/vector_service.md) for detailed API
  documentation. A lightweight FastAPI app in `vector_service_api.py`,
  initialised via `create_app(ContextBuilder(...))` which stores the builder on
  `app.state`, provides `/search`,
  `/build-context`, `/track-contributors` and `/backfill-embeddings` endpoints.
- Overview of the vectorised cognition pipeline – from embedding backfills to
  ranking with ROI feedback – is available in
  [docs/vectorized_cognition.md](docs/vectorized_cognition.md).
- Compact, offline context assembly via `ContextBuilder` which summarises error,
  bot, workflow and code records for code‑generation modules. **All prompt‑constructing
  bots require an explicit builder**; omitting it raises a `ValueError` or falls
  back to bland prompts with no retrieved context. Instantiate it with the
  standard local databases, e.g. `builder = ContextBuilder("bots.db",
  "code.db", "errors.db", "workflows.db")`, then pass the builder to
  `SelfCodingEngine`, `QuickFixEngine`, `Watchdog` or `AutomatedReviewer`
  ([docs/context_builder.md](docs/context_builder.md))
- Optional RabbitMQ integration via `UnifiedEventBus(rabbitmq_host=...)`
- Schema migrations managed through Alembic
- Long-term metrics dashboards with Prometheus ([docs/metrics_dashboard.md](docs/metrics_dashboard.md))
- Workflow benchmarking metrics exported to Prometheus with automatic early stopping when improvements level off. Additional gauges track CPU time, memory usage, network and disk I/O with statistical significance tests.
- Lightweight CLI for ad-hoc workflow scoring that logs results to ``roi_results.db`` (see ``docs/workflow_benchmark.md``).
- `metrics_exporter` tries to install `prometheus_client` during bootstrap and
  serves a fallback HTTP endpoint if the install fails
- Centralised logging via Elasticsearch or Splunk and optional Sentry alerts ([docs/monitoring_pipeline.md](docs/monitoring_pipeline.md))
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
- Human alignment flagger with background review agent ([docs/human_alignment_flagger.md](docs/human_alignment_flagger.md))
- Self-provisioning of missing packages through `SystemProvisioner`
- Distributed rollback verification via `RollbackValidator`
- ROI-driven autoscaling with `ROIScalingPolicy`
- Profile-based ROI evaluation via `ROICalculator` using YAML-configured weights and veto rules. A valid ROI profile file (such as `configs/roi_profiles.yaml`) must be available; otherwise calculator initialisation fails.
- ROI history forecasting via `ROITracker` ([docs/roi_tracker.md](docs/roi_tracker.md))
 - `calculate_raroi` derives a risk-adjusted ROI (RAROI) by applying
   catastrophic risk, recent stability and safety multipliers. Impact
   severities load from `config/impact_severity.yaml` or a file referenced by
   the `IMPACT_SEVERITY_CONFIG` environment variable. Runtime `metrics`
   from sandbox modules and failing test results passed via `failing_tests`
   shape the `safety_factor`. Each failing security or alignment test halves
   the factor, so RAROI close to the raw ROI signals a stable, low-risk
   workflow while a much lower RAROI highlights risk and volatility. When the
   score drops below `raroi_borderline_threshold` the method also returns
   bottleneck suggestions via `propose_fix`. RAROI feeds into module ranking
   and guides self-improvement decisions.
- Composite workflow evaluation and persistence via `CompositeWorkflowScorer` ([docs/composite_workflow_scorer.md](docs/composite_workflow_scorer.md))
- Scenario scorecards collate baseline and stress ROI plus metric and synergy
  deltas for each stress preset. `sandbox_runner.environment.run_scenarios`
  returns a mapping of these scorecards keyed by scenario. They can also be
  generated with `ROITracker.generate_scenario_scorecard` or via `python -m
  menace_sandbox.adaptive_roi_cli scorecard`.
- Deployment governance evaluates RAROI, confidence and scenario scores with
   optional signed overrides and a foresight promotion gate. The
   `foresight_gate.is_foresight_safe_to_promote` check enforces projected ROI thresholds and a
   minimum `0.6` confidence, rejects collapse risk or negative DAG impact and
   records each decision in `forecast_records/decision_log.jsonl` along with the
   full forecast projections, confidence and reason codes. Failed gates are
   downgraded to the borderline bucket when available or routed through a pilot
   run. See [docs/deployment_governance.md](docs/deployment_governance.md) for
   rule syntax, configuration paths and example policy/scorecard templates.

### SQLite Write Buffer

SQLite's coarse file locking can stall concurrent writers. When `USE_DB_QUEUE`
is set, calls to `insert_if_unique` are queued as JSONL records under
`SANDBOX_DATA_DIR/queues`, grouped per menace in `<menace_id>.jsonl` files. Each
entry stores the source `MENACE_ID` and a content hash for deduplication. See
[docs/shared_db_queue.md](docs/shared_db_queue.md) for queue layout,
environment variables and recovery steps.

A background daemon flushes the queue into the shared database:

```bash
python sync_shared_db.py --db-url sqlite:///menace.db \
  --queue-dir "$(python - <<'PY'
from dynamic_path_router import resolve_path
import os
print(resolve_path(f"{os.getenv('SANDBOX_DATA_DIR', 'sandbox_data')}/queues"))
PY
)"
```

Successful rows are committed and removed, retries happen up to three times and
then move to `queue.failed.jsonl`. Override the queue directory with
`SHARED_QUEUE_DIR` (or `DB_ROUTER_QUEUE_DIR` when using `DBRouter`) when running
multiple instances. The daemon polls for new records every `SYNC_INTERVAL`
seconds (default `10`) and creates the queue directory if it is missing.

For PostgreSQL or other backends that handle concurrent writes, disable the
buffer by unsetting `USE_DB_QUEUE` (or passing `queue_path=None`) so inserts go
directly to the database and the daemon is unnecessary.

### Codex database helpers

The `codex_db_helpers` module gathers training samples from Menace's
enhancement, workflow summary, discrepancy and workflow databases. It powers
fleetwide Codex training by aggregating examples from all Menace instances.
Each helper supports:

- `sort_by` – one of `"confidence"`, `"outcome_score"` or `"timestamp"`.
- `limit` – maximum number of records to return.
- `include_embeddings` – attach vector embeddings when available.
- `scope` – restrict to `Scope.LOCAL`, `Scope.GLOBAL` or `Scope.ALL` (default).

All queries run fleetwide by applying `Scope.ALL` via `build_scope_clause`.

```python
from codex_db_helpers import aggregate_samples, Scope

samples = aggregate_samples(sort_by="timestamp", limit=20, scope=Scope.ALL)
prompt = "\n\n".join(s.content for s in samples)
```

To support additional data types, implement a `fetch_*` helper that returns a
list of `TrainingSample` objects and register it with `aggregate_samples`. See
[docs/codex_db_helpers.md](docs/codex_db_helpers.md) for more details and
[docs/codex_training_data.md](docs/codex_training_data.md) for a tour of the
available sources and prompt-building examples.

### ROI toolkit

ROI evaluation combines `ROICalculator`, `ROITracker` and the RAROI formula to
measure and forecast workflow impact. `ROITracker.update()` records ROI deltas
while `calculate_raroi()` applies risk multipliers. When a workflow's RAROI or
confidence falls below the configured thresholds the tracker queues it in a
[borderline bucket](docs/borderline_bucket.md) for micro‑pilot testing. Use
`borderline_bucket.process()` (or the tracker's
`process_borderline_candidates()` wrapper) after updates to promote or
terminate these candidates based on the trial results:

```python
from menace_sandbox.roi_tracker import ROITracker

tracker = ROITracker(raroi_borderline_threshold=0.05)
tracker.update(0.12)
  base, raroi, _ = tracker.calculate_raroi(1.1)
tracker.borderline_bucket.process(
    raroi_threshold=tracker.raroi_borderline_threshold,
    confidence_threshold=tracker.confidence_threshold,
)
```

  The snippet records an ROI delta, computes a risk-adjusted score and evaluates
  borderline workflows before full adoption.

`DeploymentGovernor` invokes `foresight_gate.is_foresight_safe_to_promote` before finalising a
promote verdict. The gate requires each projected ROI to clear the supplied
threshold and the forecast confidence to meet or exceed `0.6`, rejects collapse
predictions and negative DAG impact, and logs the evaluation to
`forecast_records/decision_log.jsonl` including projections, confidence and
reason codes. Upgrades that fail the gate are downgraded to `borderline` when a
bucket is configured or run as a limited pilot.

- Debug logs report the EMA and standard deviation used for ROI thresholds along
  with per-metric synergy EMA, deviation and confidence. Window sizes and weight
  parameters are included in each `log_record` entry.
- Synergy-aware environment presets adapt CPU, memory, bandwidth and threat levels ([docs/environment_generator.md](docs/environment_generator.md))
- Sandboxed self-debugging using `SelfDebuggerSandbox` (invoked by `launch_menace_bots.py` after the test run) which requires a `SelfCodingManager`
- Comprehensive build pipeline in `launch_menace_bots.py` that plans,
  develops, tests and scales bots before deployment.  `debug_and_deploy`
  creates a `ContextBuilder` with local database paths by default, but callers
  may supply custom builders if needed.
- Automated implementation pipeline turning tasks into runnable bots ([docs/implementation_pipeline.md](docs/implementation_pipeline.md))
- Models repository workflow. The visual-agent pathway has been fully decommissioned; use the standard code generation pipeline instead ([docs/models_repo_workflow.md](docs/models_repo_workflow.md))
- Retirement of underperforming models by `ModelPerformanceMonitor`
- External dependency monitoring and failover via `DependencyWatchdog`
- Hardware discovery sets `NUM_GPUS` and `GPU_AVAILABLE` automatically
- Cross-platform service installation via `service_installer.py`
- Running `menace_master.py` as root installs the service automatically
- Background updates handled by `UnifiedUpdateService` even without the supervisor
- Automatic first-run sandbox improving the codebase before live execution
- Entropy delta detection marks modules complete when ROI gains per entropy unit
  fall below `entropy_threshold` for a configurable number of consecutive cycles
- Recursive inclusion flow discovers orphan and isolated modules, tests them
  and integrates passing ones into existing workflows. The helper
  `scripts/discover_isolated_modules.py` surfaces standalone files while
  `sandbox_runner.discover_recursive_orphans` walks each candidate's import
  chain, returning its importing `parents` and a `redundant` flag. All isolated
  modules are executed before final classification. With `SANDBOX_TEST_REDUNDANT=1`
  (or `--include-redundant`/`--test-redundant`) the self-test service also
  exercises modules tagged as redundant or legacy so their helper chains are
  validated; set `SANDBOX_TEST_REDUNDANT=0` to log them without execution. The
  improvement engine integrates passing modules whose `redundant` flag is false.
  Run the isolated scan with `SANDBOX_DISCOVER_ISOLATED=1` (or
  `--discover-isolated`) and pair it with
  `SANDBOX_AUTO_INCLUDE_ISOLATED=1`/`--auto-include-isolated` to test and merge
  the results automatically. Recursion through dependencies is enabled by
  default (`SELF_TEST_RECURSIVE_ORPHANS=1`, `SELF_TEST_RECURSIVE_ISOLATED=1`,
  `SANDBOX_RECURSIVE_ORPHANS=1`, `SANDBOX_RECURSIVE_ISOLATED=1`); the CLI
  aliases are `--recursive-include`/`--no-recursive-include` and
  `--recursive-isolated`/`--no-recursive-isolated`. Limit how far helper
  chains are followed with `SANDBOX_MAX_RECURSION_DEPTH` or
  `--max-recursion-depth`. Use
  `SANDBOX_CLEAN_ORPHANS=1` (or `--clean-orphans`) to prune processed names from
  `sandbox_data/orphan_modules.json`. Environment flags are mirrored to matching
  `SELF_TEST_*` variables so the self-test service honours the same behaviour.

  ```bash
  # Example: disable recursion but include isolated modules automatically
  SANDBOX_DISCOVER_ISOLATED=1 SANDBOX_RECURSIVE_ORPHANS=0 \
  SANDBOX_RECURSIVE_ISOLATED=0 SANDBOX_AUTO_INCLUDE_ISOLATED=1 \
  SANDBOX_CLEAN_ORPHANS=1 run_autonomous --check-settings
  ```

  ```bash
  # Same behaviour using CLI flags
  python "$(python - <<'PY'
  from dynamic_path_router import resolve_path
  print(resolve_path('run_autonomous.py'))
  PY
  )" --no-recursive-orphans --no-recursive-isolated \
      --discover-isolated --auto-include-isolated --clean-orphans \
      --check-settings
  ```

  ```bash
  # Include modules flagged as redundant (default)
  python "$(python - <<'PY'
  from dynamic_path_router import resolve_path
  print(resolve_path('run_autonomous.py'))
  PY
  )" --include-redundant

  # Skip redundant modules during tests
  SANDBOX_TEST_REDUNDANT=0 python "$(python - <<'PY'
  from dynamic_path_router import resolve_path
  print(resolve_path('run_autonomous.py'))
  PY
  )" --discover-orphans
  ```

  Set `SANDBOX_FAIL_ON_MISSING_SCENARIOS=1` to raise an error when canonical
  scenarios are missing from coverage.

  ### Step-by-step: recursive inclusion run

  ```bash
  # 1. Prepare a candidate and its dependency
  echo 'import util\n' > candidate.py
  echo 'VALUE = 1\n'   > util.py

  # 2. Discover orphans and include them recursively
  python -m sandbox_runner.cli --discover-orphans --recursive-include \
      --auto-include-isolated --clean-orphans

  # 3. Inspect the generated module map and workflows
  python - <<'PY'
  from dynamic_path_router import resolve_path
  import os
  path = resolve_path(f"{os.getenv('SANDBOX_DATA_DIR', 'sandbox_data')}/module_map.json")
  print(open(path).read())
  PY
  ```

  `sandbox_runner.discover_recursive_orphans` traces the `candidate -> util`
  chain. With `SANDBOX_AUTO_INCLUDE_ISOLATED=1` and
  `SANDBOX_RECURSIVE_ORPHANS=1` the sandbox appends both modules to
  `sandbox_data/module_map.json` and `environment.generate_workflows_for_modules`
  creates one‑step workflows for future runs.

See [docs/quickstart.md](docs/quickstart.md) for a Quickstart guide on launching the sandbox.
Run `scripts/check_personal_setup.py` afterwards to verify your environment variables.
Detailed environment notes are available in [docs/autonomous_sandbox.md](docs/autonomous_sandbox.md).

### Service layer examples

The sandbox exposes a number of background services that follow a common
``run_continuous`` pattern.  Instantiate the service, start it with an optional
``threading.Event`` and stop it when finished:

```python
import threading
from microtrend_service import MicrotrendService
from workflow_cloner import WorkflowCloner
from self_evaluation_service import SelfEvaluationService

svc = SelfEvaluationService(
    microtrend=MicrotrendService(),
    cloner=WorkflowCloner(),
)
stop = threading.Event()
svc.run_continuous(interval=3600, stop_event=stop)
# ... later ...
stop.set()
```

### Memory-aware GPT wrapper

Use `memory_aware_gpt_client.ask_with_memory` to automatically prepend past
feedback, improvement paths and error fixes to a new prompt.  Callers must
provide a ``key`` identifying the module and action so related interactions can
be retrieved and logged.  Example:

```python
from memory_aware_gpt_client import ask_with_memory
from log_tags import ERROR_FIX

result = ask_with_memory(client, "coding_bot_interface.manager_generate_helper", "Write tests",
                         memory=gpt_memory, tags=[ERROR_FIX], manager=manager)
```

For a deeper overview of the `LocalKnowledgeModule`, required tags, environment
variables and multi-run examples see
[docs/gpt_memory.md](docs/gpt_memory.md).

### Embedding lifecycle

Databases such as `BotDB`, `WorkflowDB`, `ErrorDB`, `EnhancementDB` and
`InfoDB` subclass `EmbeddableDBMixin` to persist and query vector
embeddings.  When a record is added or updated the mixin's
``try_add_embedding`` helper stores the vector in a dedicated
``<table>_embeddings`` table together with metadata like
``created_at`` and ``embedding_version``.  The mixin builds a FAISS or
Annoy index (falling back to whichever backend is available) and saves
it to ``vector_index_path`` so subsequent calls to
``search_by_vector`` can efficiently return similar records.  Updating
the embedding version automatically refreshes stored vectors and
rebuilds the index as needed.

### Orphan discovery helpers

See [docs/recursive_orphan_workflow.md](docs/recursive_orphan_workflow.md) for a
summary of the flags controlling the orphan workflow.

`discover_recursive_orphans` walks import chains for orphaned modules, while
`include_orphan_modules` queues names from `sandbox_data/orphan_modules.json`.
Passing files are merged into `sandbox_data/module_map.json` via
`auto_include_modules`. With `recursive=True` or `SANDBOX_RECURSIVE_ORPHANS=1`
the helper expands local imports and, when `SANDBOX_AUTO_INCLUDE_ISOLATED=1`,
adds files returned by `scripts.discover_isolated_modules`. Redundant entries
remain in `sandbox_data/orphan_modules.json` unless `SANDBOX_TEST_REDUNDANT=1`
allows integration. See [docs/autonomous_sandbox.md](docs/autonomous_sandbox.md)
and [docs/self_improvement.md](docs/self_improvement.md) for full
examples and environment variables such as `SANDBOX_RECURSIVE_ORPHANS`,
`SANDBOX_AUTO_INCLUDE_ISOLATED` and `SANDBOX_RECURSIVE_ISOLATED`.

Example `sandbox_data/orphan_modules.json`:

```json
{
  "detached/util.py": {"source": "static-analysis"},
  "legacy/task.py":   {"source": "runtime"}
}
```

`include_orphan_modules` reads each key and passes them to
`auto_include_modules` for validation and potential integration.

A new `menace` CLI wraps common workflows so you no longer need to remember individual scripts.

### Quick patches

Apply a patch to a module and log provenance:

```bash
python menace_cli.py patch "$(python - <<'PY'
from dynamic_path_router import path_for_prompt
print(path_for_prompt('bots/example.py'))
PY
)" --desc "Fix bug" --context '{"foo": "bar"}'
```

The command prints the patch ID and affected files. The `--context` value must be valid JSON.

### Semantic retrieval

Search across databases with optional caching. By default results are rendered
as a text table:

```bash
python menace_cli.py retrieve "database connector" --db code --top-k 3
```

To emit raw JSON, add `--json`:

```bash
python menace_cli.py retrieve "database connector" --db code --top-k 3 --json
```

Results are cached under `~/.cache/menace/retrieve.json`. Use `--no-cache` to
bypass the cache or `--rebuild-cache` to refresh. When no vector hits are found
the CLI falls back to a full‑text search.

### Embedding backfill

Populate vector indices via the `embed` subcommand. Specify a database name to
limit processing to a single class:

```bash
python menace_cli.py embed --db workflows --batch-size 100 --backend faiss
```

To embed the core databases (`code`, `bot`, `error` and `workflow`) in one go,
run:

```bash
python menace_cli.py embed core
```

Progress for each database is reported and any skipped records or licensing
issues are printed to the console or log file.

### New database scaffold

Generate a database module skeleton:

```bash
python menace_cli.py new-db sample
```

This creates `sample_db.py` and updates `__init__.py` in the current directory.

### Patch provenance queries

Use the `patches` subcommand to inspect the patch history database. Commands
output JSON for downstream processing:

```bash
python menace_cli.py patches list --limit 5
python menace_cli.py patches ancestry 42
python menace_cli.py patches search --vector v1
python menace_cli.py patches search --license MIT
```

### Bot development

Every constructor or method that assembles prompts must accept a
`ContextBuilder` and use it for token accounting and ROI tracking. Callers
provide the builder explicitly.

```python
from prompt_engine import PromptEngine
from vector_service.context_builder import ContextBuilder

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
engine = PromptEngine(context_builder=builder)
prompt = engine.build_prompt("Expand tests", context_builder=builder)
```

Internal helpers follow the same pattern, and any helper that calls
`ContextBuilder.build` must require an injected builder:

```python
from menace.bot_development_bot import BotDevelopmentBot, BotSpec

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
bot = BotDevelopmentBot(context_builder=builder)
prompt = bot._build_prompt(BotSpec(goal="Add feature"), context_builder=builder)
```

The pre-commit hook `check-context-builder-usage` wraps
`scripts/check_context_builder_usage.py` and must pass:

```bash
pre-commit run check-context-builder-usage --all-files
```

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

Orphaned modules listed in `sandbox_data/orphan_modules.json` are flagged as
"not yet tested". The `SelfTestService` now discovers and runs these modules
automatically. Orphan discovery utilities now live in
`sandbox_runner.orphan_discovery`, and the service follows dependencies via
`sandbox_runner.discover_recursive_orphans`. The helper is exported from the
package and can be imported directly. By default orphan modules are processed
(`SELF_TEST_DISABLE_ORPHANS=0`, `SELF_TEST_DISCOVER_ORPHANS=1`). Set
`SELF_TEST_DISABLE_ORPHANS=1` or pass `--include-orphans` to skip them.
Recursion through orphan dependencies is enabled by default; set
`SELF_TEST_RECURSIVE_ORPHANS=0` or `SANDBOX_RECURSIVE_ORPHANS=0`, or pass
`--no-recursive-include` to turn off dependency scanning. Disable automatic
discovery with
`SELF_TEST_DISCOVER_ORPHANS=0` or `--discover-orphans`.
When launched via `run_autonomous.py` newly discovered files are written to
`orphan_modules.json` unless `SANDBOX_DISABLE_ORPHAN_SCAN=1`. Passing orphan
modules are merged into `module_map.json` automatically after the tests run,
creating one-step workflows so they become immediately available for
benchmarking.  Use `--clean-orphans` or set `SANDBOX_CLEAN_ORPHANS=1` to remove
successful modules from the orphan list after integration.
During evaluation each candidate is executed inside an ephemeral sandbox via
`SelfTestService` which runs `pytest` for the module and its discovered
dependencies. Passing modules are appended to `sandbox_data/module_map.json` and
`sandbox_runner.environment.generate_workflows_for_modules` creates trivial
workflows so the new code participates in later simulations.
If an orphan maps onto an existing workflow group, the sandbox attempts to add it to those sequences automatically.
Each candidate is tagged with a `redundant` flag by
`orphan_analyzer.analyze_redundancy`; redundant modules are logged and skipped
during integration. Isolated modules returned by
`scripts/discover_isolated_modules.py` are included when `--auto-include-isolated`
is supplied or `SELF_TEST_AUTO_INCLUDE_ISOLATED=1`/`SANDBOX_AUTO_INCLUDE_ISOLATED=1`
is set. To run the discovery step explicitly use `--discover-isolated` or set
`SELF_TEST_DISCOVER_ISOLATED=1`/`SANDBOX_DISCOVER_ISOLATED=1`. Recursion through
their dependencies is enabled by default (`SELF_TEST_RECURSIVE_ISOLATED=1` and
`SANDBOX_RECURSIVE_ISOLATED=1`); set `SELF_TEST_RECURSIVE_ISOLATED=0` or
`SANDBOX_RECURSIVE_ISOLATED=0` or pass `--no-recursive-isolated` to disable.
Use `--clean-orphans` or `SANDBOX_CLEAN_ORPHANS=1` to prune successful modules
from the orphan cache after integration.

Example integrating an isolated module with dependencies:

```bash
echo 'import helper\n' > isolated.py
echo 'VALUE = 1\n'   > helper.py

# discover, test and integrate both files automatically
python -m sandbox_runner.cli --discover-orphans --auto-include-isolated \
    --recursive-isolated --clean-orphans
```

`discover_isolated_modules` locates `isolated.py` and `helper.py`,
`SelfTestService` exercises them recursively and non-redundant modules are
written to `sandbox_data/module_map.json`. With `--clean-orphans` (or
`SANDBOX_CLEAN_ORPHANS=1`) any processed names are pruned from
`sandbox_data/orphan_modules.json`.

Example discovering orphans:

```bash
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('run_autonomous.py'))
PY
)" --discover-orphans --clean-orphans \
    --include-orphans
```
After the run, passing modules are appended to `sandbox_data/module_map.json`
and any matching entries are dropped from `orphan_modules.json`. The next
autonomous cycle can schedule the new modules alongside existing workflows,
so they become part of broader flows automatically. Use `--no-recursive-include`
or `--no-recursive-isolated` to disable dependency traversal. Recursion through
orphan dependencies is enabled by default; set `SANDBOX_RECURSIVE_ORPHANS=0` or
`SELF_TEST_RECURSIVE_ORPHANS=0`, or pass `--no-recursive-include` to disable.
Set `SANDBOX_CLEAN_ORPHANS=1` to mirror the `--clean-orphans` option.
The sandbox can also group modules automatically by analysing imports,
function calls and optionally docstring similarity. HDBSCAN clustering can be
selected when the optional dependency is installed. Run the unified helper:

```bash
python scripts/build_module_map.py --semantic --threshold 0.2
```

This writes `sandbox_data/module_map.json` assigning each file to a numeric
cluster. `SANDBOX_AUTO_MAP=1` triggers the same process on startup.
Set `SANDBOX_SEMANTIC_MODULES=1` to include docstring-based edges when generating
the map. See [docs/dynamic_module_mapping.md](docs/dynamic_module_mapping.md)
for details on the `--algorithm` options (``greedy``, ``label`` or ``hdbscan``),
the `--threshold`, `--semantic` and `--exclude` flags. Patterns from `SANDBOX_EXCLUDE_DIRS` are passed to the helper
automatically.

When the workflow database is empty you can generate temporary workflows from
these module groups by launching the sandbox with `--dynamic-workflows`. The
clustering behaviour mirrors the mapping helper and is controlled via
`--module-algorithm`, `--module-threshold` and `--module-semantic`.

## Installation

Run the setup helper to install the required system tools and Python
dependencies.  When ``MENACE_OFFLINE_INSTALL=1`` is set the script
installs packages from ``MENACE_WHEEL_DIR`` instead of contacting PyPI.

For deterministic environments you can install the pinned Python
dependencies up front using:

```bash
scripts/install_pinned_dependencies.sh
```

This script installs the exact versions listed in ``requirements.txt``.

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

To run the sandbox directly with metrics visualisation, use the convenience
script which installs dependencies via `setup_env.sh` before launching the
dashboard:

```bash
scripts/start_sandbox.sh
```

To visualise synergy metrics separately, run:

```bash
python -m menace.self_improvement synergy-dashboard --wsgi flask
```

Replace `flask` with `gunicorn` or `uvicorn` to use a different server.

To explore the existing module index interactively, use the intent clusterer
helper:

```bash
python -m intent_clusterer "error handling"
```

Or import the convenience helpers directly:

```python
from intent_clusterer import find_modules_related_to, find_clusters_related_to
find_modules_related_to("error handling")
find_clusters_related_to("error handling")
```

``find_modules_related_to`` returns relevant module paths, while
``find_clusters_related_to`` surfaces synergy clusters.  Each result is an
``IntentMatch`` carrying a similarity ``score`` and an ``origin`` attribute
indicating the result type.

The underlying ``IntentClusterer`` is available at the package root for custom
workflows:

```python
from pathlib import Path
from menace import IntentClusterer

clusterer = IntentClusterer()
clusterer.index_repository(Path("."))
```

See [docs/intent_clusterer.md](docs/intent_clusterer.md) for additional usage
notes.

### Entropy delta detection

The sandbox monitors the ROI gain relative to entropy changes for each module.
When the ratio stays below `entropy_threshold` for a number of consecutive
cycles the section is marked complete and skipped in later runs. Tune the
behaviour with the `--entropy-threshold`/`ENTROPY_THRESHOLD` flag and the
`--consecutive`/`ENTROPY_PLATEAU_CONSECUTIVE` window size:

```bash
python -m sandbox_runner.cli --entropy-threshold 0.02 --consecutive 5
```

### Required dependencies

The following Python packages must be available. They are installed
automatically when building the Docker image or running ``pip install -e .``:

```
moviepy
pytube
selenium
pyautogui
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
torch or numpy
sentry-sdk
```

At least one numeric backend (PyTorch or NumPy) must be installed.

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

### Running integration tests

Run the deterministic integration tests with:

```bash
pytest tests/integration/test_stub_generation.py tests/integration/test_planning_cycle.py tests/integration/test_environment_cleanup.py
```

Run all integration tests via:

```bash
pytest tests/integration
```

### Optional dependencies

Some features such as anomaly detection and the `ErrorForecaster` make use of
additional libraries. **pandas** and **PyTorch** are now installed by default,
yet the modules still implement fallbacks so functionality remains intact even
if those libraries are unavailable. The `TruthAdapter` tests rely on SciPy for
Kolmogorov–Smirnov based drift detection and stub XGBoost to exercise model
selection; both dependencies are optional and have lightweight fallbacks when
absent.

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
``DATABASE_URL`` and credentials for external services like ``SERP_API_KEY``.
Code generation uses the local ``SelfCodingEngine``; prompts stay on your
machine so no external API keys are required. Additional service specific keys
(for example ``REDDIT_CLIENT_SECRET``) can be added to the same file. Stripe keys are
handled separately via ``stripe_billing_router`` and should not be stored in
configuration files.
Set ``MENACE_ENV_FILE`` to load variables from a different path or call
``auto_env_setup.ensure_env("custom.env")`` to generate it elsewhere.

When resolving settings, the loader reads YAML/JSON files first. Variables from
``.env`` or the process environment override these defaults, and any secrets
available via ``VaultSecretProvider`` take precedence over both. In short,
``vault`` > ``env vars`` > ``YAML/JSON``.

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
- `PROMPT_TEMPLATES_PATH` – path to `prompt_templates.v2.json` when running

  outside the repository.
- `METRICS_PORT` – start the internal metrics exporter on this port (same as `--metrics-port`).
- `AUTO_DASHBOARD_PORT` – start the metrics dashboard automatically on this port.
- `SANDBOX_RESOURCE_DB` – path to a `ROIHistoryDB` for resource-aware forecasts.
 - `SANDBOX_BRAINSTORM_INTERVAL` – request brainstorming ideas from local models
   every N cycles.
 - `SANDBOX_BRAINSTORM_RETRIES` – consecutive low-ROI cycles before brainstorming.
 - `SANDBOX_OFFLINE_SUGGESTIONS` – enable heuristic patches when brainstorming is
   unavailable.
 - `SANDBOX_SUGGESTION_CACHE` – JSON file with cached suggestions.
 - `PATCH_SCORE_BACKEND_URL` – optional patch score storage. Use `file://path` to
   save scores locally.
 - `SANDBOX_PRESET_RL_STRATEGY` – RL algorithm used by `AdaptivePresetAgent`
   (default `q_learning`).
 - `SELF_CODING_INTERVAL` – run `SelfCodingEngine` every N cycles.
 - `SELF_CODING_ROI_DROP` – trigger self-coding when ROI drops below this fraction.
 - `SELF_CODING_ERROR_INCREASE` – trigger self-coding when error rate increases by
   this factor.


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
- `PRUNE_INTERVAL=50`
- `SELF_LEARNING_EVAL_INTERVAL=0`
- `SELF_LEARNING_SUMMARY_INTERVAL=0`
- `SELF_TEST_LOCK_FILE=sandbox_data/self_test.lock`
- `SELF_TEST_REPORT_DIR=sandbox_data/self_test_reports`
- `SYNERGY_WEIGHTS_PATH=sandbox_data/synergy_weights.json`
- `ALIGNMENT_FLAGS_PATH=sandbox_data/alignment_flags.jsonl`
- `MODULE_SYNERGY_GRAPH_PATH=sandbox_data/module_synergy_graph.json`
- `RELEVANCY_METRICS_DB_PATH=/path/to/relevancy_metrics.db`
- `RUN_CYCLES=0`
- `RUN_UNTIL=`
 - `METRICS_PORT=8001`  # same as `--metrics-port`
- `AUTO_DASHBOARD_PORT=`
- `SANDBOX_BRAINSTORM_INTERVAL=0`
- `SANDBOX_BRAINSTORM_RETRIES=3`
- `SANDBOX_OFFLINE_SUGGESTIONS=0`
- `SANDBOX_SUGGESTION_CACHE=`
- `PATCH_SCORE_BACKEND_URL=`
 - `SELF_CODING_INTERVAL=300`
 - `SELF_CODING_ROI_DROP=-0.1`
 - `SELF_CODING_ERROR_INCREASE=1.0`
- `AD_API_URL=` (unset or empty disables ad network integration)
- `SELF_TEST_DISABLE_ORPHANS=0`
- `SELF_TEST_DISCOVER_ORPHANS=1`
- `SELF_TEST_RECURSIVE_ORPHANS=1`
- `SELF_TEST_RECURSIVE_ISOLATED=1`

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
``MENACE_SETUP_<KEY>`` (e.g. ``MENACE_SETUP_SERP_API_KEY``). Self-coding
settings are configured locally and require no API keys.

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
The sandbox evaluates patches against a dynamic baseline rather than a fixed
score threshold. Recent composite scores are tracked in a moving window and a
patch is accepted only when its score exceeds the baseline by at least
``DELTA_MARGIN``. The window length, stagnation tolerance and margin are
configured via ``BASELINE_WINDOW``, ``STAGNATION_ITERS`` and ``DELTA_MARGIN`` in
``sandbox_settings.py``.
Sandbox execution helpers live in ``sandbox_runner.py`` with the main entry
point ``_run_sandbox`` used by ``menace_master.py``. When importing
``sandbox_runner.environment`` directly, call ``register_signal_handlers()`` to
clean up pooled containers on ``Ctrl+C`` or ``SIGTERM``.

### WorkflowSandboxRunner

`WorkflowSandboxRunner` executes workflow callables inside an isolated
temporary directory. All file operations are redirected to the sandbox so host
data remains untouched. Setting ``safe_mode=True`` additionally monkeypatches
common HTTP clients and raw sockets so outbound network access raises
``RuntimeError``.

```python
from sandbox_runner import WorkflowSandboxRunner

def writer(path="out.txt"):
    with open(path, "w") as fh:
        fh.write("hello")
    return "done"

runner = WorkflowSandboxRunner()
metrics = runner.run(writer)  # network allowed
print(metrics.modules[0].result)

metrics = runner.run(writer, safe_mode=True)  # network disabled
```

``run`` returns a :class:`RunMetrics` object and the same data is stored on
``runner.telemetry``. Each ``ModuleMetrics`` entry includes ``duration``,
``memory_before``, ``memory_after``, ``memory_delta``, ``memory_peak``,
``success``, ``exception`` and ``result`` fields. ``crash_count`` records how
many modules failed. Callers can interpret these values to evaluate resource
usage and error rates. Telemetry also exposes ``memory_per_module``,
``peak_memory_per_module`` and the overall ``peak_memory`` across all modules.

Key arguments include:

* ``test_data`` – mapping of file paths or URLs to initial contents or
  responses written into the sandbox before execution.
* ``network_mocks`` – mapping of networking helpers (``"requests"``,
  ``"httpx"``, ``"urllib"`` and others) to callables returning custom
  responses.
* ``fs_mocks`` – mapping of filesystem helpers (``"open"``,
  ``"pathlib.Path.write_text"`` …) to callables intercepting file mutations.
* ``module_fixtures`` – per-module ``files`` and ``env`` mappings applied prior
  to running each callable.
* ``safe_mode`` – when enabled, outbound network access is blocked unless a
  corresponding network mock handles the request.

The runner itself does not rely on any environment variables; all behaviour is
controlled via these arguments.

Example network mock:

```python
from sandbox_runner import WorkflowSandboxRunner
import httpx

def fetch():
    return httpx.get("https://example.com").text

runner = WorkflowSandboxRunner()
metrics = runner.run(
    fetch,
    safe_mode=True,
    network_mocks={"httpx": lambda self, method, url, *a, **kw: httpx.Response(200, text="stubbed")},
)
print(metrics.modules[0].result)
```

Example file mock:

```python
from sandbox_runner import WorkflowSandboxRunner

def writer():
    from pathlib import Path
    Path("out.txt").write_text("hi")

written = {}

def capture(path, data, *a, **kw):
    written[path.name] = data
    return len(data)

runner = WorkflowSandboxRunner()
runner.run(writer, fs_mocks={"pathlib.Path.write_text": capture})
print(written["out.txt"])
```

The environment records ``sandbox_data/last_autopurge`` and runs
``purge_leftovers()`` automatically on import when the previous purge is older
than ``SANDBOX_AUTOPURGE_THRESHOLD`` (24h by default). ``purge_leftovers()``
removes any Docker containers or QEMU overlay directories left behind by
previous crashes. Bootstrapping the environment enables the
``sandbox_autopurge.timer`` systemd unit so ``python -m sandbox_runner.cli
--purge-stale`` runs every hour and cleans up leftovers automatically. Active
container IDs and overlay paths are tracked in ``sandbox_data/active_containers.json``
and ``sandbox_data/active_overlays.json`` and are guarded by file locks
(``SANDBOX_POOL_LOCK`` and ``*.lock`` files) so concurrent runs do not corrupt
the records. Two background workers then monitor the pool: a cleanup worker
that removes idle or unhealthy containers and deletes stale overlays, and a
reaper worker that collects orphaned containers. ``schedule_cleanup_check()``
acts as a watchdog, restarting the workers if they exit unexpectedly.

Cleanup events are logged and aggregated metrics are written to
``sandbox_data/cleanup_stats.json``. Failed container launches are recorded in
``sandbox_data/pool_failures.json`` while paths that could not be removed are
stored in ``sandbox_data/failed_cleanup.json`` and ``sandbox_data/failed_overlays.json``.
The cleanup worker retries any items listed in ``failed_cleanup.json`` during
each sweep, removing them from the file once deletion succeeds. Retry attempts
are counted via the ``cleanup_retry_successes`` and ``cleanup_retry_failures``
metrics with cumulative totals persisted in ``cleanup_stats.json``. Worker
durations are exposed through the ``cleanup_duration_seconds`` gauge. Override
the failed cleanup file with ``SANDBOX_FAILED_CLEANUP`` and adjust the alert
threshold for ``report_failed_cleanup`` using ``SANDBOX_FAILED_CLEANUP_AGE``.
Inspect these files or call ``collect_metrics()`` to review the cleanup
history. The metrics dashboard exposes the same information for long term
monitoring and includes ``hours_since_autopurge`` which tracks how long it has
been since the last automatic purge.

When the sandbox isn't running you can still purge leftovers with
``sandbox-runner cleanup`` (``python -m sandbox_runner.cli cleanup``). The
command acquires ``_PURGE_FILE_LOCK`` then calls ``purge_leftovers()`` followed
by ``retry_failed_cleanup()``. Schedule it as a cron job or systemd timer so
stale containers and VM overlays are removed even if the regular cleanup worker
isn't active.

To automate this run on a systemd-based distribution copy the supplied service
and timer files into `/etc/systemd/system/` then enable the timer:

```bash
sudo cp systemd/cleanup.* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cleanup.timer
```

`cleanup.timer` invokes `python -m sandbox_runner.cli cleanup` once at boot and
daily thereafter. Adjust `WorkingDirectory` in `cleanup.service` if the project
resides outside `~/menace_sandbox`.

For a completely autonomous optimisation loop run:

```bash
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('run_autonomous.py'))
PY
)"
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
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)" run-complete presets.json --max-iterations 1
```

The command forwards the presets to ``full_autonomous_run`` and starts the
dashboard when ``--dashboard-port`` (or ``AUTO_DASHBOARD_PORT``) is supplied.

When ``--auto-thresholds`` is enabled the loop recomputes the ROI and synergy
thresholds each iteration. The thresholds are derived from the rolling standard
deviation of recent metrics so manual ``--roi-threshold`` or
``--synergy-threshold`` values are unnecessary.

To execute multiple runs sequentially and launch the metrics dashboard run:

```
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('run_autonomous.py'))
PY
)" --runs 2 --preset-count 2 --dashboard-port 8002

# or simply set AUTO_DASHBOARD_PORT=8002
```

Use `--metrics-port` or `METRICS_PORT` to expose Prometheus gauges from the sandbox.

To optimise across distinct scenarios supply preset files explicitly. The
runner cycles through them when ``--preset-file`` is repeated:

```
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('run_autonomous.py'))
PY
)" --runs 3 \
  --preset-file presets/dev.json \
  --preset-file presets/prod.json \
  --preset-file presets/chaos.json
```

This prints messages such as ``Starting autonomous run 1/2`` followed by the
standard module rankings once each run finishes. Metrics from all runs are
written to ``sandbox_data/roi_history.json`` so they can be aggregated later.

### Sandbox safety

Each optimisation loop now runs a **human‑alignment flagger** on the latest
commit before changes are merged.  The checker scans the diff for removed
docstrings, rising complexity, unsafe patterns and ethics violations.  Any
issues are printed as warnings and recorded as ``alignment_flag`` events in
``logs/audit_log.jsonl`` (mirrored to ``logs/audit_log.db``).  Reviewers consult
these logs and either amend the patch or approve it when the warnings are
acceptable.  A background ``AlignmentReviewAgent`` forwards new warnings to the
``SecurityAuditor`` so Security AI can triage them.  See
[docs/human_alignment_flagger.md](docs/human_alignment_flagger.md) for
configuration details.

Example output::

    Alignment warnings detected:
    - Docstring removed in util/helpers.py.
    - Unsafe code pattern: eval on input().

Warnings at or above ``ALIGNMENT_WARNING_THRESHOLD`` are surfaced for manual
inspection, while scores beyond ``ALIGNMENT_FAILURE_THRESHOLD`` are treated as
high‑risk and typically require corrective commits before integration.
Controls are exposed via ``ENABLE_ALIGNMENT_FLAGGER`` to disable the check and
``ALIGNMENT_BASELINE_METRICS_PATH`` to override the baseline metrics snapshot.

Enable the full alignment pipeline by setting the relevant environment
variables and launching the review agent:

```bash
ENABLE_ALIGNMENT_FLAGGER=1 \
ALIGNMENT_WARNING_THRESHOLD=0.5 \
ALIGNMENT_FAILURE_THRESHOLD=0.9 \
IMPROVEMENT_WARNING_THRESHOLD=0.5 \
IMPROVEMENT_FAILURE_THRESHOLD=0.9 \
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('alignment_review_agent.py'))
PY
)" &  # start reviewer in background
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('run_autonomous.py'))
PY
)"
```

Run the checker directly against two source trees when reviewing changes::

    python human_alignment_flagger.py repo/before repo/after

For automated pipelines import the flagger and act on the reported tiers::

    from human_alignment_flagger import HumanAlignmentFlagger, flag_improvement
    import subprocess, json

    diff = subprocess.check_output(["git", "diff", "HEAD~1"]).decode()
    report = HumanAlignmentFlagger().flag_patch(diff, {"actor": "ci"})
    for issue in report["issues"]:
        print(f"{issue['tier']}: {issue['message']}")

    # Optional: validate proposed workflow changes
    changes = [{"file": "util.py", "code": "print('hi')"}]
    warnings = flag_improvement(changes, None, [])
    print(warnings["maintainability"])

The Menace integrity model interprets tiers as follows:

* ``info`` – recorded for transparency, no action required.
* ``warn`` – developer or reviewer should investigate before merging.
* ``critical`` – treated as integrity violations and usually block deployment
  until resolved or explicitly waived.

### Sandbox self-improvement

See [docs/sandbox_self_improvement.md](docs/sandbox_self_improvement.md) for a
full walkthrough covering package requirements, environment variables, example
workflows and troubleshooting tips. For a quickstart on launching the sandbox
and monitoring metrics, see
[docs/self_improvement.md](docs/self_improvement.md).

#### Required packages

- `sandbox_runner` (including the `sandbox_runner.orphan_integration` module)
- `quick_fix_engine`

The `init_self_improvement` routine verifies these dependencies at startup and
raises a `RuntimeError` with installation guidance when they are missing.

#### Optional packages

- `pandas`, `psutil` and `prometheus-client` for metrics dashboards and resource
  statistics
- `torch` for deep reinforcement learning modules

#### Key environment variables

- `SANDBOX_REPO_PATH` – path to the local sandbox repository clone
- `SANDBOX_DATA_DIR` – directory where metrics and state files are written
- `SANDBOX_ENV_PRESETS` – comma separated scenario preset files
- `AUTO_TRAIN_INTERVAL`, `SYNERGY_TRAIN_INTERVAL`, `ADAPTIVE_ROI_RETRAIN_INTERVAL`
  – control retraining frequency
- `ENABLE_META_PLANNER` – require meta-planning support when set to ``true``
- `BASELINE_WINDOW` – number of recent scores used for the moving average baseline
- `STAGNATION_ITERS` – cycles with no improvement before the baseline resets
- `DELTA_MARGIN` – minimum positive delta over baseline needed to accept a patch

Self-improvement compares ROI gains against this moving baseline and escalates
an internal urgency tier when momentum stalls, encouraging more aggressive
mutations.

#### State snapshots

The sandbox records ROI, entropy and call graph metrics for every cycle via
[`self_improvement/snapshot_tracker.py`](self_improvement/snapshot_tracker.py).
Snapshots and diffs are written to `SNAPSHOT_DIR` / `SNAPSHOT_DIFF_DIR` and
successful changes are checkpointed under `CHECKPOINT_DIR`.  Tune retention and
penalty behaviour with `CHECKPOINT_RETENTION`, `ROI_PENALTY_THRESHOLD` and
`ENTROPY_PENALTY_THRESHOLD` in `SandboxSettings`.

#### Example workflow

```bash
SANDBOX_REPO_PATH=$(pwd) python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)" --runs 1
```

#### Autonomous bootstrap script

``autonomous_bootstrap.py`` streamlines setup by verifying dependencies via
``bootstrap_environment`` and ``init_self_improvement`` before starting the
continuous self-improvement cycle.

```bash
SANDBOX_REPO_PATH=$(pwd) SANDBOX_DATA_DIR=./sandbox_data \
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('autonomous_bootstrap.py'))
PY
)"
```

The script honours the same environment variables as ``sandbox_runner.py``.
Set at least ``SANDBOX_REPO_PATH`` and ``SANDBOX_DATA_DIR`` to point to the
repository and writable data directory.

#### Troubleshooting

- ``ModuleNotFoundError`` for `sandbox_runner` or `quick_fix_engine`: install
  missing packages or rerun `./setup_env.sh`
- ``RuntimeError: ffmpeg not found``: install `ffmpeg` and `tesseract` and
  ensure they are on ``PATH``
- Stale containers or overlays: ``python -m sandbox_runner.cli --cleanup``

### Safe mode, self-improvement intervals and metrics

Run individual modules inside an isolated directory and block outbound
networking by passing ``safe_mode=True`` to
``WorkflowSandboxRunner.run``. Any unexpected network call raises
``RuntimeError`` and file writes outside the sandbox are rejected::

    from sandbox_runner import WorkflowSandboxRunner

    def fetch(url="https://example.com"):
        import httpx
        return httpx.get(url).text

    runner = WorkflowSandboxRunner()
    metrics = runner.run(fetch, safe_mode=True)

``RunMetrics`` exposes ``duration``, ``memory_*`` fields and a per-module
``success`` flag so callers can evaluate resource usage and error rates.
``crash_count`` captures the number of failing modules while
``memory_per_module`` and ``peak_memory`` highlight overall consumption.

#### Configuring self-improvement intervals

Self-improvement cycles retrain the synergy learner periodically. Set
``AUTO_TRAIN_INTERVAL`` or ``SYNERGY_TRAIN_INTERVAL`` (seconds) to adjust how
often weights are updated. ``ADAPTIVE_ROI_RETRAIN_INTERVAL`` controls ROI
retraining frequency. Lower values shorten the feedback loop while higher ones
reduce system load.

#### Personal ``SandboxSettings`` example

Override defaults directly when instantiating ``SandboxSettings`` or via
environment variables:

```python
from sandbox_settings import SandboxSettings

settings = SandboxSettings(
    sandbox_repo_path="/home/alice/menace_sandbox",
    sandbox_data_dir="/home/alice/.menace",
    synergy_train_interval=20,
    roi_cycles=5,
)
```

```
SANDBOX_REPO_PATH=/home/alice/menace_sandbox
SANDBOX_DATA_DIR=/home/alice/.menace
SYNERGY_TRAIN_INTERVAL=20
ROI_CYCLES=5
```

#### Error severity scoring

``SandboxSettings`` exposes ``severity_score_map`` to convert error severity
labels into numeric scores when deciding whether to run a self-improvement
cycle. The defaults are::

```python
{
    "critical": 100.0,
    "crit": 100.0,
    "fatal": 100.0,
    "high": 75.0,
    "error": 75.0,
    "warn": 50.0,
    "warning": 50.0,
    "medium": 50.0,
    "low": 25.0,
    "info": 0.0,
}
```

Override via the ``SEVERITY_SCORE_MAP`` environment variable (JSON encoded) or
when instantiating ``SandboxSettings``.

### Advanced sandbox commands

- ``--auto-thresholds`` recomputes ROI and synergy thresholds every cycle so
  manual ``--roi-threshold`` and ``--synergy-threshold`` values are optional.
  Fine‑tune synergy convergence with ``--synergy-threshold-window`` and
  ``--synergy-threshold-weight``.
- ``--dynamic-workflows`` generates workflows from module groups when the
  workflow database is empty.
- ``--module-algorithm`` chooses the clustering algorithm used for module
  grouping (``greedy``, ``label`` or ``hdbscan``).
- ``--module-threshold`` sets the semantic similarity threshold when grouping
  modules.
- ``--module-semantic`` enables docstring similarity for module clustering.
- Inspect sandbox restart metrics via ``sandbox_recovery_manager.py --file
  sandbox_data/recovery.json``.

Troubleshooting tips:

- Run ``./setup_env.sh && scripts/setup_tests.sh`` when tests fail to ensure
  all dependencies are installed.
- Delete ``sandbox_data/recovery.json`` if ``SandboxRecoveryManager`` keeps
  restarting unexpectedly.
- If synergy metrics diverge wildly, verify that ``synergy_history.db`` is
  writable and consider adjusting ``--synergy-threshold-weight``.
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

Enable central logging by forwarding records to an audit log:

```bash
export SANDBOX_CENTRAL_LOGGING=1
export AUDIT_LOG_PATH=/var/log/menace/audit.log
```

`setup_logging()` attaches an `AuditTrailHandler` so all services write to the
configured path. Set `SANDBOX_CENTRAL_LOGGING=0` to disable forwarding.

Set `SANDBOX_VERBOSE=1` to automatically enable debug logs when
`setup_logging()` is called without a level. The older `SANDBOX_DEBUG=1`
variable is also accepted for backward compatibility.

Enable Prometheus metrics by starting the exporter on a port of your choice:

```bash
export METRICS_PORT=8001
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('run_autonomous.py'))
PY
)"  # or use resolve_path('synergy_tools.py')
```

Add the port to your Prometheus configuration:

```yaml
scrape_configs:
  - job_name: 'menace'
    static_configs:
      - targets: ['localhost:8001']
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
 - ``SELF_CODING_INTERVAL``, ``SELF_CODING_ROI_DROP`` and
   ``SELF_CODING_ERROR_INCREASE`` – configure the local ``SelfCodingEngine``
   that powers code generation without requiring external API keys.
- Stripe billing is configured via ``stripe_billing_router``, which retrieves
  the required keys from a secure vault provider or uses baked‑in production
  values. Avoid storing these keys in the repository or configuration files.
- ``MENACE_SANITY_OPTIONAL`` – set to any value in development to allow missing
  ``menace_sanity_layer`` modules. Without this the Stripe watchdog raises
  ``SanityLayerUnavailableError`` when feedback hooks are invoked.
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

### CLI helpers

- `menace retrieve "query" --db code` performs semantic search across databases. Results are cached on disk and reused until the underlying database timestamps change. If the vector retriever fails, the command falls back to database-specific full-text helpers before returning results.
- `menace patch path/to/module.py --desc "message"` applies a patch using the self-coding engine and prints the new patch identifier followed by provenance records in JSON.
- `menace embed --db workflows` backfills missing vector embeddings, restricting the run to a single database when `--db` is supplied.
- `menace new-db demo` scaffolds a new database module and matching test. The
  generated file already includes a `build_context` helper, FTS schema setup,
  and safety hooks for license detection and secret redaction.  The module is
  automatically added to `__all__` so it can be imported without further
  wiring.

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

`ErrorLogger.log_roi_cap()` uses the `propose_fix` helper from
`roi_calculator` to identify metrics hitting ROI caps and recommends up to
three remediation hints. Default messages live in
`configs/roi_fix_rules.yaml` and can be customised per metric. The resulting
`ROIBottleneck` event powers a feedback loop where low-scoring metrics
automatically raise tickets or craft Codex prompts for suggested patches.
See [docs/roi_calculator.md](docs/roi_calculator.md) for sample integrations.

### Bottleneck Detection Bots

Performance isn’t a luxury; it’s a multiplier on ROI. Each critical function is decorated with an `@perf_monitor` that records wall-clock time via `time.perf_counter` and stores `(function_signature, runtime_ms, cpu_pct, mem_mb)` into `PerfDB`. A nightly cron triggers `bottleneck_scanner.py`, ranking the P95 latencies per module. When a spike exceeds a configurable threshold, the scanner opens a Git issue via API, tagging the responsible bot and auto-assigning the Enhancement Bot.

Technically, this layer relies on `psutil` for cross-platform metrics and `sqlite3` for zero-setup storage. For heavier loads, export to Prometheus + Grafana so you can watch Menace’s pulse in real time. The scanner feeds its findings into the Resource Allocation Optimizer, ensuring slow code is either optimized or throttled.

### Enhancement classifier configuration

`enhancement_classifier.py` scores modules based on patch history and code metrics.
Weights and thresholds live in `enhancement_classifier_config.json`:

```json
{
  "weights": {"frequency": 1.0, ...},
  "thresholds": {
    "min_patches": 3,
    "roi_cutoff": 0.0,
    "complexity_delta": 0.0
  }
}
```

- **min_patches** – minimum number of patches before a file is evaluated.
- **roi_cutoff** – average ROI delta below which a module is flagged.
- **complexity_delta** – minimum average complexity increase to trigger a suggestion.

Tune these values to control how aggressively the classifier surfaces potential improvements.

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
from vector_service.context_builder import ContextBuilder

# supply an explicit builder for downstream modules
builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
creator = BotCreationBot(
    workflow_bot=WorkflowEvolutionBot(),
    trending_scraper=TrendingScraper(),
    context_builder=builder,
)
# tasks is a list of PlanningTask objects
creator.create_bots(tasks)
```

### Universal Retriever

`UniversalRetriever` queries multiple embedding-backed databases and returns
`ResultBundle` objects with final scores and reasons. Scores combine error
frequency, enhancement ROI, workflow usage, bot deployment counts and raw
vector similarity. Related results that share bot relationships receive a
boosted score via relation-aware linking.

```python
from menace.bot_database import BotDB
from menace.task_handoff_bot import WorkflowDB
from menace.error_bot import ErrorDB
from menace.universal_retriever import UniversalRetriever

retriever = UniversalRetriever(bot_db=BotDB(), workflow_db=WorkflowDB(), error_db=ErrorDB())
hits, session_id, vectors = retriever.retrieve("upload failed", top_k=3)
for hit in hits:
    print(f"{hit.origin_db} #{hit.record_id}: {hit.reason} (confidence {hit.confidence:.2f})")
```

The output lists the originating database, record identifier, a normalised
confidence score and a short reason derived from the highest contributing
metric, making it easy to interpret why a candidate was returned.

Use the lightweight `DBRouter` to obtain connections for shared or local
tables:

```python
from db_router import GLOBAL_ROUTER, init_db_router

router = GLOBAL_ROUTER or init_db_router("alpha")
conn = router.get_connection("bots")
```

Entry-point scripts must initialise the router with explicit database paths
**before** importing modules that touch the database:

```python
import os, uuid
from db_router import init_db_router
from dynamic_path_router import resolve_path

MENACE_ID = uuid.uuid4().hex
LOCAL_DB_PATH = os.getenv(
    "MENACE_LOCAL_DB_PATH", str(resolve_path(f"menace_{MENACE_ID}_local.db"))
)
SHARED_DB_PATH = os.getenv(
    "MENACE_SHARED_DB_PATH", str(resolve_path("shared/global.db"))
)
init_db_router(MENACE_ID, LOCAL_DB_PATH, SHARED_DB_PATH)
```

For detailed router configuration, audit logging and log analysis utilities see
[docs/db_router.md](docs/db_router.md).

### Safe mode and overrides

`SelfServiceOverride` now only logs warnings when ROI or error metrics exceed
the configured thresholds. The environment variables `MENACE_SAFE` and
`EVOLUTION_PAUSED` have been removed.

The same thresholds apply&mdash;an ROI drop of more than **10%**, error rates
above **25%** or an energy score below **0.3**&mdash;but safe mode is no longer
toggled automatically. Instead `AutoRollbackService` simply logs a warning
with the offending commit so you can decide whether to revert manually.


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
`MetricsDashboard` exposes Prometheus metrics via a Flask endpoint for easy
visualization. A `/refresh` route pulls the latest telemetry and readiness stats
and pushes a notification to `/refresh/stream` so open dashboards can reload
charts automatically.

Run `start_metrics_server(8001)` before executing
`benchmark_registered_workflows()` to publish metrics that Grafana can chart in
real time. The dashboard examples in
[docs/metrics_dashboard.md](docs/metrics_dashboard.md) show how to track ROI,
resource usage and latency statistics like the new median latency gauge over
time.

### Maintenance tooling
Automated module maintenance is handled by the relevancy radar suite. See
[docs/relevancy_radar.md](docs/relevancy_radar.md) to learn how the
`RelevancyRadar`, `RelevancyRadarService` and `ModuleRetirementService`
collaborate on retirement, compression and replacement actions. The
[Recording Output Impact](docs/relevancy_radar.md#recording-output-impact)
section covers attributing ROI deltas and final output contributions. The suite
relies on the [`networkx`](https://networkx.org/) package for dependency
analysis:

```bash
pip install networkx
```

### Relevancy radar overrides
The relevancy radar tracks how often each module is exercised during runs.
Tune its sensitivity by setting environment variables or values in
`sandbox_settings.py`:

- `RELEVANCY_THRESHOLD` – minimum usage count before a module is considered
  for replacement (default `20`).
- `RELEVANCY_WINDOW_DAYS` – number of days of history to inspect when
  computing relevancy (default `30`). Usage data older than this window is
  discarded during statistics loading and evaluation, ensuring decisions
  reflect recent activity only.
- `RELEVANCY_WHITELIST` – comma-separated modules that the radar should
  never flag.

Example `sandbox_settings.yaml` snippet:

```yaml
relevancy_threshold: 10
relevancy_window_days: 14
relevancy_whitelist:
  - critical_module.py
  - legacy/analytics.py
```

The ``relevancy_radar_cli.py`` helper can annotate modules for retirement,
compression or replacement based on these metrics. Passing ``--final`` will run
the dependency-aware evaluation step described in
[docs/relevancy_radar.md](docs/relevancy_radar.md):

```bash
python relevancy_radar_cli.py --retire old_mod --compress slow_mod --replace alt_mod
```

Annotations are persisted to ``sandbox_data/relevancy_metrics.json`` for later
review.

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
- `MutationLogger` records code changes and `MutationLineage` reconstructs trees and clones branches for A/B tests (see [docs/mutation_lineage.md](docs/mutation_lineage.md)).

### Workflow impact analysis

`WorkflowGraph` maintains a DAG of workflow dependencies and keeps it current by seeding from `WorkflowDB` and listening to workflow events on `UnifiedEventBus`. Self-improvement routines can build the graph and simulate how ROI or synergy changes ripple through downstream modules:

```python
from unified_event_bus import UnifiedEventBus
from workflow_graph import WorkflowGraph

bus = UnifiedEventBus()
graph = WorkflowGraph()
graph.attach_event_bus(bus)

impacts = graph.simulate_impact_wave("42", 0.5, 0.0)
for wid, delta in impacts.items():
    print(wid, delta["roi"], delta["synergy"])
```

- Use the returned mapping to prioritise follow-up improvement cycles.

### TruthAdapter calibration

`TruthAdapter` calibrates ROI predictions and flags feature drift. Instantiate
and train it with sandbox metrics and a profit proxy:

```python
from menace.truth_adapter import TruthAdapter

adapter = TruthAdapter()
adapter.fit(X, y)
metrics, drift = adapter.check_drift(X_recent)
preds, low_conf = adapter.predict(X_recent)
```

`low_conf` becomes `True` when drift metrics exceed their thresholds. The
adapter's state persists to `sandbox_data/truth_adapter.pkl`; set
`ENABLE_TRUTH_CALIBRATION=0` to disable calibration. When drift occurs, gather
fresh samples and call `fit` again or run:

```bash
python self_improvement.py fit-truth-adapter live.npz shadow.npz
```

See [docs/truth_adapter.md](docs/truth_adapter.md) for details and additional
configuration options.


## Module Synergy Grapher

Builds a weighted graph of module relationships and queries related modules.
See [docs/module_synergy_grapher.md](docs/module_synergy_grapher.md) for graph
components, scoring, CLI options and workflow examples using
`get_synergy_cluster`.

This tool requires the optional [`networkx`](https://networkx.org/) package:

```bash
pip install networkx
```

```bash
python module_synergy_grapher.py --build [--config cfg.toml]
python module_synergy_grapher.py --cluster <module> --threshold 0.8
```

The `--threshold` flag controls cluster expansion.  `--config` points to a
JSON/TOML file whose `coefficients` section overrides the default weighting.
The `ModuleSynergyGrapher` constructor accepts the same via its `config`
parameter or a raw `coefficients` mapping.  `make synergy-graph` can be used in
automation to refresh the graph.

## Workflow Synthesizer

Combines structural relationships from `ModuleSynergyGrapher` with semantic
intent search to propose small, ordered module workflows. It inspects module
inputs and outputs to resolve dependency order and emits candidate steps.

### CLI usage

Use `workflow_synthesizer_cli.py` to explore and save workflows from the
command line:

```bash
python workflow_synthesizer_cli.py --start module_a --save
python workflow_synthesizer_cli.py --start "summarise data" --save summary
python workflow_synthesizer_cli.py --list
```

The first example seeds the synthesizer from an existing module and stores the
result under `sandbox_data/generated_workflows/module_a.workflow.json`. The
second accepts a free-text problem description and saves the spec using the
provided name. The `--list` flag prints previously saved workflow files.

Candidates can be exported to `.workflow.json` for registration with
`WorkflowDB`. The JSON contains a `steps` list describing each module's inputs
and outputs, making it consumable by the sandbox evaluation pipeline. Once
stored in `WorkflowDB`, evaluation services and workers can execute and score
the workflow alongside existing ones. See
[docs/workflow_synthesizer.md](docs/workflow_synthesizer.md) for design notes
and more examples.

## Vector Analytics

`vector_metrics_analytics` surfaces stored vector metrics for quick
inspection. ROI trends and ranking weight changes can be printed from the
command line:

```bash
python -m vector_metrics_analytics --roi-summary --days 30
python -m vector_metrics_analytics --weight-summary
```

Use `--db` to point at an alternative `vector_metrics.db` file.

### Dependencies

Some features rely on optional third‑party libraries. Missing packages trigger
warnings and degrade functionality gracefully.

- `pandas` – enables DataFrame operations in performance assessment and metrics
  queries.
- `psutil` – provides detailed CPU, memory and I/O statistics for the data bot.
- `prometheus-client` – exposes collected metrics via a Prometheus endpoint.

Install the extras with:

```bash
pip install pandas psutil prometheus-client
```

## Service Configuration

### Self-Learning Service

| Environment variable | Description | Default |
| --- | --- | --- |
| `SELF_LEARNING_PERSIST_EVENTS` | Optional path where the event bus persists state | unset |
| `SELF_LEARNING_PERSIST_PROGRESS` | Optional path storing evaluation results on shutdown | unset |
| `PRUNE_INTERVAL` | Number of new interactions before pruning GPT memory | `50` |

### Self-Test Service

| Environment variable | Description | Default |
| --- | --- | --- |
| `SELF_TEST_LOCK_FILE` | File used to serialise self-test runs | `sandbox_data/self_test.lock` |
| `SELF_TEST_REPORT_DIR` | Directory used to store self-test reports | `sandbox_data/self_test_reports` |

Example `.env` snippet for personal deployments:

```env
SELF_LEARNING_PERSIST_EVENTS=/var/menace/events.db
SELF_LEARNING_PERSIST_PROGRESS=/var/menace/progress.json
PRUNE_INTERVAL=100
SELF_TEST_LOCK_FILE=/var/menace/self_test.lock
SELF_TEST_REPORT_DIR=/var/menace/reports
```

## Legal Notice

See [LEGAL.md](LEGAL.md) for the full legal terms. In short, this project may
only be used for lawful and ethical activities. The authors do not condone
malicious or unlawful behaviour. The software is provided **as-is** without any
warranties, and the maintainers accept no liability for damages arising from its
use.

