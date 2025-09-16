# Autonomous Sandbox

This guide describes the prerequisites and environment variables used when
running the fully autonomous sandbox.

## Configuration

`SandboxSettings` reads configuration from the environment or an optional
`.env` file. The most important variables are:

- `MENACE_MODE` – operating mode (`test` or `production`).
- `DATABASE_URL` – database connection string.
- `SANDBOX_REPO_PATH` – repository root used by background services.
- `SANDBOX_DATA_DIR` – directory where sandbox state and metrics are stored.
 - `SELF_CODING_INTERVAL`, `SELF_CODING_ROI_DROP` and `SELF_CODING_ERROR_INCREASE`
   – tune the local `SelfCodingEngine`; code generation runs entirely offline and
   requires no API keys.
  - `LOCAL_KNOWLEDGE_REFRESH_INTERVAL` – refresh interval for the local
  knowledge module (default `600`).
- `MENACE_LOCAL_DB_PATH` and `MENACE_SHARED_DB_PATH` – override default SQLite
  database locations.
- `OPTIONAL_SERVICE_VERSIONS` – JSON mapping of optional service modules to
  minimum versions, e.g.
  `{"relevancy_radar": "1.2.0", "quick_fix_engine": "1.1.0"}`.

The `.env` file referenced by `MENACE_ENV_FILE` is loaded automatically when
present. Additional configuration files such as synergy weights or self‑test
locks are created inside `SANDBOX_DATA_DIR` during execution.

For information on built-in scenario profiles, hostile input stubs and
concurrency options, see the sections on [Predefined Profiles](sandbox_runner.md#predefined-profiles),
[Hostile Input Stub Strategy](sandbox_runner.md#hostile-input-stub-strategy) and
[Concurrency Settings](sandbox_runner.md#concurrency-settings) in
`sandbox_runner.md` as well as [Predefined Profiles](environment_generator.md#predefined-profiles),
[Hostile Input Stub Strategy](environment_generator.md#hostile-input-stub-strategy) and
[Concurrency Settings](environment_generator.md#concurrency-settings) in
`environment_generator.md`.

## Sandboxed workflow execution

Workflows can be exercised in isolation via
`WorkflowSandboxRunner`. Each invocation runs the supplied callable inside a
fresh temporary directory and redirects all file operations to that location so
the host file system remains untouched. Test fixtures may pre-populate files by
passing a ``test_data`` mapping and assert on results with
``expected_outputs``.

Setting ``safe_mode=True`` disables outbound network access by monkeypatching
common HTTP clients such as ``urllib``, ``requests`` and ``httpx``. Any network
attempt raises ``RuntimeError``, which is returned to the caller instead of
propagating the exception.

Additional instrumentation or telemetry can be attached through the
``mock_injectors`` hook. Injectors receive the sandbox root path and can
monkeypatch modules to record activity or provide further isolation. Each
injector returns a teardown callable that is executed after the workflow
completes to restore the original state.

## Upgrade forecasting

See [UpgradeForecaster](upgrade_forecaster.md) for projecting ROI and risk across upcoming improvement cycles.

## Stability gating

The sandbox's `SelfImprovementEngine` queries
`ForesightTracker.predict_roi_collapse` before promoting a workflow. Promotions
are blocked when the forecast reports a high risk or brittleness:

```python
from menace_sandbox.foresight_tracker import ForesightTracker
from menace_sandbox.self_improvement import SelfImprovementEngine
from context_builder_util import create_context_builder

builder = create_context_builder()
engine = SelfImprovementEngine(
    context_builder=builder,
    foresight_tracker=ForesightTracker(),
)
wf = "workflow-1"
risk = engine.foresight_tracker.predict_roi_collapse(wf)
if risk["risk"] in {"Immediate collapse risk", "Volatile"} or risk["brittle"]:
    engine.enqueue_preventative_fixes([wf])
    print("promotion blocked")
else:
    print("workflow is safe to promote")
```

This check prevents unstable trajectories from reaching production.

Possible risk labels returned by `predict_roi_collapse` are:

- **Stable** – non-negative slope with volatility below the configured threshold.
- **Slow decay** – gently negative slope while volatility remains low.
- **Volatile** – volatility above the threshold regardless of slope.
- **Immediate collapse risk** – steep negative slope or a projected drop below zero in the next cycles.

The result also includes a `brittle` flag when small entropy changes cause outsized ROI drops.

## Workflow evolution

`SelfImprovementEngine` embeds a `WorkflowEvolutionManager` to refine workflow
definitions. The manager benchmarks the current sequence with
`CompositeWorkflowScorer`, generates variants via
`WorkflowEvolutionBot.generate_variants` and promotes the highest‑ROI option.
The `limit` argument controls how many variants are produced (default `5`).

Tune the gating behaviour with the environment variables
`ROI_GATING_THRESHOLD` and `ROI_GATING_CONSECUTIVE`. Evolution is skipped when
the exponential moving average of ROI deltas fails to exceed these thresholds.

Example configuration:

```bash
export ROI_GATING_THRESHOLD=0.05
export ROI_GATING_CONSECUTIVE=5
```

Once the gate fires the manager logs messages such as:

```
INFO workflow 42 stable (ema gating)
```

Example usage:

```python
from menace_sandbox.self_improvement import (
    SelfImprovementEngine,
    ImprovementEngineRegistry,
)
from context_builder_util import create_context_builder

builder = create_context_builder()
engine = SelfImprovementEngine(context_builder=builder, bot_name="alpha")
registry = ImprovementEngineRegistry()
registry.register_engine("alpha", engine)
results = registry.run_all_cycles()
print(results["alpha"].workflow_evolution)
```

The `workflow_evolution` field summarises whether a variant was promoted or the
workflow was deemed stable.

### WorkflowSynergyComparator

The self‑improvement loop uses `WorkflowSynergyComparator` to decide whether a
candidate workflow should merge into an existing branch or remain a separate
variant. The helper exposes:

- `WorkflowSynergyComparator.compare(a, b)` – return a
- :class:`SynergyScores` exposing `similarity`, `shared_module_ratio`,
  per‑workflow entropy, efficiency, modularity and an aggregate score.
- Overfitting reports (`overfit_a` / `overfit_b`) indicating low entropy or
  repeated modules. Use :meth:`WorkflowSynergyComparator.analyze_overfitting`
  to inspect a single workflow.
- `WorkflowSynergyComparator.is_duplicate(scores, thresholds=None)` – flag
  near‑identical workflows using the scores returned by
  `WorkflowSynergyComparator.compare()`.
- `WorkflowSynergyComparator.merge_duplicate(base_id, dup_id)` – merge the
  duplicate into the canonical workflow.

Entropy and similarity thresholds control merging behaviour. When
``scores.similarity`` meets or exceeds ``WORKFLOW_MERGE_SIMILARITY`` and the
entropy delta stays below ``WORKFLOW_MERGE_ENTROPY_DELTA`` the manager merges
the variant. Relaxed thresholds encourage more merging while stricter values
favour branching and independent evolution. Duplicate detection thresholds for
`is_duplicate` default to ``WORKFLOW_DUPLICATE_SIMILARITY`` and
``WORKFLOW_DUPLICATE_ENTROPY``.

Example API usage inside the self‑improvement loop:

```python
from menace_sandbox.workflow_synergy_comparator import WorkflowSynergyComparator

scores = WorkflowSynergyComparator.compare("main_flow", "candidate_flow")
if WorkflowSynergyComparator.is_duplicate(
    scores, {"similarity": 0.96, "entropy": 0.04}
):
    WorkflowSynergyComparator.merge_duplicate("main_flow", "candidate_flow")
```

Run the autonomous loop with custom thresholds via environment variables:

```bash
WORKFLOW_MERGE_SIMILARITY=0.96 WORKFLOW_MERGE_ENTROPY_DELTA=0.04 \
WORKFLOW_DUPLICATE_SIMILARITY=0.95 WORKFLOW_DUPLICATE_ENTROPY=0.05 \
  python run_autonomous.py
```

## Foresight promotion gate

Before final promotion the sandbox invokes
``foresight_gate.is_foresight_safe_to_promote``.  The helper evaluates
four gating conditions and returns ``(ok, reason_codes, forecast)``:

1. all projected ROI values meet or exceed the supplied ``roi_threshold``;
2. forecast ``confidence`` is at least ``0.6``;
3. :func:`ForesightTracker.predict_roi_collapse` reports neither immediate
   collapse risk nor a ``collapse_in`` value within the forecast horizon; and
4. ``WorkflowGraph.simulate_impact_wave`` reports no negative downstream ROI
   deltas.

If any condition fails the caller downgrades the workflow to the borderline
bucket when one is configured, otherwise a micro‑pilot run is scheduled.
Reason codes – ``projected_roi_below_threshold``, ``low_confidence``,
``roi_collapse_risk`` and ``negative_dag_impact`` – reveal which gate triggered
the downgrade.

### Example: borderline/pilot downgrade

```python
from menace_sandbox.deployment_governance import evaluate
from menace_sandbox.foresight_tracker import ForesightTracker
from menace_sandbox.borderline_bucket import BorderlineBucket

bucket = BorderlineBucket("sandbox_data/borderline_bucket.jsonl")
tracker = ForesightTracker()

result = evaluate(
    {"alignment": {"status": "pass", "rationale": ""}},
    {"raroi": 1.0, "confidence": 0.8},
    patch=["step_a"],
    foresight_tracker=tracker,
    workflow_id="wf-1",
    borderline_bucket=bucket,
)
print(result["verdict"], result["reason_codes"])
# -> 'borderline', ['low_confidence']
```

Dropping ``borderline_bucket`` from the call yields ``verdict: 'pilot'`` for the
same failing condition.

## GPT Interaction Tags

All GPT interactions are recorded with a standard tag so that feedback and
generated fixes can be queried later.  The canonical labels are:

- **FEEDBACK** – commentary or evaluation of previous behaviour.
- **IMPROVEMENT_PATH** – suggested steps or strategies for improvement.
- **ERROR_FIX** – information about a bug or how it was resolved.
- **INSIGHT** – novel ideas or observations not tied to a specific error.

Use :func:`memory_logging.log_with_tags` to enforce these tags when storing a
prompt/response pair:

```python
from memory_logging import log_with_tags

log_with_tags(memory, prompt, response, tags=[FEEDBACK])
```

The helper normalises any supplied labels and discards unknown values so every
interaction ends up with one of the standard tags above.

## Scenario Types

The sandbox consumes presets generated by `environment_generator`. Canonical
scenarios include:

- **high_latency_api** – artificial network delay.
- **hostile_input** – adversarial payload generation.
- **user_misuse** – invalid API calls and file access attempts.
- **concurrency_spike** – bursts of threads and async tasks.

Each profile exposes both "low" and "high" severity levels. The helper
`generate_canonical_presets()` returns a mapping of scenarios to these levels so
every module can be exercised under both intensities:

```python
from environment_generator import generate_canonical_presets
from vector_service.context_builder import ContextBuilder

presets = generate_canonical_presets()
# access individual levels
low_latency = presets["high_latency_api"]["low"]
high_latency = presets["high_latency_api"]["high"]
# run all scenarios under both severities
builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
run_repo_section_simulations(
    repo_path, env_presets=presets, context_builder=builder
)
```

Set `SANDBOX_FAIL_ON_MISSING_SCENARIOS=1` to raise an error if any canonical
scenario lacks coverage.

Generate presets for specific scenarios and supply them to the sandbox:

```bash
python environment_cli.py generate --profiles hostile_input high_latency_api --out presets.json
python run_autonomous.py --preset-file presets.json --runs 1
```

## Configuring and Extending Presets

Preset JSON files can be edited directly or produced on the fly with
`environment_cli.py`. Pass multiple `--preset-file` options to rotate through
different scenarios or set `SANDBOX_ENV_PRESETS` to a JSON string for ad-hoc
runs. Setting `SANDBOX_PRESET_MODE=canonical` loads the built‑in profiles
without needing an external file.

## Interpreting Per-Scenario Metrics

Each preset contains a `SCENARIO_NAME` which is recorded alongside ROI and
synergy metrics. After a run, inspect `sandbox_data/roi_history.json` or
`sandbox_data/metrics.db` and filter by this name to compare behaviour across
scenarios. A low `synergy_resilience` during a `hostile_input` run highlights
weak resistance to malicious stubs. The `sandbox_dashboard.py` utility and the
metrics dashboard expose these per-scenario values.

## RAROI-driven ranking

`ROITracker.calculate_raroi()` converts raw ROI into a risk-adjusted value that
penalises volatile or unsafe workflows. The autonomous sandbox uses this
RAROI when ranking modules and deciding whether to integrate or skip them.
Impact severities come from `config/impact_severity.yaml` (override with
`IMPACT_SEVERITY_CONFIG`) and `SelfTestService` feeds error rates and failing
test names so high-risk modules drop in priority. See the
[ROI tracker](roi_tracker.md#risk-adjusted-roi) and [RAROI overview](raroi.md)
for the full formula.

## Borderline bucket and micro‑pilots

When a workflow's RAROI drops below ``BORDERLINE_RAROI_THRESHOLD`` or the
confidence score falls under ``BORDERLINE_CONFIDENCE_THRESHOLD`` it is queued in the
[borderline bucket](borderline_bucket.md) instead of being discarded. The bucket
stores recent RAROI values and the latest confidence so a lightweight
**micro‑pilot** can collect additional evidence. ``MICROPILOT_MODE`` controls how
these candidates are handled:

- ``auto`` – queue and immediately run a micro‑pilot.
- ``queue`` – queue only; a later run must call
  ``process_borderline_candidates``.
- ``off`` – disable the bucket.

Run the sandbox with automatic micro‑pilots enabled:

```bash
MICROPILOT_MODE=auto BORDERLINE_RAROI_THRESHOLD=0.1 \
python run_autonomous.py --runs 1
```

## Entropy delta tracking and module completion

`ROITracker` derives a Shannon entropy ratio for each patched module by comparing
the latest `synergy_shannon_entropy` metric with the change in code complexity.
These ratios accumulate in `module_entropy_deltas` and are used to decide when a
module has stabilised. If the mean ROI gain per entropy delta falls below
`ENTROPY_THRESHOLD` or a module's ratios stay under
`ENTROPY_PLATEAU_THRESHOLD` for `ENTROPY_PLATEAU_CONSECUTIVE` cycles, the
section is marked complete and skipped in future improvement rounds.

Tune the sensitivity with the `--entropy-threshold`/`ENTROPY_THRESHOLD` flag and
the `--consecutive`/`ENTROPY_PLATEAU_CONSECUTIVE` window size or rely on the
default threshold returned by `ROITracker.diminishing()`. Additional gates such
as `ROI_THRESHOLD`,
`SYNERGY_THRESHOLD`, `ROI_CONFIDENCE`, `SYNERGY_CONFIDENCE`,
`SYNERGY_STATIONARITY_CONFIDENCE` and `SYNERGY_VARIANCE_CONFIDENCE` can further
restrict automatic completion. When a module is flagged the runner emits a
`sandbox diminishing` log listing the affected sections so you can review them or
reset their history. Flagged modules are persisted to ``*.flags`` files so later
cycles automatically skip them.

## Temporal trajectory simulations

Use :func:`simulate_temporal_trajectory` to exercise a workflow through a
series of entropy stages and measure how performance degrades. The helper
progresses through ``normal``, ``high_latency``, ``resource_strain``,
``schema_drift`` and ``chaotic_failure`` presets, each introducing additional
network delay, resource pressure or data corruption. The function accepts a
``workflow_id`` and automatically loads the associated steps from
``WorkflowDB``. For every stage the sandbox records the scenario's name under a
``stage`` field together with the ROI, the associated resilience score and how
far the ROI falls from the baseline run. A ``stability`` value derived from the
rolling window summarises how rapidly performance decays across the stages.

Supplying a :class:`ForesightTracker` records these measurements with
``compute_stability=True`` so the tracker stores per‑stage ROI/resilience values,
the scenario degradation and the computed ``stability`` reading:

```python
from sandbox_runner.environment import simulate_temporal_trajectory
from foresight_tracker import ForesightTracker

foresight = ForesightTracker()
simulate_temporal_trajectory(workflow_id, foresight_tracker=foresight)
```

Each cycle logged in ``foresight.history`` includes ``stage``, ``roi_delta``,
``resilience``, ``scenario_degradation`` and the computed ``stability`` score,
allowing long‑term decay to be modelled as stability drops between stages.

## Human alignment flagger

Each autonomous cycle runs `HumanAlignmentFlagger` against the most recent
commit. The checker parses the Git diff, highlighting removed docstrings,
logging statements, missing tests and potential ethics or risk/reward issues.
Results are appended to `sandbox_data/alignment_flags.jsonl` and published on
the event bus as `alignment:flag`.

Security AI or developers should monitor the JSON log or subscribe to the event
bus to review any warnings. Alerts labelled `alignment_warning` are raised when
scores exceed `ALIGNMENT_WARNING_THRESHOLD`; values beyond
`ALIGNMENT_FAILURE_THRESHOLD` warrant immediate investigation. The flagger can
be tuned or disabled via `ENABLE_ALIGNMENT_FLAGGER` and
`ALIGNMENT_BASELINE_METRICS_PATH`.

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

8. **Start the autonomous loop** in a terminal as described below.

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

### Recursive orphan discovery and integration

The sandbox walks the repository for modules with no inbound references and
recursively follows their imports. The helper script
`scripts/discover_isolated_modules.py` locates standalone files with no
references elsewhere in the tree. Setting `SANDBOX_DISCOVER_ISOLATED=1` (or
passing `--discover-isolated`) runs this scan before the orphan pass, while
`SANDBOX_AUTO_INCLUDE_ISOLATED=1` (or `--auto-include-isolated`) feeds the
results directly into the self‑tests. Isolated modules are included in the scan
and their dependencies are discovered recursively by default
(`SANDBOX_RECURSIVE_ISOLATED=1`). `sandbox_runner.discover_recursive_orphans`
starts at each orphan and, when `SANDBOX_RECURSIVE_ORPHANS=1`, follows the
import chain so that helper files are discovered automatically. Each candidate
module is executed by `SelfTestService`; all isolated modules run before final
classification so redundancy is based on runtime behaviour. Modules that pass are appended to
`sandbox_data/module_map.json` while failures are logged and skipped. Setting
`SANDBOX_CLEAN_ORPHANS=1` cleans processed names out of
`sandbox_data/orphan_modules.json` after a successful run. Disable recursion with
`SANDBOX_RECURSIVE_ISOLATED=0` (or `--no-recursive-isolated`) and
`SANDBOX_RECURSIVE_ORPHANS=0` (or `--no-recursive-orphans`/`--no-recursive-include`).
With `SANDBOX_AUTO_INCLUDE_ISOLATED=1` isolated files participate in the same
scan. These defaults are enabled by `auto_env_setup.ensure_env()` through the
corresponding `SELF_TEST_*` variables.

Modules tagged as *redundant* or *legacy* can also be exercised during this
walk. Setting `SANDBOX_TEST_REDUNDANT=1` (or passing
`--include-redundant`/`--test-redundant`) keeps these modules in the candidate
set so their helper chains are traversed and executed like any other orphan.
Set `SANDBOX_TEST_REDUNDANT=0` to skip them while still recording their
classification.

#### Environment variables and CLI flags

The sandbox enables recursion and isolated-module inclusion by default. Key
flags and their associated environment variables are summarised below:

| Variable | Default | CLI flag | Purpose |
|---------|---------|---------|---------|
| `SANDBOX_RECURSIVE_ORPHANS` | `1` | `--recursive-orphans` / `--no-recursive-orphans` (`--recursive-include`/`--no-recursive-include`) | Follow dependencies of orphan modules. |
| `SANDBOX_RECURSIVE_ISOLATED` | `1` | `--recursive-isolated` / `--no-recursive-isolated` | Include dependencies of isolated modules. |
| `SANDBOX_DISCOVER_ISOLATED` | `1` | `--discover-isolated` / `--no-discover-isolated` | Scan for modules with no inbound references before the orphan pass. |
| `SANDBOX_AUTO_INCLUDE_ISOLATED` | `1` | `--auto-include-isolated` | Queue isolated modules for self‑tests automatically. |
| `SANDBOX_CLEAN_ORPHANS` | `1` | `--clean-orphans` | Remove processed entries from `sandbox_data/orphan_modules.json`. |
| `SANDBOX_TEST_REDUNDANT` | `1` | `--include-redundant` / `--test-redundant` | Run tests for modules marked as redundant or legacy. |

Use the CLI flags `--recursive-orphans`/`--no-recursive-orphans` (aliases
`--recursive-include`/`--no-recursive-include`), `--recursive-isolated` or
`--no-recursive-isolated`, `--discover-isolated`, `--auto-include-isolated`,
`--include-redundant`/`--test-redundant` and `--clean-orphans` to override the
behaviour. Each flag mirrors an environment variable
(`SANDBOX_RECURSIVE_ORPHANS`, `SANDBOX_RECURSIVE_ISOLATED`,
`SANDBOX_DISCOVER_ISOLATED`, `SANDBOX_AUTO_INCLUDE_ISOLATED`,
`SANDBOX_TEST_REDUNDANT`, `SANDBOX_CLEAN_ORPHANS`) which can be set directly as
needed. The redundant flag also sets `SELF_TEST_INCLUDE_REDUNDANT` so the test
runner honours the choice.

To toggle recursion via the environment instead of CLI flags:

```bash
# skip orphan and isolated recursion
export SANDBOX_RECURSIVE_ORPHANS=0
export SANDBOX_RECURSIVE_ISOLATED=0
# force isolated modules to be included recursively
export SANDBOX_AUTO_INCLUDE_ISOLATED=1
# disable redundant module tests
export SANDBOX_TEST_REDUNDANT=0
```

Example CLI usage:

```bash
# disable orphan recursion but force isolated modules to be included
python -m sandbox_runner.cli --no-recursive-orphans --auto-include-isolated

# run a full autonomous cycle with explicit recursion flags
python run_autonomous.py --recursive-orphans --recursive-isolated

# include modules tagged as redundant or legacy (default)
python run_autonomous.py --include-redundant

# skip redundant modules during tests
SANDBOX_TEST_REDUNDANT=0 python run_autonomous.py --discover-orphans
```

#### Classification and metrics storage

`discover_recursive_orphans` records analysis results in
`sandbox_data/orphan_classifications.json` alongside the primary orphan list
`sandbox_data/orphan_modules.json`. Passing modules are appended to
`sandbox_data/module_map.json` while self‑test statistics and orphan metrics are
persisted to `sandbox_data/metrics.db` for later inspection.

### Example: auto‑detecting and integrating an isolated module

1. **Create an isolated module** that is not imported anywhere else:

   ```bash
   echo "def ping(): return 'pong'" > isolated_ping.py
   ```

2. **Run the autonomous cycle** (recursion and isolated inclusion are enabled by default):

   ```bash
   python run_autonomous.py
   ```

3. The pre‑scan discovers `isolated_ping` and queues it for `SelfTestService`.
   If the test passes, the module is appended to `sandbox_data/module_map.json`
   and removed from `sandbox_data/orphan_modules.json`.

4. Subsequent runs treat the module as part of the workflow system without
   further manual intervention.

`run_autonomous` and `sandbox_runner` mirror these values to the matching
`SELF_TEST_*` variables so that `SelfTestService` observes the same behaviour.

- `AUTO_SANDBOX=1` – run the sandbox on first launch
- `SANDBOX_CYCLES=5` – number of self‑improvement iterations
- `SANDBOX_ROI_TOLERANCE=0.01` – ROI delta required to stop early
- `RUN_CYCLES=0` – unlimited run cycles for the orchestrator
- `AUTO_DASHBOARD_PORT=8001` – start the metrics dashboard
- `METRICS_PORT=8001` – start the internal metrics exporter (same as `--metrics-port`)
- `EXPORT_SYNERGY_METRICS=1` – enable the Synergy Prometheus exporter
- `SYNERGY_METRICS_PORT=8003` – port for the exporter
- `SELF_TEST_METRICS_PORT=8004` – port exposing self‑test metrics
- `SELF_TEST_DISABLE_ORPHANS=0` – include orphan module discovery and execution (set to `1` to skip)
- `SELF_TEST_DISCOVER_ORPHANS=1` – automatically scan for orphan modules (set to `0` to disable)
 - `SELF_TEST_DISCOVER_ISOLATED=1` – automatically discover isolated modules (set to `0` or use `--no-discover-isolated` to disable)
 - `SELF_TEST_AUTO_INCLUDE_ISOLATED=1` – queue results from `discover_isolated_modules` for self‑testing (set to `0` to disable)
 - `SELF_TEST_RECURSIVE_ISOLATED=1` – recursively process isolated modules (set to `0` or use `--no-recursive-isolated` to disable)
- `SELF_TEST_RECURSIVE_ORPHANS=1` – recursively follow orphan dependencies
  (`--no-recursive-include` to disable)
- `SELF_TEST_DISABLE_AUTO_INTEGRATION=0` – enable automatic integration of passing modules (`1` to disable)
- `SANDBOX_DISABLE_ORPHANS=1` – disable orphan testing when running via `sandbox_runner`
- `SANDBOX_DISABLE_ORPHAN_SCAN=1` – skip orphan discovery during improvement cycles
- `SANDBOX_DISCOVER_ISOLATED=1` – run `discover_isolated_modules` during orphan scans (set to `0` or use `--no-discover-isolated` to disable)
- `SANDBOX_RECURSIVE_ORPHANS=1` – recurse through orphan dependencies when refreshing the module map (default; set to `0` or use `--no-recursive-include` to disable)
- `SANDBOX_RECURSIVE_ISOLATED=1` – recurse through isolated modules when
  building the module map (set to `0` or use `--no-recursive-isolated` to disable)
- `SANDBOX_CLEAN_ORPHANS=1` – prune processed names from `orphan_modules.json`
  after successful integration
 - `SANDBOX_AUTO_INCLUDE_ISOLATED=1` – automatically discover isolated modules and recurse through them when scanning (`--auto-include-isolated`; set to `0` to disable)
- `ROI_THRESHOLD` – override the diminishing ROI threshold
- `ROI_CONFIDENCE` – t-test confidence when flagging modules
- `ENTROPY_PLATEAU_THRESHOLD` – entropy delta threshold for plateau detection
- `ENTROPY_PLATEAU_CONSECUTIVE` – entropy samples below threshold before flagging
- `MIN_INTEGRATION_ROI` – minimum ROI delta required before adding modules to existing workflows
- `SYNERGY_THRESHOLD` – fixed synergy convergence threshold
- `SYNERGY_THRESHOLD_WINDOW` – samples used for adaptive synergy threshold
- `SYNERGY_THRESHOLD_WEIGHT` – exponential weight for threshold calculation
- `SYNERGY_CONFIDENCE` – confidence level for synergy convergence checks
- `SYNERGY_STATIONARITY_CONFIDENCE` – confidence for stationarity tests
- `SYNERGY_VARIANCE_CONFIDENCE` – confidence for variance tests
- `SANDBOX_PRESET_RL_PATH` – path to the RL policy used for preset adaptation
- `SANDBOX_PRESET_RL_STRATEGY` – reinforcement learning algorithm
- `SANDBOX_ADAPTIVE_AGENT_PATH` – path to the adaptive RL agent state
- `SANDBOX_ADAPTIVE_AGENT_STRATEGY` – algorithm for the adaptive agent
- `ADAPTIVE_ROI_TRAIN_INTERVAL` – seconds between scheduled adaptive ROI training
- `ADAPTIVE_ROI_RETRAIN_INTERVAL` – cycles between adaptive ROI model retraining
- `SELF_IMPROVEMENT_BACKUP_COUNT` – number of rotated backups to keep for self-improvement data

### Recursion environment variables

The sandbox and the self‑test service share a set of flags that control how
deeply orphaned or isolated modules are followed and merged into the workflow
database:

- `SELF_TEST_RECURSIVE_ORPHANS` / `SANDBOX_RECURSIVE_ORPHANS` – recurse through
  orphan modules and their imports.
- `SELF_TEST_RECURSIVE_ISOLATED` / `SANDBOX_RECURSIVE_ISOLATED` – include
  dependencies of isolated modules.
- `SANDBOX_AUTO_INCLUDE_ISOLATED` – force inclusion of modules returned by
  `discover_isolated_modules`.
- `SANDBOX_CLEAN_ORPHANS` – drop passing entries from `orphan_modules.json`
  after integration.

### Recursive inclusion workflow

`environment.auto_include_modules(mods, recursive=True)` expands each entry by
following its local imports and, when `SANDBOX_AUTO_INCLUDE_ISOLATED=1`, appends
files reported by `scripts.discover_isolated_modules`. Passing modules and their
helpers are written to `sandbox_data/module_map.json`. Modules classified as
redundant are left in `sandbox_data/orphan_modules.json` unless
`SANDBOX_TEST_REDUNDANT=1` instructs the sandbox to integrate them. Set
`SANDBOX_RECURSIVE_ORPHANS=0` to disable the import walk or
`SANDBOX_AUTO_INCLUDE_ISOLATED=0` to skip isolated discovery.

### Orphan discovery helpers

`sandbox_runner.discover_recursive_orphans` traces orphan candidates and, when
`SANDBOX_RECURSIVE_ORPHANS=1`, follows their imports so helper modules are
queued automatically. The cycle calls `sandbox_runner.cycle.include_orphan_modules`
to load names from `sandbox_data/orphan_modules.json` and pass them to
`SelfTestService`. Once a candidate and its dependencies pass,
`environment.auto_include_modules` merges the files into
`sandbox_data/module_map.json`. When `recursive=True` (the default when
`SANDBOX_RECURSIVE_ORPHANS=1` or `SANDBOX_RECURSIVE_ISOLATED=1`) the helper
also discovers local imports during integration. Modules classified as legacy
increment the `orphan_modules_legacy_total` Prometheus gauge and the value is
decremented when those entries are reclassified or integrated.

#### Surfacing orphan chains

`discover_recursive_orphans` starts from the names listed in
`sandbox_data/orphan_modules.json` and walks their `import` statements
recursively. The helper returns a mapping where each entry notes which
`parents` pulled it into the chain so the sandbox can queue every
dependent module in one pass.

#### Validating and integrating candidates

After discovery, `auto_include_modules` executes the candidate set with
`SelfTestService`. Modules whose simulated ROI increase is below
`MIN_INTEGRATION_ROI` are skipped. When validation succeeds the modules
and any helpers found during recursion are merged into
`sandbox_data/module_map.json` and patched into existing workflows so
future cycles exercise the new code.

#### Monitoring orphan module metrics

The inclusion flow records its progress via the Prometheus gauges
`orphan_modules_tested_total`, `orphan_modules_reintroduced_total`,
`orphan_modules_failed_total`, `orphan_modules_redundant_total` and
`orphan_modules_legacy_total`. Start the metrics server with
`metrics_exporter.start_metrics_server(PORT)` and point Prometheus at
the port to observe how many orphan modules are tested, reintroduced or
skipped over time.

Example running a simulation that discovers orphans and includes them
recursively:

```bash
python run_autonomous.py --discover-orphans --include-orphans --recursive-include
```

### Redundant modules

`sandbox_runner.discover_recursive_orphans` returns a mapping where each
module includes the modules that imported it in `parents` and carries a
`redundant` flag. `SelfTestService` records this metadata and the improvement
engine only integrates modules whose `redundant` flag is false. Redundant
modules are logged but skipped during automatic inclusion.

`run_autonomous.py` mirrors the `SANDBOX_*` values to the matching
`SELF_TEST_*` variables so that `SelfTestService` honours the same recursion
strategy. Passing modules and their helpers are written to
`module_map.json` and existing flows via `try_integrate_into_workflows`.
Integration uses repository-relative paths to avoid filename collisions.
Tests such as `tests/test_recursive_isolated.py` and
`tests/test_self_test_service_recursive_integration.py` exercise this path and
assert that both isolated modules and their dependencies are executed and
integrated. `tests/integration/test_auto_include_isolated_dependency.py`
demonstrates that `environment.auto_include_modules` pulls in a standalone
module together with its local imports and records redundant entries in
`orphan_modules.json`.

### Automatic recursion for isolated modules

The sandbox can automatically pick up modules that are otherwise disconnected
from the rest of the codebase. Setting `SANDBOX_AUTO_INCLUDE_ISOLATED=1`
or passing `--auto-include-isolated` forces `discover_isolated_modules` to run
and implicitly sets `SANDBOX_DISCOVER_ISOLATED=1` and
`SANDBOX_RECURSIVE_ISOLATED=1` unless they are overridden. With
`SANDBOX_RECURSIVE_ISOLATED=1` their import chains are traversed so supporting
files are executed alongside the target module, while
`SANDBOX_RECURSIVE_ORPHANS=1` lets deeper orphan dependencies join the scan.
Once the tests succeed, the modules are merged into `module_map.json` and
existing flows through `try_integrate_into_workflows`. Entries flagged by
`orphan_analyzer.analyze_redundancy` are skipped to avoid duplicating
functionality.

Recursion through orphan dependencies is enabled by default. Disable it by
setting `SELF_TEST_RECURSIVE_ORPHANS=0` or `SANDBOX_RECURSIVE_ORPHANS=0`, or
pass `--no-recursive-include` to the relevant commands.

### Example: isolated module discovery and integration

```bash
# create a trivial isolated module and its helper
echo 'import helper\n' > isolated.py
echo 'VALUE = 1\n' > helper.py

# test the isolated module inside the sandbox and pull in its helper
python -m menace.self_test_service run isolated.py \
    --auto-include-isolated --recursive-isolated

# integrate the module and clean the orphan cache during the next cycle
python run_autonomous.py --auto-include-isolated --include-orphans \
    --recursive-isolated --clean-orphans
```

`discover_isolated_modules` locates both files and `SelfTestService` executes
them recursively. Non-redundant modules are appended to `sandbox_data/module_map.json`
and scheduled automatically via `try_integrate_into_workflows`. With
`--clean-orphans` (or `SANDBOX_CLEAN_ORPHANS=1`) the processed names are removed
from `sandbox_data/orphan_modules.json`. See
`tests/test_self_test_service_recursive_integration.py::test_recursive_isolated_integration`
for an automated assertion of this workflow.

### Example: recursive orphan inclusion with cleanup

```bash
# create an orphan and its local helper
echo 'import helper\n' > orphan.py
echo 'VALUE = 1\n'   > helper.py

# run the self tests recursively and prune passing entries
python -m menace.self_test_service run --recursive-include --clean-orphans

# merge the modules during the next autonomous cycle
python run_autonomous.py --include-orphans --recursive-include --clean-orphans
```

After the run both `orphan.py` and `helper.py` are appended to
`module_map.json`. Because `--clean-orphans` (or `SANDBOX_CLEAN_ORPHANS=1`)
was used, the processed names are removed from
`sandbox_data/orphan_modules.json`. Add `--auto-include-isolated` and
`--recursive-isolated` when the starting file is not referenced anywhere
else so the sandbox discovers isolated modules in the same manner.

### Example: reintroducing a dormant module

```bash
# assume util.py was previously removed from the workflow system
echo 'def util(): return 1' > util.py
echo '["util.py"]' > sandbox_data/orphan_modules.json
python run_autonomous.py --include-orphans
# util.py is merged into module_map.json and metrics reflect the reintroduction
```

SelfCodingEngine runs locally so no additional API keys are required.

## Launch sequence

1. Generate a `.env` file or update an existing one using `auto_env_setup.ensure_env()`:

   ```bash
   python -c 'import auto_env_setup as a; a.ensure_env()'
   ```

   Review the generated file and add any missing secrets.

2. Launch the autonomous loop:

   ```bash
   python run_autonomous.py
   ```

   Use `--metrics-port` or `METRICS_PORT` to expose Prometheus metrics from the sandbox.

   The script verifies system dependencies, creates default presets using `environment_generator` and invokes the sandbox runner. The metrics dashboard is available at `http://localhost:${AUTO_DASHBOARD_PORT}` once started.
   If the previous run terminated unexpectedly you can append `--recover` to reload the last recorded ROI and synergy histories.
   Use `--preset-debug` to enable verbose logging of preset adaptation. Combine it with `--debug-log-file <path>` to write these logs to a file for later inspection.
   Use `--forecast-log <path>` to store ROI forecasts and threshold values for each run.
   The optional [ForesightTracker](foresight_tracker.md) captures these cycle metrics and checks short‑term trend stability. Cold start templates live in `configs/foresight_templates.yaml`.
   Use `--dynamic-workflows` to build temporary workflows from module groups when the workflow database is empty. Control the clustering with `--module-algorithm`, `--module-threshold` and `--module-semantic` which mirror the options of `discover_module_groups`.

   Example:

   ```bash
   python run_autonomous.py --dynamic-workflows \
     --module-semantic --module-threshold 0.25
   ```

    Example scanning for orphan modules:

    ```bash
     python run_autonomous.py --discover-orphans --auto-include-isolated \
       --include-orphans
    ```

   Use `--no-recursive-include` or `--no-recursive-isolated` to skip dependency
   traversal.

   Set `SELF_TEST_INCLUDE_ORPHANS=1` to achieve the same behaviour via
   environment variables. Recursion is enabled by default via
   `SELF_TEST_RECURSIVE_ORPHANS=1` and `SELF_TEST_RECURSIVE_ISOLATED=1` (or
   `SANDBOX_RECURSIVE_ORPHANS=1` and `SANDBOX_RECURSIVE_ISOLATED=1` when
   launching via `sandbox_runner`); set them to `0` or use the corresponding
   `--no-recursive-*` options to disable recursion. Modules that pass their
    tests are merged into `module_map.json` and existing flows are updated via
    `try_integrate_into_workflows` so the new code can be benchmarked
    immediately. Modules flagged as redundant by
    `orphan_analyzer.analyze_redundancy` are skipped during this integration.

## Task processing

`run_autonomous.py` handles submissions directly and executes them sequentially
inside the sandbox. State is stored in `SANDBOX_DATA_DIR` so unfinished work
resumes automatically after restarts.
## Local run essentials

The sandbox reads several paths and authentication tokens from environment variables. These defaults are suitable for personal deployments and can be overridden in your `.env`:

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
SANDBOX_DATA_DIR=~/menace_data
PATCH_SCORE_BACKEND_URL=http://example.com/api/scores
DATABASE_URL=sqlite:///menace.db
BOT_DB_PATH=bots.db
BOT_PERFORMANCE_DB=bot_performance_history.db
MAINTENANCE_DB=maintenance.db
SELF_TEST_DISABLE_ORPHANS=0
SELF_TEST_DISCOVER_ORPHANS=1
SANDBOX_RECURSIVE_ORPHANS=1
SANDBOX_RECURSIVE_ISOLATED=1
SELF_CODING_INTERVAL=5
```

Edit the resulting `.env` to override any defaults as needed.

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

2. **Launch the autonomous loop**:

   ```bash
   python run_autonomous.py
   ```

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
python -m menace.self_improvement synergy-dashboard --file sandbox_data/synergy_history.db
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

#### Task concurrency

The sandbox processes one job at a time. Submit tasks sequentially and wait for
completion before sending the next request.

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
from menace.self_improvement.api import SelfImprovementEngine, DQNSynergyLearner
from context_builder_util import create_context_builder

engine = SelfImprovementEngine(
    context_builder=create_context_builder(),
    synergy_learner_cls=DQNSynergyLearner,
    synergy_weights_path="sandbox_data/synergy_weights.json",
)
engine.run_cycle()
```

This variant relies on PyTorch and persists the Q‑network weights alongside the
JSON file. See [synergy_learning.md](synergy_learning.md) for background on how
the learner adjusts the metrics.

## Orphan and isolated module discovery

### Recursive orphan discovery

Orphan discovery surfaces modules that are not referenced anywhere else so the
sandbox can test and integrate them instead of leaving potentially useful code
behind. Before each run `sandbox_runner.discover_recursive_orphans` walks the
import tree of known orphans and collects their local dependencies, while
`scripts.discover_isolated_modules` locates standalone files with no inbound
references. The resulting paths are passed to
`run_repo_section_simulations(..., modules=modules)` for sandbox execution. A
`SelfTestService` instance exercises each module and, when the module passes, it
is appended to `sandbox_data/module_map.json`. Entries are indexed using
repository‑relative paths so files with the same name in different directories
remain distinct. `environment.generate_workflows_for_modules`
then creates one‑step workflows so later simulations include the newly
discovered functionality.

### Automatic self-testing and integration

The discovery helpers feed an automated pipeline that promotes passing modules:

1. `sandbox_runner.discover_recursive_orphans` and `scripts.discover_isolated_modules` gather orphaned or isolated files.
2. `SelfTestService` runs each candidate in isolation and reports failures without halting the sandbox.
3. Successful modules are appended to `sandbox_data/module_map.json` and `environment.generate_workflows_for_modules` creates one‑step workflows so future runs schedule them automatically.

`SandboxSettings` exposes flags to tune this behaviour:

- `auto_include_isolated` (`SANDBOX_AUTO_INCLUDE_ISOLATED`) forces isolated discovery.
- `recursive_orphan_scan` (`SANDBOX_RECURSIVE_ORPHANS`) follows orphan dependencies.
- `recursive_isolated` (`SANDBOX_RECURSIVE_ISOLATED`) expands isolated modules through their imports.

### Environment variables and CLI flags

Discovery behaviour can be tuned through CLI flags or their environment
variable counterparts:

- `--discover-orphans` / `SELF_TEST_DISCOVER_ORPHANS=1` – scan for modules with
  no inbound references.
- `--auto-include-isolated` / `SANDBOX_AUTO_INCLUDE_ISOLATED=1` /
  `SELF_TEST_AUTO_INCLUDE_ISOLATED=1` – force inclusion of modules returned by
  `discover_isolated_modules`.
- `--recursive-include` / `SANDBOX_RECURSIVE_ORPHANS=1` /
  `SELF_TEST_RECURSIVE_ORPHANS=1` – follow orphan dependencies.
- `--recursive-isolated` / `SANDBOX_RECURSIVE_ISOLATED=1` /
  `SELF_TEST_RECURSIVE_ISOLATED=1` – pull in dependencies of isolated modules.
- `--clean-orphans` / `SANDBOX_CLEAN_ORPHANS=1` – drop entries for files that no
  longer exist.

Disable the scans entirely with `SELF_TEST_DISABLE_ORPHANS=1` or
`SANDBOX_DISABLE_ORPHAN_SCAN=1`. When `SANDBOX_AUTO_INCLUDE_ISOLATED=1`,
`environment.auto_include_modules` merges passing modules into the module map
and generates one‑step workflows for them automatically.

### Classification and metrics storage

`discover_recursive_orphans` records classifications in
`sandbox_data/orphan_classifications.json` alongside the primary cache
`sandbox_data/orphan_modules.json`. During sandbox cycles `MetricsDB` logs test
results and module counts to `sandbox_data/metrics.db` so Prometheus exporters
can expose gauges such as `orphan_modules_tested_total`.

### Dependency expansion and validation

`environment.auto_include_modules` recursively discovers helpers for each
candidate by calling `dependency_utils.collect_local_dependencies`. When
`SANDBOX_RECURSIVE_ORPHANS=1` or `SANDBOX_RECURSIVE_ISOLATED=1`, this walk
follows each import chain so supporting files are queued alongside the target
module. Use `SANDBOX_MAX_RECURSION_DEPTH` or `--max-recursion-depth` to cap how
deep these chains are explored. The collected set is executed by `SelfTestService`, which validates the
isolated module and only forwards passing paths back to
`auto_include_modules` for integration into `sandbox_data/module_map.json`.

**Triggering recursive inclusion**

Enable recursion directly from the command line:

```bash
python -m menace.self_test_service run module.py --recursive-include --recursive-isolated
```

or via environment variables:

```bash
SANDBOX_RECURSIVE_ORPHANS=1 SANDBOX_RECURSIVE_ISOLATED=1 \
  python run_autonomous.py --auto-include-isolated
```

Both approaches ensure that helper modules discovered through
`collect_local_dependencies` are validated by `SelfTestService` before being
merged into the workflow.

### Walkthrough

1. `sandbox_runner.discover_recursive_orphans` and
   `scripts.discover_isolated_modules` identify unreferenced files.
2. The resulting set is passed to `run_repo_section_simulations` and exercised by
   `SelfTestService`.
3. Passing modules are appended to `sandbox_data/module_map.json` via
   `environment.auto_include_modules`.
4. `environment.generate_workflows_for_modules` creates placeholder workflows so
   future sandbox cycles can schedule the new functionality.

### Example: integrating an isolated module with dependencies

1. **Create a package and helper module**:

   ```bash
   mkdir -p demo_pkg
   cat <<'PY' > demo_pkg/util.py
   def double(x):
       return x * 2
   PY
   ```

2. **Add an entry point that imports the helper**:

   ```bash
   cat <<'PY' > demo_pkg/main.py
   import demo_pkg.util

   def run():
       return demo_pkg.util.double(21)
   PY
   ```

3. **Run discovery and tests**:

   ```bash
   SANDBOX_AUTO_INCLUDE_ISOLATED=1 SANDBOX_RECURSIVE_ISOLATED=1 \
     python -m sandbox_runner.cli --discover-orphans --auto-include-isolated
   ```

4. **Observe integration** – `SelfTestService` executes both files. Passing
   modules are written to `sandbox_data/module_map.json` and
   `environment.generate_workflows_for_modules` creates single‑step workflows for
   future cycles.

### Example: handling duplicate filenames

```bash
mkdir -p pkg_a pkg_b
echo 'VALUE=1' > pkg_a/common.py
echo 'VALUE=2' > pkg_b/common.py
SANDBOX_AUTO_INCLUDE_ISOLATED=1 \
  python -m sandbox_runner.cli pkg_a/common.py pkg_b/common.py --auto-include-isolated
# module_map.json lists pkg_a/common.py and pkg_b/common.py separately
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

## Troubleshooting

- **Missing dependencies** – run `./setup_env.sh` again to ensure all Python
  packages are installed. On bare metal verify that `ffmpeg` and `tesseract`
  are present in your `$PATH`.
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
- **Orphan modules not discovered** – ensure `SELF_TEST_DISABLE_ORPHANS` and `SANDBOX_DISABLE_ORPHAN_SCAN` are unset. Use `--discover-orphans` or set `SELF_TEST_DISCOVER_ORPHANS=0` only when disabling the automatic scans.
