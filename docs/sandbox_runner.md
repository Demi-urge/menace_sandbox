# Sandbox Runner

`sandbox_runner.py` executes self-improvement cycles without creating a fresh
repository clone. Temporary databases and event logs are created so production
data remains untouched.

The sandbox uses the existing repository defined by `SANDBOX_REPO_PATH`. This is
expected to be a checkout of `https://github.com/Demi-urge/menace_sandbox` and
is modified and evaluated directly.

## Multi-environment Setup

`_run_sandbox` copies the repository into a new directory and overrides environment variables such as `DATABASE_URL`, `BOT_DB_PATH`, `BOT_PERFORMANCE_DB` and `MAINTENANCE_DB`. Each cycle runs under these temporary paths and the original values are restored afterwards. The optional `SANDBOX_RESOURCE_DB` variable points to a `ROIHistoryDB` used for resource-aware forecasts.

## Multi-environment Section Simulations

When `env_presets` provides multiple dictionaries each extracted section is executed under every preset. ROI deltas and metric values are grouped per preset so behavioural differences become obvious. The tracker aggregates ROI and metric histories under scenario names so presets can later be ranked.

`run_repo_section_simulations` now executes these presets concurrently using `asyncio`. Each snippet runs in its own subprocess while updates to `ROITracker` are applied sequentially once the task completes. This significantly reduces the runtime when many scenarios are tested.

When running a standard sandbox session `_sandbox_main` also iterates over `SANDBOX_ENV_PRESETS` for each section. The preset variables are merged into the temporary environment before `_cycle()` executes and metric names are prefixed with the scenario so per-preset trends can be analysed.

## Full-environment Mode

When the environment variable `SANDBOX_ENV_PRESETS` contains a JSON list of
configuration dictionaries `_run_sandbox` iterates over them. Each preset runs in
the existing repository. `simulate_full_environment` starts this checkout inside
a Docker container using `_execute_in_container`. The container writes its
`roi_history.json` back into the mounted repository so the host can aggregate ROI
deltas. After each run the ROI delta and the last recorded `security_score` are
stored via `ROITracker` so presets can be ranked by their impact.

Set `SANDBOX_DOCKER=1` to launch each preset in a container. The repository is
mounted read/write and the resulting `sandbox_data/roi_history.json` file is
collected after the container exits.

If `OS_TYPE` is set to `windows` or `macos` and Docker isn't available the
sandbox starts a lightweight VM instead. `simulate_full_environment` uses QEMU
with the parameters from `VM_SETTINGS` to boot the image and mount the repository
using a 9p share. Provide a `windows_image` or `macos_image` entry
in `VM_SETTINGS` pointing to your QEMU disk file. The VM runs `sandbox_runner.py`
just like the host or container. After shutdown the generated `roi_history.json`
is read from the shared directory and merged into the main ROI tracker.

## Scenario Generation

When no presets are supplied `run_repo_section_simulations` and `_run_sandbox`
call `environment_generator.generate_presets()` to build random scenarios. The
generator now varies CPU, memory and disk limits, introduces failure modes and
simulates network conditions such as latency and packet loss. New parameters
like `SECURITY_LEVEL` and `THREAT_INTENSITY` tune the security posture while
`NETWORK_JITTER_MS`, `MIN_BANDWIDTH`, `MAX_BANDWIDTH` and `PACKET_DUPLICATION`
allow for finer grained network emulation. Set `SANDBOX_GENERATE_PRESETS=0` to
disable this behaviour and fall back to a single empty preset. Pass
`--preset-count N` on the command line to control how many presets the
generator creates (default is `3`).
See [environment_generator.md](environment_generator.md) for a description of the generated variables.

`adapt_presets()` further tweaks these scenarios before each run. The function
raises CPU, memory and bandwidth limits when ROI gains stall and lowers them
again once improvements resume. Positive synergy ROI also decreases
`NETWORK_LATENCY_MS` while increasing `MAX_BANDWIDTH`; negative synergy does the
opposite. These adjustments keep resource usage aligned with recent trends.

## Environment Variables

Several environment variables influence how each section runs. `CPU_LIMIT` and
`MEMORY_LIMIT` apply resource limits via `resource.setrlimit` while `DISK_LIMIT`
caps the workspace size. `TIMEOUT` defines how long a snippet may run before it
is aborted. When these limits cannot be enforced the sandbox monitors resource
usage via `psutil` and stops the process or falls back to Docker execution.
Network behaviour can be tuned with:

- `NETWORK_LATENCY_MS` – base latency in milliseconds added through `tc netem`.
- `NETWORK_JITTER_MS` – additional jitter for the delay.
- `PACKET_LOSS` – percentage of packets randomly dropped.
- `PACKET_DUPLICATION` – percentage of packets duplicated.
- `MIN_BANDWIDTH`/`MAX_BANDWIDTH` and `BANDWIDTH_LIMIT` – values used by the
  preset generator for future bandwidth throttling.
- `SECURITY_LEVEL` and `THREAT_INTENSITY` – adjust the simulated security
  posture.
- `SANDBOX_PATCH_RETRIES` – number of patch attempts `SelfDebuggerSandbox`
  performs before giving up (default `3`).

When any network variables are set and the `tc` binary is available the sandbox
temporarily applies a `netem` queueing discipline to the loopback interface
before executing the snippet. The settings are removed afterwards so subsequent
cycles are unaffected.

## Runtime Simulation

`simulate_execution_environment` performs a static safety check on snippets. Set
`SANDBOX_DOCKER=1` or pass `container=True` to execute the snippet in a minimal
Docker/Podman container with the specified resource limits (`CPU_LIMIT`,
`MEMORY_LIMIT` and `DISK_LIMIT`). CPU usage, memory consumption and disk I/O are
recorded and returned in a `runtime_metrics` dictionary.

## Section Targeting

`run_repo_section_simulations(repo_path, input_stubs=None, env_presets=None)` analyses each Python file with `_extract_sections` and simulates execution for every section. A `ROITracker` records risk flags per module and section so trends can be ranked. Pass custom input stubs to exercise different code paths and provide multiple environment presets to compare behaviour across configurations. Set `return_details=True` to receive raw results grouped by preset.

When no `input_stubs` argument is supplied the function calls `generate_input_stubs()` from `environment.py`. This helper first inspects the target function signature (when available) and derives argument dictionaries from defaults and type hints. The `SANDBOX_INPUT_STUBS` variable overrides this behaviour. When unset, history or template files are consulted before falling back to the signature, a smart faker-based strategy, a synthetic language-model strategy or a random strategy. Plugins discovered via `SANDBOX_STUB_PLUGINS` may augment or override the generated stubs.

Sections with declining ROI trigger dedicated improvement cycles. Only the flagged section is iteratively modified while metrics are tracked. When progress stalls the sandbox issues a GPT‑4 brainstorming request if `SANDBOX_BRAINSTORM_INTERVAL` is set. Consecutive low‑ROI cycles before brainstorming can be tuned via `SANDBOX_BRAINSTORM_RETRIES`.

`_SandboxMetaLogger.diminishing()` evaluates these ROI deltas using a rolling mean and standard deviation over the last `consecutive` cycles. A module is flagged when the mean is within the given threshold and the standard deviation falls below a small epsilon, preventing sporadic fluctuations from triggering improvements.

## Workflow Simulations

`run_workflow_simulations(workflows_db, env_presets=None)` replays stored
workflow sequences under each environment preset. ROI deltas and metrics are
aggregated per workflow ID. After iterating over individual workflows a single
combined snippet containing all steps is executed. Metrics from this run are
recorded under the module name `all_workflows`, allowing the tracker to show the
overall behaviour across every workflow.

## Synergy Metrics

After each section has been simulated individually the sandbox executes a
combined phase containing every previously flagged section. The ROI and metrics
from this run are compared against the average of the individual section runs.
The differences are stored under `synergy_roi` and `synergy_<metric>` entries in
`roi_history.json`. The delta is also attributed to each involved module so
`ROITracker.rankings()` reflects the overall cross‑module impact.

`run_workflow_simulations()` performs the same comparison for entire workflows.
Each workflow is executed on its own and then as part of a single combined
snippet. `synergy_roi` and `synergy_<metric>` values capture how the result of
the combined workflow run differs from the average metrics recorded during the
individual runs. These synergy metrics are stored alongside the regular
per-section values so cross‑workflow effects can be analysed in the same way as
section‑level interactions.

Additional synergy metrics introduced recently track cross-module changes to:

- `synergy_shannon_entropy`
- `synergy_flexibility`
- `synergy_energy_consumption`
- `synergy_profitability`
- `synergy_revenue`
- `synergy_efficiency`
- `synergy_antifragility`
- `synergy_resilience`
- `synergy_projected_lucrativity`
- `synergy_adaptability`
- `synergy_safety_rating`
- `synergy_maintainability`
- `synergy_code_quality`
- `synergy_network_latency`
- `synergy_throughput`
- `synergy_risk_index`
- `synergy_recovery_time`

### Forecasting Synergy Metrics

`ROITracker` can forecast the next `synergy_roi` or `synergy_<metric>` value once
enough history is recorded. Call `predict_synergy()` or
`predict_synergy_metric()` to estimate the combined effect of upcoming runs.
Positive predictions indicate that modules are expected to reinforce each other
while negative values signal possible interference.

Example snippet:

```python
from menace.roi_tracker import ROITracker

tracker = ROITracker()
tracker.metrics_history["synergy_security_score"] = [0.01, 0.03, 0.02]
tracker.roi_history = [0.2, 0.4, 0.5]

pred = tracker.predict_synergy_metric("security_score")
rel = tracker.reliability(metric="synergy_security_score")
print("next synergy security", pred, rel)
```

These forecasts help interpret whether the collaboration between sections or
workflows is trending in a positive or negative direction. The reliability
score summarises how close recent synergy predictions were to the actual
measurements; values near `1` suggest highly consistent forecasts.

## GPT‑4 Integration

When `OPENAI_API_KEY` is set the sandbox requests improvements from GPT‑4 after ROI gains diminish. Suggestions are applied via `SelfCodingEngine` and reverted if they fail to increase ROI. Set `SANDBOX_BRAINSTORM_INTERVAL` to a positive integer to periodically ask GPT‑4 for high level ideas during the run. Use `SANDBOX_BRAINSTORM_RETRIES` to specify how many consecutive low‑ROI cycles trigger extra brainstorming.
## Metric Tracking and Prediction Bots

Sandbox cycles record extended metrics such as `security_score`, `safety_rating`,
`adaptability`, `antifragility`, `shannon_entropy`, `flexibility`,
`efficiency` and `projected_lucrativity`. `security_score` is derived from
Bandit scan results of the touched modules. `adaptability` and `flexibility`
use line coverage information when a `.coverage` file is present, falling back
to simple module counts otherwise. Additional values including
`profitability`, `patch_complexity`, `energy_consumption`, `resilience`,
`network_latency`, `throughput`, `risk_index` and `recovery_time` are also tracked. The bundled
prediction plugin generates forecasts for all of these metrics so their future
trends are available even when the raw values remain unchanged.
`ROITracker` aggregates these per section and can request forecasts through a
`PredictionManager`. Bots registered with the manager return predictions which
are stored via `predict_metric_with_manager`.

## Adaptive ROI Tolerance

`_sandbox_main` calls `ROITracker.reliability()` every cycle and scales `roi_tolerance` based on the returned score. Reliable forecasts lower the threshold so the run ends sooner, while noisy predictions increase it.

## Patch Verification Loop

`SelfDebuggerSandbox.analyse_and_fix` retries failed patches. After each
attempt the generated tests are executed again. If the tests still fail the
debugger fetches new telemetry and tries again up to `SANDBOX_PATCH_RETRIES`
times.

Recent patch metrics are kept in a rolling history. The sandbox calculates
the mean and standard deviation for coverage change, error reduction, ROI
delta and patch complexity. `_composite_score` normalises the current
values using these statistics and applies adaptive weights. Metrics with a
consistent improvement (low variance) have a stronger influence while
increased complexity is penalised based on how volatile it usually is.

## Metrics Plugins

Custom metric collectors can be added without modifying `sandbox_runner.py`. Set
the environment variable `SANDBOX_METRICS_PLUGIN_DIR` to a directory containing
Python files or list directories under the `plugin_dirs` key in
`SANDBOX_METRICS_FILE`. Each file must implement a `collect_metrics` function:

```python
def collect_metrics(prev_roi: float, roi: float, resources: dict | None) -> dict:
    """Return additional metrics based on the latest cycle."""
```

All plugin results are merged with the built-in metrics before they are passed
to `ROITracker.update`. Example plugin:

```python
# plugins/custom.py
def collect_metrics(prev_roi, roi, resources):
    cpu = resources.get("cpu", 0.0) if resources else 0.0
    return {"cpu_delta": cpu - 50.0}
```

Run the sandbox with `SANDBOX_METRICS_PLUGIN_DIR=/path/to/plugins` to enable the
plugin. Alternatively add plugin paths to your metrics configuration file:

```yaml
plugin_dirs:
  - plugins
```


### Prediction-enabled Plugins

Plugins can request forecasts for their own metrics using `PredictionManager`.
When loading plugins, pass the manager instance so they can call it as needed.

```python
from menace.prediction_manager_bot import PredictionManager
from sandbox_runner.metrics_plugins import load_metrics_plugins

manager = PredictionManager(data_bot=data_bot)
plugins = load_metrics_plugins("plugins")
for plugin in plugins:
    if hasattr(plugin, "register"):
        plugin.register(manager)
```

Example plugin generating a predicted CPU delta:

```python
# plugins/cpu_predict.py
manager = None

def register(pm):
    global manager
    manager = pm

def collect_metrics(prev_roi, roi, resources):
    cpu = resources.get("cpu", 0.0) if resources else 0.0
    pred = 0.0
    if manager:
        pred = manager.registry[next(iter(manager.registry))].bot.predict_metric(
            "cpu_delta", [cpu]
        )
    return {"cpu_delta": cpu - 50.0, "cpu_delta_pred": pred}
```

Run the sandbox as before and the predicted value will be logged alongside the
actual metric.

### Metrics Prediction Plugin

This repository includes a plugin that predicts many of the built-in metrics,
including `security_score`, `safety_rating`, `adaptability`, `antifragility`,
`shannon_entropy`, `efficiency`, `flexibility`, `projected_lucrativity`,
`profitability`, `patch_complexity`, `energy_consumption`, `resilience`,
`network_latency`, `throughput`, `risk_index` and `recovery_time`. Enable it by setting the
environment variable `SANDBOX_METRICS_PLUGIN_DIR` to the directory containing
the plugin file:

```bash
SANDBOX_METRICS_PLUGIN_DIR=plugins python sandbox_runner.py
```

The plugin stores its forecasts with `ROITracker.record_metric_prediction` so
you can evaluate prediction accuracy via `rolling_mae_metric()`. Register the
plugin with both a `PredictionManager` and your tracker before running:

```python
manager = PredictionManager(data_bot=data_bot)
tracker = ROITracker()
plugins = load_metrics_plugins("plugins")
for plug in plugins:
    if hasattr(plug, "register"):
        plug.register(manager, tracker)
tracker.predict_all_metrics(manager, [0.0, 0.1])  # store predictions
print(
    tracker.predicted_metrics["flexibility"][-1],
    tracker.predicted_metrics["antifragility"][-1],
    tracker.predicted_metrics["shannon_entropy"][-1],
)
```

### Synergy Prediction Plugin

`plugins/synergy_predict.py` predicts `synergy_roi` and all recorded
`synergy_<metric>` values using the bots registered with `PredictionManager`.
Forecasts are stored via `ROITracker.record_metric_prediction` so their
accuracy can be inspected later. Enable the plugin through
`SANDBOX_METRICS_PLUGIN_DIR` and register it just like the metrics plugin:

```python
manager = PredictionManager(data_bot=data_bot)
tracker = ROITracker()
plugins = load_metrics_plugins("plugins")
for plug in plugins:
    if hasattr(plug, "register"):
        plug.register(manager, tracker)
```

After each cycle predicted synergy values are available under keys such as
`pred_synergy_roi` in the returned metrics dictionary.

### Predicting Security & Adaptability

`ROITracker` can also request forecasts for its built-in metrics. After a few
cycles you may want to predict how security or adaptability will evolve:

```python
from menace.roi_tracker import ROITracker
from menace.prediction_manager_bot import PredictionManager

tracker = ROITracker()
manager = PredictionManager()

tracker.update(1.0, 1.1, metrics={"security_score": 0.8, "adaptability": 0.6})

sec_pred = tracker.predict_metric_with_manager(manager, "security_score", [0.8])
adapt_pred = tracker.predict_metric_with_manager(manager, "adaptability", [0.6])
print("predicted security", sec_pred)
print("predicted adaptability", adapt_pred)
```

The returned values are stored alongside the actual metrics so forecast accuracy
can be evaluated via `rolling_mae_metric()`.

## Prompt Template

`build_section_prompt` automatically loads every `.j2` file found in
`templates/auto_prompts/` and chooses one based on recent metrics from
`ROITracker`. A falling `security_score` will select `security.j2`, poor
efficiency chooses `efficiency.j2` and ROI stagnation triggers `roi.j2`.
If no metric stands out the templates are cycled at random. Custom templates can
still be provided via `GPT_SECTION_TEMPLATE`, but this is no longer required.
The template receives the section name, recent metric values, ROI deltas and
any extracted code snippet. Set `GPT_SECTION_PROMPT_MAX_LENGTH` to limit the
size of the rendered prompt – snippet and metric text are truncated when this
value is exceeded.

`SelfCodingEngine.build_visual_agent_prompt` can also be customised via three
environment variables:

- `VA_PROMPT_TEMPLATE` – path to a template (or inline template string) used to
  build the visual agent prompt. The template receives `{path}`, `{description}`,
  `{context}` and `{func}` placeholders.
- `VA_PROMPT_PREFIX` – additional text prepended before the generated prompt.
- `VA_REPO_LAYOUT_LINES` – number of repository layout lines to include.

Example:

```bash
VA_PROMPT_PREFIX="[internal]" VA_PROMPT_TEMPLATE=va.tmpl python sandbox_runner.py full-autonomous-run
```

## Visual Agent Prompt Format

`build_visual_agent_prompt` assembles a structured message for the
`VisualAgentClient`. When `VA_PROMPT_TEMPLATE` is not provided the sandbox emits
a descriptive block with labelled sections. Each prompt begins with an
"Introduction" that explains the helper to create, followed by subsections for
functions, dependencies and coding standards. Repository layout, metadata and
the snippet context are included so the agent has full background information.
Use `VA_PROMPT_PREFIX` to prepend extra text and `VA_PROMPT_TEMPLATE` to supply
a custom template.

Example of the default format:

```text
### Introduction
Add a Python helper to `helper.py` that print hello.

### Functions
- `auto_print_hello(*args, **kwargs)`

### Dependencies
standard library

### Coding standards
Follow PEP8 with 4-space indents and <79 character lines. Use Google style docstrings and inline comments for complex logic.

### Repository layout
helper.py

### Environment
3.11.12

### Metadata
description: print hello

### Version control
commit all changes to git using descriptive commit messages

### Testing
Run `scripts/setup_tests.sh` then execute `pytest --cov`. Report any failures.
The suite includes integration checks under `tests/test_sandbox_integration.py`
that exercise preset adaptation and synergy metric forecasts. These tests run in
CI along with the rest of the collection.

### Snippet context
def hello():
    pass
```

## Discrepancy Detection

After each iteration the sandbox calls `DiscrepancyDetectionBot.scan()` to
analyse models and workflows for irregularities. The number of detections is
added to the metrics dictionary as `discrepancy_count` before it is passed to
`ROITracker.update`.

## Sandbox Dashboard

After the sandbox runs, ROI trends are stored in `sandbox_data/roi_history.json`.
Launch a metrics dashboard to visualise these values and predicted metrics:

```bash
python -m menace.metrics_dashboard --file sandbox_data/roi_history.json --port 8002
```

When running the autonomous loop you can start the dashboard automatically with
the `--dashboard-port` option or by setting `AUTO_DASHBOARD_PORT`:

```bash
python sandbox_runner.py full-autonomous-run --dashboard-port 8002
```

Open the displayed address in your browser to see graphs. The `/roi` endpoint
returns ROI deltas and `/metrics/<name>` serves time series for metrics such as
`security_score` or `projected_lucrativity`. Each series includes `predicted`
and `actual` arrays so you can gauge forecast accuracy. To see the error as a
number use the CLI:

```bash
python -m menace.roi_tracker reliability sandbox_data/roi_history.json --metric security_score
```

## Ranking Preset Scenarios

Run multiple sandbox sessions with different environment presets and collect the
`roi_history.json` file from each run. Use the `rank-scenarios` subcommand to
compare their effectiveness:

```bash
python sandbox_runner.py rank-scenarios run1 run2/roi_history.json
```

The command prints scenario names sorted by cumulative ROI and includes the last
recorded `security_score`:

```
scenario_a ROI=3.25 security_score=0.82
scenario_b ROI=2.10 security_score=0.90
```

Use `rank-synergy` to compare the combined effect of modules across presets.
The command aggregates `synergy_roi` by default. Specify `--metric` to rank by
other synergy metrics:

```bash
python sandbox_runner.py rank-synergy run1 run2 --metric security_score
```

This prints scenario names with their cumulative synergy values.

Use `rank-scenario-synergy` to compare synergy metrics recorded per scenario.
The command aggregates `synergy_roi` by default and can target any other
`synergy_<metric>`:

```bash
python sandbox_runner.py rank-scenario-synergy run1 run2 --metric revenue
```

Scenario names are sorted by the total value of the chosen synergy metric.

## Fully Autonomous Runs

The `full-autonomous-run` subcommand wraps the logic from
`scripts/full_autonomous_run.py`. It repeatedly generates environment presets
and executes sandbox cycles until the :class:`ROITracker` reports diminishing
returns for every module.

```bash
python sandbox_runner.py full-autonomous-run --preset-count 3 --dashboard-port 8002
```

Use `--max-iterations` to limit the number of iterations when running
non-interactively. Final module rankings and metric values are printed once the
loop finishes. Pass `--dashboard-port PORT` or set `AUTO_DASHBOARD_PORT` to
monitor progress live via the metrics dashboard.

To replay a specific set of presets use the `run-complete` subcommand and pass
the preset JSON directly:

```bash
python sandbox_runner.py run-complete presets.json --max-iterations 1
```

This will invoke `full_autonomous_run` with the provided presets and also launch
`MetricsDashboard` when `--dashboard-port` or `AUTO_DASHBOARD_PORT` is
specified.

The `run_autonomous.py` helper exposes the same functionality while verifying
dependencies first. It keeps launching new runs until ROI improvements fade for
all modules and workflows. `--roi-cycles` and `--synergy-cycles` cap how many
consecutive below-threshold cycles trigger convergence. The optional `--runs`
argument acts as an upper bound:

Synergy convergence now checks the rolling correlation of metric values to
ensure improvements are not simply trending upward or downward. Confidence
levels are derived from a dynamic t-distribution, providing more reliable
bounds for small sample sizes.

Adaptive synergy thresholds rely on a weighted EMA of the difference between
predicted and actual synergy metrics. ``--synergy-threshold-window`` controls
how many recent values feed into the EMA while ``--synergy-threshold-weight``
adjusts how strongly newer samples influence the result.

``_adaptive_threshold`` and ``_adaptive_synergy_threshold`` return ``0`` when
insufficient history is available. Otherwise they scale the exponentially
weighted standard deviation of the last ``window`` values by ``factor`` to
derive a dynamic bound. ``_synergy_converged`` builds on these thresholds to
decide when synergy metrics have stabilised. When ``statsmodels`` and ``scipy``
are installed the Augmented Dickey–Fuller and Levene tests refine the
confidence score; otherwise simpler mean and variance comparisons are used.

```bash
python run_autonomous.py --runs 2 --preset-count 2 --dashboard-port 8002
```
Each iteration prints a `Starting autonomous run` message. The loop ends early
whenever ROI deltas remain below the tracker threshold for the configured
number of cycles. The dashboard remains available throughout to monitor
aggregated metrics.

## Monitoring Recovery Status

`SandboxRecoveryManager` tracks how often the sandbox has been restarted and the
timestamp of the last failure. These values are available via the
`metrics` property and can be exported with the built in metrics server:

```python
from menace.metrics_exporter import start_metrics_server
from sandbox_recovery_manager import SandboxRecoveryManager

start_metrics_server(8001)
recovery = SandboxRecoveryManager(main)
```

Prometheus will expose two gauges named `sandbox_restart_count` and
`sandbox_last_failure_time`. The latter holds a Unix timestamp or `0` when no
failure has occurred yet.
