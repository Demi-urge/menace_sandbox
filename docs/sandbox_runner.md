# Sandbox Runner

`sandbox_runner.py` executes self-improvement cycles without creating a fresh
repository clone. Temporary databases and event logs are created so production
data remains untouched. The runner requires the `foresight_tracker` package to
project ROI trends; the full implementation is bundled with the repository and
must be importable.

The sandbox uses the existing repository defined by `SANDBOX_REPO_PATH`. This is
expected to be a checkout of `https://github.com/Demi-urge/menace_sandbox` and
is modified and evaluated directly.

Paths in the examples below are resolved with
`dynamic_path_router.resolve_path`. The router honours `SANDBOX_REPO_PATH` when
set and falls back to Git metadata, allowing forked layouts or nested clones to
share the same tooling.

See [dynamic_path_router.md](dynamic_path_router.md) for caching behaviour and
migration guidance when adding new files.

> **Note:** The former `sandbox_runner.workflow_runner` module has been
> replaced by `sandbox_runner.workflow_sandbox_runner`. Import
> `WorkflowSandboxRunner` from the new module; the old path remains as a
> deprecated alias.

## Usage Notes

Run the runner against the current repository by setting `SANDBOX_REPO_PATH`
and specifying how many cycles to execute:

```bash
SANDBOX_REPO_PATH=$(pwd) python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)" --runs 1
```

Use `--preset-file` or the `SANDBOX_ENV_PRESETS` environment variable to supply
scenario presets. When omitted the runner generates a small set of defaults.

To evaluate a specific workflow across common sandbox scenarios use the
`--run-scenarios` option or call the helper directly from Python:

```bash
python -m sandbox_runner.cli --run-scenarios 42
```

```python
from sandbox_runner import run_scenarios

tracker, summary = run_scenarios(["simple_functions:print_ten"])
print(summary["worst_scenario"], summary["scenarios"]["normal"]["roi_delta"])
```

Each workflow step must specify its module path using ``module:function`` or
``module.function``. Bare function names are no longer supported and will
raise ``ValueError``.

Both forms run the workflow in five predefined scenarios. Each scenario is
executed twice – once with the workflow enabled and once with it disabled – so
the ROI delta reflects the workflow's direct contribution. The disabled run
serves as the counterfactual baseline. The scenarios are:

| Scenario | Description |
| --- | --- |
| `normal` | baseline environment |
| `concurrency_spike` | increased concurrency level |
| `hostile_input` | adversarial or malicious input |
| `schema_drift` | changes in dependent data schemas |
| `flaky_upstream` | unstable upstream service |

Example output highlighting the worst performer:

```text
normal: +0.500
hostile_input: -0.250 <== WORST
schema_drift: -0.100
flaky_upstream: +0.050
```

ROI delta is calculated as ``roi_with_workflow - roi_without_workflow``. The
second run therefore acts as a counterfactual that reveals the workflow's
impact. Positive values indicate an improvement over the baseline, while
negative values mean the workflow performs worse. The scenario with the most
negative delta is reported as the worst case.

### Temporal trajectory simulations

Use :func:`simulate_temporal_trajectory` to gauge how a workflow behaves as
conditions degrade. The helper runs the workflow through a deterministic set of
presets and records the stage name, ROI, resilience, degradation and a
``stability`` score for each step. These extra fields allow the sandbox to model
long‑term decay by tracking how stability drops as stages progress. It can be
invoked from the CLI or directly from Python:

```bash
python -m sandbox_runner.cli --simulate-temporal-trajectory 42
```

```python
from sandbox_runner.environment import simulate_temporal_trajectory
from foresight_tracker import ForesightTracker

ft = ForesightTracker()
simulate_temporal_trajectory(42, foresight_tracker=ft)
```

The stages execute in the following order:

| Scenario | Description |
| --- | --- |
| `normal` | baseline environment |
| `high_latency` | artificial network delay |
| `resource_strain` | throttled CPU and disk resources |
| `schema_drift` | mismatched or legacy schemas |
| `chaotic_failure` | broken authentication and corrupt payloads |

Metrics for each stage are returned alongside the updated tracker and, when a
``ForesightTracker`` is supplied, appended to ``foresight.history`` with the
associated ``stage`` and ``stability`` values for later analysis.

## Workflow impact analysis

The runner can project how a change to one workflow ripples through the rest of
the system by consulting :mod:`workflow_graph`.  ``WorkflowGraph`` seeds its DAG
from ``WorkflowDB`` and keeps it up to date by listening to workflow events on a
``UnifiedEventBus``.  When a workflow is selected for improvement the dependency
graph is loaded and :meth:`workflow_graph.WorkflowGraph.simulate_impact_wave`
estimates downstream ROI and synergy deltas.  Self‑improvement routines can
consume this projection to prioritise follow‑up cycles:

```python
from unified_event_bus import UnifiedEventBus
from workflow_graph import WorkflowGraph
from self_improvement.api import SelfImprovementEngine
from context_builder_util import create_context_builder

bus = UnifiedEventBus()
graph = WorkflowGraph()
graph.attach_event_bus(bus)
builder = create_context_builder()
engine = SelfImprovementEngine(context_builder=builder, event_bus=bus)

projection = graph.simulate_impact_wave("42", 1.0, 0.0)
# use `projection` to decide which dependant workflow to schedule next
```

## Monitoring sandbox performance

``sandbox_runner`` emits basic health gauges via :mod:`metrics_exporter`. Start
the exporter and point Prometheus at the port to scrape them:

```python
from menace.metrics_exporter import start_metrics_server

start_metrics_server(8001)
```

The runner updates ``sandbox_cpu_percent``, ``sandbox_memory_mb`` and
``sandbox_crashes_total`` for each cycle.  ``sandbox_dashboard.py`` renders a
small Chart.js page showing these values alongside ROI history:

```bash
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_dashboard.py'))
PY
)" --port 8002
```

Open ``http://localhost:8002`` to view CPU and memory usage and the cumulative
crash count.

## Algorithm Overview

Each cycle proceeds through a consistent set of stages:

1. load or generate environment presets;
2. execute the target modules inside an isolated copy of the repository or a
   container/VM for full-environment mode;
3. collect ROI and entropy deltas for every module;
4. persist metrics via `_SandboxMetaLogger`, which also detects entropy ceilings
   and diminishing returns;
5. rank modules and terminate once all sections are flagged.

This high level flow mirrors production behaviour while keeping runs
self-contained and reproducible.

The sandbox's ROI predictions are calibrated by a dedicated
[TruthAdapter](truth_adapter.md). It can be trained on live or shadow data and
flags low-confidence outputs when feature drift is detected.

## GPT Memory Persistence

`sandbox_runner` and associated bots can persist previous GPT interactions using
`gpt_memory.GPTMemoryManager`. To enable memory across sessions, create a
`GPTMemoryManager` with a file path and pass it to bots via the `knowledge`
parameter when asking the model. The default database file is `gpt_memory.db` in
the working directory. Supplying a different path allows memories to be shared
between runs, while using `:memory:` keeps the history ephemeral.

Key configuration options when instantiating the manager:

- `db_path` – SQLite file used for storage.
- `embedder` – optional `SentenceTransformer` model enabling semantic search.
- `event_bus` – optional `UnifiedEventBus` publishing a `"memory:new"` event for
  each logged interaction.

Interactions should include the canonical tag set from `log_tags.py` so other
services can reason about the stored context:

```python
from log_tags import INSIGHT

mgr.log_interaction("user question", "assistant reply", tags=[INSIGHT])
```

When an event bus is supplied the sandbox can stream new memories into the
knowledge graph:

```python
from unified_event_bus import UnifiedEventBus
from knowledge_graph import KnowledgeGraph

bus = UnifiedEventBus()
mgr = GPTMemoryManager("sandbox_memory.db", event_bus=bus)
kg = KnowledgeGraph("kg.gpickle")
kg.listen_to_memory(bus, mgr)
```

Configure retention by setting `GPT_MEMORY_RETENTION` and optionally
`GPT_MEMORY_COMPACT_INTERVAL` to control how often `compact()` runs:

```bash
export GPT_MEMORY_RETENTION="insight=40,error_fix=20"
export GPT_MEMORY_COMPACT_INTERVAL=1800  # seconds
```

Example usage enabling persistence across sessions:

```python
from gpt_memory import GPTMemoryManager

# first run
mgr = GPTMemoryManager("sandbox_memory.db")
mgr.log_interaction("user question", "assistant reply", tags=["note"])
mgr.close()

# subsequent run
mgr = GPTMemoryManager("sandbox_memory.db")
context = mgr.search_context("user question")
```

Equivalent shell commands showing persistence:

```bash
# first run
python - <<'PY'
from gpt_memory import GPTMemoryManager
from log_tags import FEEDBACK
mgr = GPTMemoryManager("sandbox_memory.db")
mgr.log_interaction("Is it working?", "Yes", tags=[FEEDBACK])
mgr.close()
PY

# second run
python - <<'PY'
from gpt_memory import GPTMemoryManager
mgr = GPTMemoryManager("sandbox_memory.db")
print([e.response for e in mgr.search_context("working")])
PY
```

Using `db_path=":memory:"` instead of a file path disables persistence and
keeps interactions only for the current process.

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

Both Docker and QEMU startups are retried with exponential backoff. When all
attempts fail the error message is stored in the returned tracker under
`diagnostics['docker_error']` or `diagnostics['vm_error']`.

Example configuration:

```python
VM_SETTINGS = {
    "windows_image": "/var/lib/vms/windows.qcow2",
    "memory": "2G",
    "timeout": 600,
}
```

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

Set `SANDBOX_PRESET_MODE=canonical` to use a deterministic set of presets
returned by `generate_canonical_presets()`. These cover common scenarios such
as `high_latency_api`, `hostile_input`, `user_misuse` and `concurrency_spike`
so every module is exercised under each condition.
Set `SANDBOX_FAIL_ON_MISSING_SCENARIOS=1` to raise an error when a module skips
any canonical scenario during coverage verification.

## Predefined Profiles

When `SANDBOX_PRESET_MODE=canonical` is enabled or `generate_presets()` is
called with explicit profile names, the runner iterates over several
deterministic scenarios:

- `high_latency_api` – introduces significant network delay.
- `hostile_input` – replaces normal stubs with adversarial payloads.
- `user_misuse` – attempts incorrect API usage and unauthorized access.
- `concurrency_spike` – spawns bursts of threads and async tasks.

Each profile provides a `low` and `high` severity preset. The mapping returned
by `generate_canonical_presets()` can be passed directly to
`run_repo_section_simulations`, which will exercise every module under both
levels for the same scenario:

```python
from environment_generator import generate_canonical_presets
from vector_service.context_builder import ContextBuilder
from sandbox_runner import run_repo_section_simulations

presets = generate_canonical_presets()
builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
tracker = run_repo_section_simulations(
    "/repo", env_presets=presets, context_builder=builder
)
```

Run all canonical profiles with:

```bash
SANDBOX_PRESET_MODE=canonical python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)" --runs 1
```

To target specific profiles, generate presets via the CLI and pass them to the
runner:

```bash
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('environment_cli.py'))
PY
)" --profiles hostile_input concurrency_spike --count 1 > presets.json
SANDBOX_ENV_PRESETS="$(cat presets.json)" python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)"
```

## Automatic Failure Mode Coverage

`run_repo_section_simulations` guarantees that every module experiences the
four canonical stress profiles – `high_latency_api`, `hostile_input`,
`user_misuse` and `concurrency_spike`. If a provided preset set omits any of
them, the runner injects additional runs so these failure modes are still
exercised. It also performs keyword matching on module paths to generate
module‑specific presets, ensuring that components such as databases or parsers
receive relevant scenarios.

### Custom keyword profile mappings

`environment_generator` suggests profiles for modules by scanning their paths
for known domain keywords. The defaults cover terms like `api`, `parser` and
`concurrency`. Create a `sandbox_settings.yaml` file with a `keyword_profiles`
section to override or extend these mappings without touching the code:

```yaml
keyword_profiles:
  database: [high_latency_api, concurrency_spike]
  cache: [high_latency_api]
  auth: [user_misuse, hostile_input]
```

Set the `SANDBOX_SETTINGS_YAML` environment variable to point to an alternate
configuration file if needed. When present the mappings are merged with the
defaults before profile suggestions are made.

## Hostile Input Stub Strategy

The hostile profile or the `hostile_input` failure mode forces the sandbox to
use malicious input stubs by setting `SANDBOX_STUB_STRATEGY=hostile` (alias
`misuse`). Enable it manually to fuzz a run:

```bash
export SANDBOX_STUB_STRATEGY=hostile
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)" --runs 1
```

## Concurrency Settings

Profiles or presets containing `concurrency_spike` stress thread and task
handling. The preset can supply `THREAD_BURST` and `ASYNC_TASK_BURST` values to
control the spike. They may also be set directly:

```bash
export FAILURE_MODES=concurrency_spike
export THREAD_BURST=32
export ASYNC_TASK_BURST=128
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)" --runs 1
```

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
- `SANDBOX_PATCH_RETRY_DELAY` – delay in seconds between patch attempts
  (default `0.1`).
- `SANDBOX_REPO_PATH` – local path of the sandbox repository clone.
- `SANDBOX_DATA_DIR` – directory used for ROI history and patch records.
- `SANDBOX_AUTO_MAP` – builds or refreshes `module_map.json` on
  startup. See [dynamic_module_mapping.md](dynamic_module_mapping.md) for the
  `--algorithm`, `--threshold` and `--semantic` options. The legacy
  `SANDBOX_AUTODISCOVER_MODULES` variable is still supported with a warning.
- `SANDBOX_REFRESH_MODULE_MAP` – rebuild the module map before running the sandbox cycles.
- `--refresh-module-map` performs the same refresh at startup and also enables automatic
  updates when new files are patched.
- `SANDBOX_POOL_LABEL` – Docker label used for pooled containers (default
  `menace_sandbox_pool`).
- `SANDBOX_POOL_LOCK` – optional lock file guarding container pool operations.
- `SANDBOX_OVERLAY_MAX_AGE` – duration after which stray VM overlays are purged
  (default `7d`).
- `SANDBOX_ACTIVE_CONTAINERS` – file that records running container IDs so stray instances can be purged on startup.
- `SANDBOX_ACTIVE_OVERLAYS` – JSON file tracking overlay directories currently in use.
- `SANDBOX_FAILED_OVERLAYS` – stores overlays that failed to delete for later cleanup.
- `SANDBOX_POOL_CLEANUP_INTERVAL` – seconds between cleanup sweeps (default `60`).
- `SANDBOX_WORKER_CHECK_INTERVAL` – frequency for verifying cleanup workers (default `30`).
- `SANDBOX_CONTAINER_IDLE_TIMEOUT` – idle duration before pooled containers are removed (default `300`).
- `SANDBOX_CONTAINER_MAX_LIFETIME` – maximum age of a pooled container in seconds (default `3600`).
  Containers, Docker volumes and networks created by the sandbox runner that exceed this age are removed on
  startup even when the pool label is missing.
- `SANDBOX_CONTAINER_DISK_LIMIT` – size limit for container directories; `0` disables the check.
- `SANDBOX_CONTAINER_USER` – user specification passed to Docker containers when set.
- `SANDBOX_CLEANUP_ALERT_THRESHOLD` – consecutive failed cleanup retries before an error is logged (default `3`).
- `SANDBOX_MAX_CONTAINER_COUNT` – maximum number of active containers across all pools (default `10`).
- `SANDBOX_MAX_OVERLAY_COUNT` – maximum number of active VM overlays before oldest entries are purged (default `0` for unlimited).

When any network variables are set and the `tc` binary is available the sandbox
temporarily applies a `netem` queueing discipline to the loopback interface
before executing the snippet. The settings are removed afterwards so subsequent
cycles are unaffected.

## Runtime Simulation

`simulate_execution_environment` performs a static safety check on snippets. Set
`SANDBOX_DOCKER=1` or pass `container=True` to execute the snippet in a minimal
Docker/Podman container with the specified resource limits (`CPU_LIMIT`,
`MEMORY_LIMIT` and `DISK_LIMIT`). CPU usage, memory consumption and disk I/O are
recorded and returned in a `runtime_metrics` dictionary. When container startup
fails the error is logged and the dictionary includes a `container_error`
entry describing the problem.

## Section Targeting

`run_repo_section_simulations(repo_path, input_stubs=None, env_presets=None, modules=None)` analyses each Python file with `_extract_sections` and simulates execution for every section. A `ROITracker` records risk flags per module and section so trends can be ranked. Pass custom input stubs to exercise different code paths and provide multiple environment presets to compare behaviour across configurations. Provide a list of relative `modules` to restrict analysis to specific paths. Set `return_details=True` to receive raw results grouped by preset.

When no `input_stubs` argument is supplied the function calls `generate_input_stubs()` from `environment.py`. This helper first inspects the target function signature (when available) and derives argument dictionaries from defaults and type hints. The `SANDBOX_INPUT_STUBS` variable overrides this behaviour. When unset, history or template files are consulted before falling back to the signature, a smart faker-based strategy, a synthetic language-model strategy, a hostile fuzzing strategy, a misuse strategy or a random strategy. Plugins discovered via `SANDBOX_STUB_PLUGINS` may augment or override the generated stubs.

The `hostile` strategy generates adversarial examples such as SQL injection strings, XSS payloads, oversized values and malformed JSON. The `misuse` strategy omits required fields or provides values of incorrect types to mimic user errors. Presets with the `hostile_input` failure mode automatically set `SANDBOX_STUB_STRATEGY=hostile` and `_inject_failure_modes` replaces any lower-case stub variables with these malicious payloads before execution.
Presets that include the `user_misuse` failure mode set `SANDBOX_STUB_STRATEGY=misuse` and attempt to call functions with incorrect argument counts and touch disallowed files. These errors are printed to `stderr` but execution proceeds so the sandbox can observe misuse safely.
### Stub generation settings

The generative stub provider exposes several knobs via environment variables:

- `SANDBOX_STUB_TIMEOUT` – maximum time in seconds to wait for stub generation.
- `SANDBOX_STUB_RETRIES` – number of retry attempts when generation fails.
- `SANDBOX_STUB_RETRY_BASE` – initial delay for exponential back-off in seconds.
- `SANDBOX_STUB_RETRY_MAX` – ceiling for the back-off delay.
- `SANDBOX_STUB_CACHE_MAX` – maximum number of cached stub responses.
- `SANDBOX_STUB_MODEL` – model name used for stub generation. When unset a
  deterministic rule-based strategy is used.
- `SANDBOX_STUB_FALLBACK_MODEL` – model name used when the preferred provider is unavailable.

The `concurrency_spike` failure mode starts bursts of threads and async tasks. The sandbox records how many threads and tasks were spawned in the metrics.

Sections with declining ROI trigger dedicated improvement cycles. Only the flagged section is iteratively modified while metrics are tracked. When progress stalls the sandbox issues a brainstorming request to local models if `SANDBOX_BRAINSTORM_INTERVAL` is set. Consecutive low‑ROI cycles before brainstorming can be tuned via `SANDBOX_BRAINSTORM_RETRIES`.

`_SandboxMetaLogger.diminishing()` evaluates these ROI deltas using a rolling mean and standard deviation over the last `consecutive` cycles. A module is flagged when the mean is within the given threshold and the standard deviation falls below a small epsilon, preventing sporadic fluctuations from triggering improvements.

### Orphan discovery and integration

Isolated modules that lack inbound references may still contain useful logic.
`discover_recursive_orphans` and `scripts.discover_isolated_modules` surface
these files before each run. Use `--discover-isolated` or `--no-discover-isolated` to control the isolated scan. `discover_recursive_orphans` returns a mapping from
each orphan module to the module(s) that imported it, allowing downstream tools
to construct workflow segments per dependency chain. The collected paths are
then passed to
`run_repo_section_simulations(..., modules=modules)` so only the candidates are
executed. When `SANDBOX_AUTO_INCLUDE_ISOLATED=1` or `--auto-include-isolated`
is passed, the sandbox adds modules that pass to
`sandbox_data/module_map.json` and `environment.generate_workflows_for_modules`
creates one‑step workflows for them. Enable recursive dependency scanning with
`--recursive-include`/`SANDBOX_RECURSIVE_ORPHANS=1` or
`--recursive-isolated`/`SANDBOX_RECURSIVE_ISOLATED=1`. Disable the scan with
`SANDBOX_DISABLE_ORPHAN_SCAN=1` or `SELF_TEST_DISABLE_ORPHANS=1`. Set
`SANDBOX_DISCOVER_ISOLATED=0` or pass `--no-discover-isolated` to skip isolated
module discovery.

## WorkflowSandboxRunner

`WorkflowSandboxRunner` runs one or more callables inside an isolated temporary
directory. File mutations such as `open` or `shutil.copy` are redirected into
this sandbox so the host filesystem remains untouched. When `safe_mode=True`
the runner monkeypatches `requests`, `httpx`, `urllib` and raw sockets so that
any outbound network attempt raises `RuntimeError` unless a matching stub is
provided. Socket creation helpers such as ``socket.socket``, ``create_connection``,
``create_server`` and ``fromfd`` are blocked while Unix domain sockets remain
available. Supplying ``network_mocks["socket"]`` returns a custom socket object
instead of raising. Per‑module telemetry reporting execution time, CPU time,
peak memory, crash frequency and return values is available through
``runner.telemetry``.

### Example with injected test data

```python
from sandbox_runner import WorkflowSandboxRunner

def read_file():
    with open("payload.txt") as fh:
        return fh.read()

runner = WorkflowSandboxRunner()
metrics = runner.run(read_file, safe_mode=True,
                     test_data={"payload.txt": "fake"})
print(metrics.modules[0].success)
print(runner.telemetry["time_per_module"]["read_file"])
print(runner.telemetry["cpu_per_module"]["read_file"])
print(runner.telemetry["memory_per_module"]["read_file"])
print(runner.telemetry["peak_memory_per_module"]["read_file"])
```

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

Example socket mock:

```python
from sandbox_runner import WorkflowSandboxRunner

class DummySocket:
    def __init__(self, *a, **kw):
        pass
    def connect(self, *a, **kw):
        return b"ok"
    def close(self):
        pass

def ping():
    import socket
    s = socket.socket()
    s.connect(("example.com", 80))
    s.close()

runner = WorkflowSandboxRunner()
runner.run(ping, safe_mode=True, network_mocks={"socket": DummySocket})
```

Example filesystem mock:

```python
from sandbox_runner import WorkflowSandboxRunner
from pathlib import Path

def read_conf():
    return Path("config.json").read_text()

runner = WorkflowSandboxRunner()
metrics = runner.run(
    read_conf,
    fs_mocks={"pathlib.Path.read_text": lambda self, *a, **kw: "{}"},
)
print(metrics.modules[0].result)
```

Per-module fixtures can be supplied via the ``module_fixtures`` argument. The
mapping uses each module's ``__name__`` as the key and may define ``files`` and
``env`` sub-mappings. ``files`` pre-populate paths inside the sandbox before the
module executes while ``env`` temporarily sets environment variables just for
that step. Because fixtures are ordinary dictionaries they can be reused across
multiple runs.

```python
fixtures = {
    "writer": {"env": {"TOKEN": "abc"}},
    "reader": {"files": {"out.txt": "seed"}, "env": {"TOKEN": "def"}},
}
runner.run([writer, reader], module_fixtures=fixtures)
```

Custom stub providers may be supplied via the ``stub_providers`` argument to
inject domain‑specific payloads. The runner's automatic input generation and
network/file mocking allow workflows to run without touching the host
environment while ``runner.telemetry`` exposes the collected per‑module metrics.

### Resource limits and telemetry

Each module may be constrained with a ``timeout`` (seconds) and ``memory_limit``
(bytes) when calling :meth:`WorkflowSandboxRunner.run`. Exceeding either limit
aborts the module and records a crash. CPU time and RSS deltas for every
module are available in ``runner.telemetry`` under ``cpu_per_module`` and
``memory_per_module`` while overall peak RSS metrics appear under
``peak_memory`` and ``peak_memory_per_module`` so downstream tools such as the
self-improvement engine can consume them.

```python
runner.run(task, timeout=2.0, memory_limit=50_000_000)
```

## Workflow Simulations

`run_workflow_simulations(workflows_db, env_presets=None, context_builder=builder)` replays stored
workflow sequences under each environment preset. ROI deltas and metrics are
aggregated per workflow ID. After iterating over individual workflows a single
combined snippet containing all steps is executed. Metrics from this run are
recorded under the module name `all_workflows`, allowing the tracker to show the
overall behaviour across every workflow.
When the workflow database is empty or `--dynamic-workflows` is enabled the
groups returned by `discover_module_groups()` are converted into temporary
workflows and executed instead.

## Synergy Metrics

After each section has been simulated individually the sandbox executes a
combined phase containing every previously flagged section. The ROI and metrics
from this run are compared against the average of the individual section runs.
The differences are stored under `synergy_roi` and `synergy_<metric>` entries in
`roi_history.json`. The delta is also attributed to each involved module so
`ROITracker.rankings()` reflects the overall cross‑module impact.

`run_workflow_simulations(..., context_builder=builder)` performs the same comparison for entire workflows.
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

## SelfCodingEngine Integration

The sandbox requests improvements from local models after ROI gains diminish. Suggestions are applied via `SelfCodingEngine` and reverted if they fail to increase ROI. Set `SANDBOX_BRAINSTORM_INTERVAL` to a positive integer to periodically ask the local models for high‑level ideas during the run. Use `SANDBOX_BRAINSTORM_RETRIES` to specify how many consecutive low‑ROI cycles trigger extra brainstorming. No external API keys are required.
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

When the test subprocess exits with a non-zero status `_coverage_percent`
raises `CoverageSubprocessError`. `_run_tests` catches this, writes the
combined output to `SANDBOX_DATA_DIR` and re-raises a `RuntimeError` so the
patch attempt is marked as failed. Score history entries are skipped for
these attempts.

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
SANDBOX_METRICS_PLUGIN_DIR=plugins python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)"
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

## Discrepancy Detection

After each iteration the sandbox calls `DiscrepancyDetectionBot.scan()` to
analyse models and workflows for irregularities. The number of detections is
added to the metrics dictionary as `discrepancy_count` before it is passed to
`ROITracker.update`.

## Sandbox Dashboard

After the sandbox runs, ROI trends are stored in `sandbox_data/roi_history.json`.
Launch a metrics dashboard to visualise these values and predicted metrics:

```bash
python -m menace.metrics_dashboard --file "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_data/roi_history.json'))
PY
)" --port 8002
```

When running the autonomous loop you can start the dashboard automatically with
the `--dashboard-port` option or by setting `AUTO_DASHBOARD_PORT`:

```bash
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)" full-autonomous-run --dashboard-port 8002
```

Open the displayed address in your browser to see graphs. The `/roi` endpoint
returns ROI deltas and `/metrics/<name>` serves time series for metrics such as
`security_score` or `projected_lucrativity`. Each series includes `predicted`
and `actual` arrays so you can gauge forecast accuracy. To see the error as a
number use the CLI:

```bash
python -m menace.roi_tracker reliability "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_data/roi_history.json'))
PY
)" --metric security_score
```

## Ranking Preset Scenarios

Run multiple sandbox sessions with different environment presets and collect the
`roi_history.json` file from each run. Use the `rank-scenarios` subcommand to
compare their effectiveness:

```bash
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)" rank-scenarios run1 run2/roi_history.json
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
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)" rank-synergy run1 run2 --metric security_score
```

This prints scenario names with their cumulative synergy values.

Use `rank-scenario-synergy` to compare synergy metrics recorded per scenario.
The command aggregates `synergy_roi` by default and can target any other
`synergy_<metric>`:

```bash
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)" rank-scenario-synergy run1 run2 --metric revenue
```

Scenario names are sorted by the total value of the chosen synergy metric.

### Synergy Metric Summaries

The `synergy-metrics` subcommand displays the most recent synergy values along
with their exponential moving averages. Pass `--file` to specify the
`roi_history.json` path and use `--window` to change the EMA length. Adding
`--plot` creates a small bar chart when `matplotlib` is available.

```bash
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)" synergy-metrics --file "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_data/roi_history.json'))
PY
)" --window 4
```

## Fully Autonomous Runs

The `full-autonomous-run` subcommand wraps the logic from
`scripts/full_autonomous_run.py`. It repeatedly generates environment presets
and executes sandbox cycles until the :class:`ROITracker` reports diminishing
returns for every module.

```bash
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)" full-autonomous-run --preset-count 3 --dashboard-port 8002
```

Use `--max-iterations` to limit the number of iterations when running
non-interactively. Final module rankings and metric values are printed once the
loop finishes. Pass `--dashboard-port PORT` or set `AUTO_DASHBOARD_PORT` to
monitor progress live via the metrics dashboard.

To replay a specific set of presets use the `run-complete` subcommand and pass
the preset JSON directly:

```bash
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)" run-complete presets.json --max-iterations 1
```

This will invoke `full_autonomous_run` with the provided presets and also launch
`MetricsDashboard` when `--dashboard-port` or `AUTO_DASHBOARD_PORT` is
specified.

The `run_autonomous.py` helper exposes the same functionality while verifying
dependencies first. It keeps launching new runs until ROI improvements fade for
all modules and workflows. `--roi-cycles` and `--synergy-cycles` cap how many
consecutive below-threshold cycles trigger convergence. The optional
`--entropy-threshold` flag (or `ENTROPY_THRESHOLD`) adjusts the minimum ROI
gain per entropy delta before entropy increases are ignored. The optional `--runs`
argument acts as an upper bound. Use `--log-level LEVEL` or set
`SANDBOX_LOG_LEVEL` to adjust console verbosity. Pass `--verbose` to force
debug output regardless of the configured level:

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

Additional environment variables offer fine grained control over these
checks:

- ``ROI_CYCLES`` and ``SYNERGY_CYCLES`` override ``--roi-cycles`` and
  ``--synergy-cycles``.
- ``ENTROPY_THRESHOLD`` sets the ROI gain per entropy delta ceiling used by
  ``ROITracker``.
- ``SYNERGY_MA_WINDOW`` sets the rolling window for the EMA and stationarity
  test.
- ``SYNERGY_STATIONARITY_CONFIDENCE`` and ``SYNERGY_VARIANCE_CONFIDENCE``
  adjust the significance thresholds for the Augmented Dickey–Fuller and
  Levene tests.

When ``--synergy-threshold`` is omitted the runner reads ``synergy_history.db``
from the data directory and derives the threshold using
``_adaptive_synergy_threshold``. Likewise ``--synergy-cycles`` defaults to the
length of this history with a minimum of ``3``.

Synergy weights used during self-improvement are updated based on these
metrics. See [self_improvement.md#synergy-weight-learners](self_improvement.md#synergy-weight-learners)
for details on how the history influences weight adjustments.

Writes to this history file are protected with a ``filelock.FileLock``. The lock
``synergy_history.db.lock`` is acquired before saving and released once the
update completes to avoid corruption when multiple runs execute concurrently.

## Entropy delta detection

The sandbox tracks how much entropy each patch adds relative to ROI. When the
mean ROI gain per unit entropy delta falls below a ceiling the affected modules
are marked complete and skipped in later cycles. This prevents endless
microscopic tweaks that add complexity without improving results.

Configuration is controlled via:

- ``--entropy-threshold`` or ``ENTROPY_THRESHOLD`` – minimum ROI gain per
  entropy delta before further increases are ignored.
- ``--consecutive``/``--entropy-plateau-consecutive`` or
  ``ENTROPY_PLATEAU_CONSECUTIVE`` – entropy samples that must remain below the
  plateau threshold before a module converges.
- ``ENTROPY_PLATEAU_THRESHOLD`` – entropy ratios below this threshold for the
  configured number of consecutive samples trigger convergence.

During runs the meta logger emits debug lines such as
``modules hitting entropy ceiling: ['m.py']``. Seeing a module in this list
means its recent ROI-to-entropy ratios were too low and it will no longer be
tweaked. Flagged modules are persisted to ``*.flags`` files so later cycles
skip them.

Example CLI usage:

```bash
python -m sandbox_runner.cli --entropy-threshold 0.02 --consecutive 5
```

Example usage:

```bash
SYNERGY_MA_WINDOW=4 SYNERGY_STATIONARITY_CONFIDENCE=0.99 \
SYNERGY_VARIANCE_CONFIDENCE=0.9 SYNERGY_CYCLES=5 \
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('run_autonomous.py'))
PY
)" --runs 1
```

```bash
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('run_autonomous.py'))
PY
)" --runs 2 --preset-count 2 --dashboard-port 8002
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
from menace.sandbox_recovery_manager import SandboxRecoveryManager

start_metrics_server(8001)
recovery = SandboxRecoveryManager(main)
```

Prometheus will expose two gauges named `sandbox_restart_total` and
`sandbox_last_failure_ts`. The latter holds a Unix timestamp or `0` when no
failure has occurred yet.

When the `prometheus_client` package isn't available the recovery manager falls
back to writing these metrics to `sandbox_data/recovery.json`. Use the included
CLI to inspect the file:

```bash
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_recovery_manager.py'))
PY
)" --file "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_data/recovery.json'))
PY
)"
```

## Container Pool Failures

Each time a pooled container fails to start the counters `failures` and
`consecutive` are written to `sandbox_data/pool_failures.json`. A warning is
logged when the consecutive count reaches `SANDBOX_POOL_FAIL_THRESHOLD`
(default `5`). Review this file and the warnings to identify images that are
consistently failing to launch. A call to
`alert_dispatcher.dispatch_alert` is now issued **for every failed attempt**,
providing the image name and failure count. Each alert increments the
`container_creation_alerts_total` Prometheus gauge. The duration of each
attempt is exposed via the `container_creation_seconds` gauge labelled by
image.

## Automatic Cleanup

When `sandbox_runner.environment` is imported it immediately calls
`purge_leftovers()` to remove containers and QEMU overlay files that may have
been left behind by a previous crash.  Containers, Docker volumes and networks older than
`SANDBOX_CONTAINER_MAX_LIFETIME` whose command includes `sandbox_runner.py` are
also purged even if the pool label is missing.  Once this cross-run sweep has finished
you can avoid loading the environment altogether by setting `MENACE_LIGHT_IMPORTS=1`
before importing `sandbox_runner`.  This skips the cleanup step and defers all
Docker/QEMU initialisation until sandbox features are explicitly used, allowing
utility helpers such as ``discover_orphan_modules`` to run without Docker
installed.
two background workers are launched.  The regular cleanup worker removes idle or
unhealthy containers and deletes stale VM overlays while the reaper worker
collects orphaned containers that were not tracked correctly.  Both workers log
their actions and the counts accumulate in a set of metrics exposed through
`collect_metrics()`:

- `cleanup_idle` – containers removed after exceeding
  `SANDBOX_CONTAINER_IDLE_TIMEOUT`.
- `cleanup_unhealthy` – pooled containers that failed a health check.
- `cleanup_lifetime` – containers purged after
  `SANDBOX_CONTAINER_MAX_LIFETIME` seconds.
- `cleanup_disk` – containers removed because their temporary directory grew
  beyond `SANDBOX_CONTAINER_DISK_LIMIT`.
- `cleanup_volume` – Docker volumes deleted during startup or cleanup sweeps.
- `cleanup_network` – Docker networks removed when they become stale.
- `stale_containers_removed` – stale containers deleted during startup or by the
  background cleanup worker.
- `stale_vms_removed` – orphaned VM overlay directories removed.
- `runtime_vms_removed` – VM overlays deleted while the sandbox is running.
- `cleanup_failures` – failed attempts to stop or remove a container.
- `force_kills` – containers killed via the fallback CLI removal path.
- `active_container_limit_reached` – container creations skipped when the active limit was hit.
- `cleanup_duration_seconds_cleanup` and `cleanup_duration_seconds_reaper` –
  time spent in each cleanup sweep. Monitor these durations with the
  `cleanup_duration_seconds` Prometheus gauge. Unusually long sweeps may
  signal blocked disk operations or Docker issues.

Administrators can tune `SANDBOX_OVERLAY_MAX_AGE` to control when leftover overlay directories are deleted. Lower values free disk space sooner at the expense of longer boot times for VMs. Combine this with `SANDBOX_POOL_CLEANUP_INTERVAL` to run cleanup sweeps more often if overlays accumulate. Reducing `SANDBOX_CONTAINER_IDLE_TIMEOUT` or `SANDBOX_CONTAINER_MAX_LIFETIME` further shortens the lifespan of containers and their overlays.

Additional metrics include `container_failures_<image>` and
`consecutive_failures_<image>` for each image the pool attempted to create.
The Prometheus gauges `container_creation_failures_total`,
`container_creation_success_total` and `container_creation_alerts_total`
expose these counts using the `image` label. `container_creation_seconds`
records the duration of the last creation attempt for each image.
`container_backoff_base` reports the current exponential backoff base used when
creation repeatedly fails.

`schedule_cleanup_check()` periodically calls `ensure_cleanup_worker` to verify
that the cleanup and reaper tasks are still running.  The interval is
controlled by `SANDBOX_WORKER_CHECK_INTERVAL` (default `30` seconds).  If either
task stopped due to an error or manual cancellation this check automatically
restarts it so stale resources continue to be collected.

If you embed the sandbox into another application call
`register_signal_handlers()` after importing the environment.  The installed
handlers shut down the cleanup workers and remove pooled containers on
`SIGINT` or `SIGTERM` so interrupted runs do not leak resources.

If the process crashes you can safely clean up any leftover containers or QEMU
overlay files by running:

```bash
python -m sandbox_runner.cli --cleanup
```

To verify the cleanup run and see how many resources were removed use:

```bash
python -m sandbox_runner.cli check-resources
```

For additional safety you can schedule periodic cleanup with the
`--purge-stale` command. Run it hourly via cron:

```cron
0 * * * * /usr/bin/python -m sandbox_runner.cli --purge-stale
```

Or let the CLI set up an hourly purge automatically:

```bash
python -m sandbox_runner.cli install-autopurge
```

On Linux or macOS with systemd this copies the unit files and enables the timer
for the current user (run the command as root to install system-wide). On
Windows the same command imports the task from
`systemd/windows_sandbox_purge.xml`. If automatic installation fails copy
`systemd/sandbox_autopurge.*` manually and enable the timer with:

```bash
systemctl --user enable --now sandbox_autopurge.timer
```

Omit `--user` for a system-wide installation.

These units assume the repository lives at `%h/menace_sandbox`. Adjust
`WorkingDirectory` in `sandbox_autopurge.service` if you cloned the repository
elsewhere. Once enabled the timer will invoke
`python -m sandbox_runner.cli --purge-stale` every hour.

Enabling this timer purges leftover containers or VM overlay directories even
when the main sandbox is never started.

If you do not run the sandbox on a schedule, enable this timer (or the cron
entry above) so stale containers are removed automatically. The CLI command
`install-autopurge` installs and starts the timer for you. Verify it is active
with:

```bash
systemctl status --user sandbox_autopurge.timer
```

Omit `--user` when installing system-wide.


Keep an eye on the logs for messages from the cleanup workers.  A steady stream
of cleanup events may indicate resource leaks or overly strict timeouts.  When
either worker exits it logs `cleanup worker cancelled` or `reaper worker
cancelled`; if this appears unexpectedly check for unhandled exceptions.

### Known Limitations

 - When `psutil` is unavailable the cleanup falls back to using `pgrep` and
   `kill` to terminate stray QEMU processes. Overlay directories are still
   removed.
- Docker must be available for container cleanup to succeed and containers
  started outside the sandbox will not be touched.
 - On platforms that lock open files (e.g. Windows) overlay directories are
   retried for deletion in a helper subprocess using exponential backoff. Paths
   that still fail to delete are recorded for the next purge.

## Troubleshooting

- **Missing dependencies** – rerun `./setup_env.sh` to install required packages and verify that `ffmpeg` and `tesseract` are available.
