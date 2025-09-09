# Self Test Service

`SelfTestService` periodically runs the project's unit tests to catch regressions.
Specify custom pytest arguments via `SELF_TEST_ARGS` or the constructor. By
default tests run directly on the host, but enabling `use_container=True`
executes each provided test path inside a Docker container when available. Each
container is removed on completion so test runs are isolated.

When multiple test paths are supplied together with the `--workers` option,
the total worker count is distributed across the spawned containers.  This
keeps the overall level of parallelism consistent while still allowing each
container to run tests in parallel via `pytest -n`.

Coverage percentage and total runtime are recorded in `MetricsDB` when a
`DataBot` is supplied. Results can also be streamed incrementally by providing
`result_callback` which receives a dictionary after each file and a final
summary.

After each run the gauges `self_test_passed_total`, `self_test_failed_total`,
`self_test_average_runtime_seconds` and `self_test_average_coverage` expose the
number of passed and failed tests plus the average runtime and coverage for
Prometheus scraping. When containers are used two additional gauges record
container issues:

- `self_test_container_failures_total` counts cleanup or listing failures
  (for example when a container cannot be removed).
- `self_test_container_timeouts_total` increments whenever a test container
    exceeds the configured timeout and is terminated.

Orphan module processing is also exported via Prometheus gauges:

- `orphan_modules_tested_total` – total orphan modules executed during self tests.
- `orphan_modules_reintroduced_total` – modules that passed and were reintegrated.
- `orphan_modules_failed_total` – orphan modules whose tests failed.
- `orphan_modules_redundant_total` – modules skipped due to redundancy.

## Dependency resolution

`SelfTestService` relies on `sandbox_runner.dependency_utils.collect_local_dependencies`
to walk import graphs.  When `sandbox_runner` is not installed, an internal
resolver is used.  This fallback understands package‑relative imports,
namespace packages (PEP 420) and `from ... import *` patterns so self tests can
still determine local dependencies.  Optional modules are skipped when missing
and a best‑effort dependency set is produced.

## Recursive orphan discovery

`SelfTestService` cooperates with `sandbox_runner` to locate orphan modules and
their helper files. The process loads names from
`sandbox_data/orphan_modules.json`, walks each module's imports recursively and
executes the resulting set inside a temporary sandbox so side effects are
contained.

### Environment variables and CLI flags

- `SELF_TEST_RECURSIVE_ORPHANS` / `SANDBOX_RECURSIVE_ORPHANS` or the
  `--recursive-include` flag – follow orphan dependencies.
- `SELF_TEST_RECURSIVE_ISOLATED` / `SANDBOX_RECURSIVE_ISOLATED` or the
  `--recursive-isolated` flag – traverse helpers of isolated modules.
- `SELF_TEST_AUTO_INCLUDE_ISOLATED` / `SANDBOX_AUTO_INCLUDE_ISOLATED` or the
  `--auto-include-isolated` flag – queue modules from `discover_isolated_modules`.
- `SANDBOX_CLEAN_ORPHANS` or `--clean-orphans` – drop integrated entries from
  `orphan_modules.json` after a successful run.

### Classification and metrics storage

`discover_recursive_orphans` writes classifications to
`sandbox_data/orphan_classifications.json` alongside the main cache
`sandbox_data/orphan_modules.json`. Modules that pass are appended to
`sandbox_data/module_map.json` and metrics from each run are stored in
`sandbox_data/metrics.db`, powering Prometheus gauges like
`orphan_modules_tested_total`.

## Return values and integration callbacks

Calling `run_once` returns a tuple `(results, passed_modules)`. `results` is a
dictionary summarising the test run and now includes an `integration` field with
`integrated` and `redundant` lists. The second element is the list of modules
that passed in isolation. When an `integration_callback` is supplied it receives
`passed_modules` and may return a mapping with the same keys. These values are
merged into `results["integration"]`, allowing custom callbacks to report which
modules were incorporated or skipped.

```python
from menace.self_test_service import SelfTestService
from menace.context_builder_util import create_context_builder

def integrate(paths):
    # decide which modules to merge
    return {"integrated": paths, "redundant": []}

builder = create_context_builder()
svc = SelfTestService(integration_callback=integrate, context_builder=builder)
results, passed = svc.run_once()
print(results["integration"])
```

## SandboxSettings flags

`sandbox_settings.SandboxSettings` provides switches that control the discovery
and integration pipeline:

- `auto_include_isolated` (`SANDBOX_AUTO_INCLUDE_ISOLATED`) – include modules
  found by `discover_isolated_modules` and merge passing ones automatically.
- `recursive_orphan_scan` (`SANDBOX_RECURSIVE_ORPHANS`) – follow orphan
  dependencies when generating the test list.
- `recursive_isolated` (`SANDBOX_RECURSIVE_ISOLATED`) – expand isolated modules
  through their imports.

When enabled, passing modules are written to
`sandbox_data/module_map.json` and `environment.generate_workflows_for_modules`
creates single‑step workflows so subsequent cycles can schedule the new code.
Setting `SANDBOX_CLEAN_ORPHANS=1` removes integrated entries from the orphan
cache after each run.

## Recursive inclusion flow

### Recursive orphan discovery

The service participates in the sandbox's recursive module discovery. During
discovery the helper `sandbox_runner.discover_recursive_orphans` walks each
orphan's imports, collects any local dependencies and returns a mapping where
each module lists its importing `parents` and whether it was deemed
`redundant`. Setting `SANDBOX_RECURSIVE_ORPHANS=1` causes the walk to follow an
orphan's entire import chain so new modules are discovered recursively. When
`SANDBOX_AUTO_INCLUDE_ISOLATED=1` is set, isolated modules are added to the scan
and their dependencies are explored in the same recursive manner. Every
candidate is then executed in an ephemeral sandbox via
`pytest` so side effects are contained.
Modules that pass are automatically appended to `sandbox_data/module_map.json`
via `module_index_db.ModuleIndexDB` and merged into the sandbox's workflows
through `sandbox_runner.environment.auto_include_modules` with
`recursive=True` by default. Entries are indexed using repository-relative paths
so files sharing a name in different directories do not collide:

```bash
mkdir -p pkg_a pkg_b
echo 'VALUE=1' > pkg_a/common.py
echo 'VALUE=2' > pkg_b/common.py
python -m menace.self_test_service run pkg_a/common.py pkg_b/common.py --auto-include-isolated
# module_map.json now lists pkg_a/common.py and pkg_b/common.py separately
```
Toggle recursion with
`SELF_TEST_RECURSIVE_ORPHANS=0` or `SANDBOX_RECURSIVE_ORPHANS=0`. Opt out by
setting `SELF_TEST_DISABLE_AUTO_INTEGRATION=1` or by supplying a custom
`integration_callback`. When no callback is provided the service uses this
behaviour by default. By default `auto_env_setup.ensure_env` sets
`SELF_TEST_RECURSIVE_ORPHANS=1` and
`SELF_TEST_RECURSIVE_ISOLATED=1` so orphan and isolated modules are followed
through their import chains. Disable recursion with the CLI flags
`--no-recursive-include` or `--no-recursive-isolated`, or set the corresponding
environment variables to `0`. The complementary flags `--recursive-include` and
`--recursive-isolated` explicitly enable this behaviour and set
`SANDBOX_RECURSIVE_ORPHANS` and `SANDBOX_RECURSIVE_ISOLATED` respectively. Use
`--auto-include-isolated` (or `SANDBOX_AUTO_INCLUDE_ISOLATED=1`) to force
isolated discovery and `--clean-orphans`/`SANDBOX_CLEAN_ORPHANS=1` to drop
passing entries from `orphan_modules.json`.

The constructor exposes `auto_include_isolated` and `recursive_isolated`
parameters that mirror `SandboxSettings`. Setting
`auto_include_isolated=True` forces isolated discovery and automatically
enables recursive processing of their dependencies. `recursive_isolated`
controls whether discovered isolated modules have their imports followed.

### Environment variables and CLI flags

- `--recursive-include` / `SELF_TEST_RECURSIVE_ORPHANS` /
  `SANDBOX_RECURSIVE_ORPHANS` – recurse through orphan modules and their
  imports.
- `--recursive-isolated` / `SELF_TEST_RECURSIVE_ISOLATED` /
  `SANDBOX_RECURSIVE_ISOLATED` – include dependencies of isolated modules.
- `--auto-include-isolated` / `SANDBOX_AUTO_INCLUDE_ISOLATED` – force inclusion
  of modules returned by `discover_isolated_modules`.
- `--clean-orphans` / `SANDBOX_CLEAN_ORPHANS` – remove passing entries from
  `orphan_modules.json` after integration.
- `SELF_TEST_DISABLE_AUTO_INTEGRATION` – skip automatic merging of passing
  modules into `module_map.json` and workflow updates.

Example enabling recursion:

```bash
# CLI flags
python -m menace.self_test_service run --recursive-include --recursive-isolated

# Environment variables
SELF_TEST_RECURSIVE_ORPHANS=1 SELF_TEST_RECURSIVE_ISOLATED=1 \
  python -m menace.self_test_service run
```

### Classification and metrics storage

Classification results from `discover_recursive_orphans` are written to
`sandbox_data/orphan_classifications.json` alongside the primary orphan cache
`sandbox_data/orphan_modules.json`. Test coverage and runtime statistics are
logged via `MetricsDB` in `sandbox_data/metrics.db`, where they back the
Prometheus gauges such as `orphan_modules_tested_total`.

### Redundant module handling

`discover_recursive_orphans` annotates each entry with a `redundant` flag based
on `orphan_analyzer.analyze_redundancy`. Modules flagged as redundant are logged
but skipped during auto-inclusion; their classification and `parents` information
is still recorded. The `SelfImprovementEngine` consults the same metadata and
refuses to merge redundant modules into workflows even if they pass their
tests.

These flags drive the workflow integration stage: passing modules and their
helpers are written to `module_map.json` and merged into existing flows via
`try_integrate_into_workflows`. Tests such as `tests/test_recursive_isolated.py`
and `tests/test_self_test_service_recursive_integration.py` verify that
recursive discovery executes supporting files and integrates them into the
module map. See the
[isolated module example](autonomous_sandbox.md#example-isolated-module-discovery-and-integration)
for a step-by-step walkthrough.

## Command Line Interface

Run the tests manually via the CLI:

```bash
python -m menace.self_test_service run --workers 4 --metrics-port 8004 tests/unit
```

Options include:

- `--workers` – number of pytest workers
- `--container-image` – Docker image used with `--use-container`
- `--use-container` – execute tests inside a container when available
- `--container-runtime` – container runtime executable (e.g. `docker` or `podman`)
- `--docker-host` – Docker/Podman host or URL for remote engines
- `--metrics-port` – expose Prometheus gauges on this port
- `--include-orphans` – also run modules listed in `sandbox_data/orphan_modules.json`
 - `--discover-orphans` – automatically run `scripts/find_orphan_modules.py` and include the results
 - `--auto-include-isolated` – automatically run `discover_isolated_modules` and append results (use `--no-auto-include-isolated` to disable)
 - `--recursive-include` – recurse through orphan dependencies (`--no-recursive-include` to disable)
 - `--recursive-isolated` – recurse through dependencies of isolated modules (`--no-recursive-isolated` to disable)
 - `--no-recursive-include` – do not recurse through orphan dependencies
 - `--no-recursive-isolated` – do not recurse through dependencies of isolated modules
 - `--clean-orphans` – remove passing entries from `orphan_modules.json`

Recursion through orphan dependencies is enabled by default; use
`--no-recursive-include` to limit the search to top-level modules or set
`SELF_TEST_RECURSIVE_ORPHANS=0` or `SANDBOX_RECURSIVE_ORPHANS=0`.
Remove stale containers left over from interrupted runs with:

```bash
python -m menace.self_test_service cleanup
```

This command accepts `--container-runtime`, `--docker-host` and `--retries`
options identical to those used with `run`.

Example running tests inside a remote Podman instance:

```bash
python -m menace.self_test_service run \
    --use-container \
    --container-runtime podman \
    --docker-host ssh://user@remote.example.com/run/podman.sock \
    tests/unit
```

Run tests continuously on a schedule using Docker or Podman:

```bash
python -m menace.self_test_service run-scheduled --interval 3600 --history test_history.json \
    --metrics-port 8004
```

The `run-scheduled` command shares the same options as `run` and writes each
cycle's pass/fail counts, coverage and runtime to the file specified with
`--history`.

## Offline Workflow

Set `MENACE_OFFLINE_INSTALL=1` when dependencies and container images are
available locally.  If `--use-container` is specified the service loads a
prebuilt image tarball before running the tests.  Provide the tarball path via
`MENACE_SELF_TEST_IMAGE_TAR`.

Container runs are serialized across processes using a file lock. Set the lock
file path with `SELF_TEST_LOCK_FILE` (default `sandbox_data/self_test.lock`).

## Orphan Modules

Run `scripts/find_orphan_modules.py` to locate Python files that are not
referenced by any tests. The script writes the list to
`sandbox_data/orphan_modules.json` which the service loads automatically. Orphan
modules are processed by default (`SELF_TEST_DISABLE_ORPHANS=0`); set
`SELF_TEST_DISABLE_ORPHANS=1` or pass `--include-orphans` to skip the file.

Automatic scanning is enabled by default (`SELF_TEST_DISCOVER_ORPHANS=1`). Set
`SELF_TEST_DISCOVER_ORPHANS=0` to disable it. The discovered modules are saved to the
same file and appended to the test queue on the next run. Use `--refresh-orphans`
to force a new scan when the list already exists. Recursion through orphan
dependencies is enabled by default, mirroring the effect of setting
`SELF_TEST_RECURSIVE_ORPHANS=1` and `SANDBOX_RECURSIVE_ORPHANS=1`. These defaults
are written to the generated `.env` by `auto_env_setup.ensure_env`. Disable
recursion with `SELF_TEST_RECURSIVE_ORPHANS=0`, `SANDBOX_RECURSIVE_ORPHANS=0`
or the `--no-recursive-include` option. The search uses
`sandbox_runner.discover_recursive_orphans` from `sandbox_runner.orphan_discovery`
to walk each orphan's imports until no new local modules remain, returning a
mapping from each module to the module(s) that imported it. This function
is part of the public `sandbox_runner` API and may be imported for custom
workflows. When Docker or other heavy dependencies are unavailable set
`MENACE_LIGHT_IMPORTS=1` before the import to skip environment
initialisation. Modules returned by `discover_isolated_modules` are not included
unless `--auto-include-isolated` is used or `SELF_TEST_AUTO_INCLUDE_ISOLATED=1` is set.
Dependency traversal for these modules is enabled by default through `SELF_TEST_RECURSIVE_ISOLATED=1` and
`SANDBOX_RECURSIVE_ISOLATED=1`; these values are also persisted to `.env` by
`auto_env_setup.ensure_env`. Disable it with `SELF_TEST_RECURSIVE_ISOLATED=0`,
`SANDBOX_RECURSIVE_ISOLATED=0` or the `--no-recursive-isolated` option.

Example running tests with orphan and isolated discovery:

```bash
python -m menace.self_test_service run tests/unit \
    --include-orphans --discover-orphans --auto-include-isolated \
    --clean-orphans
```

Passing orphan modules are merged into `module_map.json` so subsequent sandbox
runs can schedule them alongside existing workflows.

When launched from `sandbox_runner`, recursion through orphan and isolated
dependencies is enabled by default. Set `SANDBOX_RECURSIVE_ORPHANS=0` to skip
their dependency chains and `SANDBOX_RECURSIVE_ISOLATED=0` to disable isolated
recursion. Passing modules
are merged into `module_map.json` and existing workflows are updated via
`try_integrate_into_workflows` on the next run. If a discovered module fits an existing
workflow group, the sandbox tries to merge it into those flows automatically.
Setting `SANDBOX_AUTO_INCLUDE_ISOLATED=1` forces discovery of isolated modules
so their dependencies are scanned.
Modules flagged as redundant by `orphan_analyzer.analyze_redundancy` are skipped
during this integration step.
Set `SANDBOX_CLEAN_ORPHANS=1` to mirror the `--clean-orphans` option and remove
integrated modules from the orphan list after each run. The improvement engine
also honours `SANDBOX_RECURSIVE_ISOLATED`; set it to `0`, `false` or `no` to
disable recursion when discovering isolated modules. When present, the service
forces both `discover_isolated` and recursive discovery to `True` regardless of
the provided arguments or environment variables.

## Automatic Orphan Detection

By default the service scans `sandbox_data/orphan_modules.json` and runs any
listed modules. If the file does not exist it automatically searches for orphans
using `sandbox_runner.discover_recursive_orphans`. Orphans are processed unless
`SELF_TEST_DISABLE_ORPHANS=1` or `--include-orphans` is supplied. Disable the
search with `SELF_TEST_DISCOVER_ORPHANS=0` and turn off recursion with
`SELF_TEST_RECURSIVE_ORPHANS=0`, `SANDBOX_RECURSIVE_ORPHANS=0` or
`--no-recursive-include`. The generated list is saved for future runs and the
new modules are tested immediately. Isolated modules discovered by
`discover_isolated_modules` are processed when `--auto-include-isolated` is supplied
or `SELF_TEST_AUTO_INCLUDE_ISOLATED=1` is set. Disable this step by setting
`SELF_TEST_AUTO_INCLUDE_ISOLATED=0`. Dependency traversal is enabled by default
through `SELF_TEST_RECURSIVE_ISOLATED=1` and `SANDBOX_RECURSIVE_ISOLATED=1`;
use `--no-recursive-isolated` or set those variables to `0` to skip their
dependencies when generating the test list. Passing orphan modules are merged
into `module_map.json` automatically once the tests complete.

Results for these files are summarised under the `orphan_total`,
`orphan_failed` and `orphan_passed` fields of the returned statistics. The
update step only records new modules in `orphan_modules.json`. They are merged
into `module_map.json` when `_refresh_module_map()` runs with the list of
passing modules so subsequent cycles treat them like normal modules.
