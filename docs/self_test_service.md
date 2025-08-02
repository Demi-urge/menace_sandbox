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
- `--discover-isolated` – automatically run `discover_isolated_modules` and append results
- `--no-recursive-orphans` – do not recurse through orphan dependencies
- `--no-recursive-isolated` – do not recurse through dependencies of isolated modules
- `--auto-include-isolated` – automatically discover isolated modules (recursion enabled by default; equivalent to `SANDBOX_AUTO_INCLUDE_ISOLATED=1`)
- `--clean-orphans` – remove passing entries from `orphan_modules.json`
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
`sandbox_data/orphan_modules.json` which the service loads automatically. Set
`SELF_TEST_DISABLE_ORPHANS=1` or pass `--include-orphans` to skip the file.

Automatic scanning is enabled by default. Set `SELF_TEST_DISCOVER_ORPHANS=0`
to disable it. The discovered modules are saved to the
same file and appended to the test queue on the next run. Use `--refresh-orphans`
to force a new scan when the list already exists. Recursion through orphan
dependencies is enabled by default via `SELF_TEST_RECURSIVE_ORPHANS=1` and
`SANDBOX_RECURSIVE_ORPHANS=1`. Disable recursion with
`SELF_TEST_RECURSIVE_ORPHANS=0`, `SANDBOX_RECURSIVE_ORPHANS=0` or the
`--no-recursive-orphans` option. The search uses
`sandbox_runner.discover_recursive_orphans` from `sandbox_runner.orphan_discovery`
to walk each orphan's imports until no new local modules remain. This function
is part of the public `sandbox_runner` API and may be imported for custom
workflows. When Docker or other heavy dependencies are unavailable set
`MENACE_LIGHT_IMPORTS=1` before the import to skip environment
initialisation. Modules returned by `discover_isolated_modules` are not included
unless `--discover-isolated` is used or `SELF_TEST_DISCOVER_ISOLATED=1` is set.
Dependency traversal for these modules is enabled by default through `SELF_TEST_RECURSIVE_ISOLATED=1` and
`SANDBOX_RECURSIVE_ISOLATED=1`. Disable it with `SELF_TEST_RECURSIVE_ISOLATED=0`,
`SANDBOX_RECURSIVE_ISOLATED=0` or the `--no-recursive-isolated` option.

Example running tests with orphan and isolated discovery:

```bash
python -m menace.self_test_service run tests/unit \
    --include-orphans --discover-orphans --discover-isolated \
    --clean-orphans
```

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
using `sandbox_runner.discover_recursive_orphans`. Set `SELF_TEST_DISABLE_ORPHANS=1`
or pass `--include-orphans` to skip the feature. Disable the search with
`SELF_TEST_DISCOVER_ORPHANS=0` and turn off recursion with
`SELF_TEST_RECURSIVE_ORPHANS=0`, `SANDBOX_RECURSIVE_ORPHANS=0` or
`--no-recursive-orphans`. The generated list is saved for future runs and the
new modules are tested immediately. Isolated modules discovered by
`discover_isolated_modules` are processed when `--discover-isolated` is supplied
or `SELF_TEST_DISCOVER_ISOLATED=1` is set. Disable this step by setting
`SELF_TEST_DISCOVER_ISOLATED=0`. Dependency traversal is enabled by default
through `SELF_TEST_RECURSIVE_ISOLATED=1` and `SANDBOX_RECURSIVE_ISOLATED=1`;
use `--no-recursive-isolated` or set those variables to `0` to skip their
dependencies when generating the test list. Passing orphan modules are merged
into `module_map.json` automatically once the tests complete.

Results for these files are summarised under the `orphan_total`,
`orphan_failed` and `orphan_passed` fields of the returned statistics. The
update step only records new modules in `orphan_modules.json`. They are merged
into `module_map.json` when `_refresh_module_map()` runs with the list of
passing modules so subsequent cycles treat them like normal modules.
