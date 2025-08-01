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
- `--discover-isolated` – load modules returned by `discover_isolated_modules` before scanning for orphans
- `--recursive-orphans` – recursively include dependencies of discovered orphans
- `--recursive-isolated` – recurse through dependencies of isolated modules
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

Automatic scanning is enabled by default. Set `SELF_TEST_DISCOVER_ORPHANS=0` or
use `--discover-orphans` to disable it. The discovered modules are saved to the
same file and appended to the test queue on the next run. Use `--refresh-orphans`
to force a new scan when the list already exists. Disable recursion with
`SELF_TEST_RECURSIVE_ORPHANS=0` or `--recursive-orphans`. The search uses
`sandbox_runner.discover_recursive_orphans` to walk each orphan's imports
until no new local modules remain.
Pass `--discover-isolated` or set `SELF_TEST_DISCOVER_ISOLATED=1` to include
modules returned by `discover_isolated_modules` before searching for orphans.
Enable dependency traversal for these modules with `SELF_TEST_RECURSIVE_ISOLATED=1`
or the `--recursive-isolated` option.

Example running tests with recursive orphan and isolated discovery:

```bash
python -m menace.self_test_service run tests/unit \
    --include-orphans --discover-orphans --discover-isolated \
    --recursive-orphans --recursive-isolated
```

When launched from `sandbox_runner`, set `SANDBOX_RECURSIVE_ORPHANS=1` and
`SANDBOX_RECURSIVE_ISOLATED=1` to achieve the same behaviour. Passing modules
are merged into `module_map.json` and may be incorporated into existing
workflows automatically on the next run. If a discovered module fits an existing
workflow group, the sandbox tries to merge it into those flows automatically.
Setting `SANDBOX_AUTO_INCLUDE_ISOLATED=1` applies both flags for convenience.

## Automatic Orphan Detection

By default the service scans `sandbox_data/orphan_modules.json` and runs any
listed modules. If the file does not exist it automatically searches for orphans
using `sandbox_runner.discover_recursive_orphans`. Set `SELF_TEST_DISABLE_ORPHANS=1`
or pass `--include-orphans` to skip the feature. Disable the search with
`SELF_TEST_DISCOVER_ORPHANS=0` or `--discover-orphans` and turn off recursion
with `SELF_TEST_RECURSIVE_ORPHANS=0` or `--recursive-orphans`.
The generated list is
saved for future runs and the new modules are tested immediately.
Setting `SELF_TEST_DISCOVER_ISOLATED=1` or passing `--discover-isolated` tells
the service to include modules returned by `discover_isolated_modules` during
these automatic scans. Use `SELF_TEST_RECURSIVE_ISOLATED=1` or `--recursive-isolated`
to follow their dependencies when generating the test list.

Results for these files are summarised under the `orphan_total`,
`orphan_failed` and `orphan_passed` fields of the returned statistics. When
triggered from `sandbox_runner` any successfully tested orphans are merged into
the module map so subsequent cycles treat them like normal modules.
