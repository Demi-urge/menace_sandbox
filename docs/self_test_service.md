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
`sandbox_data/orphan_modules.json` which the service can load when started with
`--include-orphans`.

Set `SELF_TEST_DISCOVER_ORPHANS=1` or pass `--discover-orphans` to scan the
repository automatically. The discovered modules are saved to the same file and
appended to the test queue on the next run. Use `--refresh-orphans` to force a
new scan when the list already exists. Enable recursive discovery with
`SELF_TEST_RECURSIVE_ORPHANS=1` or `--recursive-orphans` so that dependent
modules are added when an orphan imports other files. The search uses
`sandbox_runner.discover_recursive_orphans` to walk each orphan's imports
until no new local modules remain.
Pass `--discover-isolated` or set `SELF_TEST_DISCOVER_ISOLATED=1` to include
modules returned by `discover_isolated_modules` before searching for orphans.

Example running tests with recursive orphan discovery:

```bash
python -m menace.self_test_service run tests/unit \
    --include-orphans --discover-orphans --recursive-orphans
```

When launched from `sandbox_runner`, set `SANDBOX_RECURSIVE_ORPHANS=1` to
achieve the same behaviour. Passing modules are added to `module_map.json`
and simple one-step workflows are generated automatically on the next run.
If a discovered module fits an existing workflow group, the sandbox tries to merge it into those flows automatically.

## Automatic Orphan Detection

When the service is launched with `--include-orphans` or the environment
variable `SANDBOX_INCLUDE_ORPHANS=1`, it searches for
`sandbox_data/orphan_modules.json` and runs each listed module alongside the
regular test suite. If the file does not exist, the service attempts to locate
orphan modules automatically using `sandbox_runner.discover_recursive_orphans`
when `SELF_TEST_RECURSIVE_ORPHANS=1` or falling back to
`sandbox_runner.discover_orphan_modules` and the helper `scripts/find_orphan_modules.py`.
The generated list is
saved for future runs and the new modules are tested immediately.
Setting `SELF_TEST_DISCOVER_ISOLATED=1` or passing `--discover-isolated` tells
the service to include modules returned by `discover_isolated_modules` during
these automatic scans.

Results for these files are summarised under the `orphan_total`,
`orphan_failed` and `orphan_passed` fields of the returned statistics. When
triggered from `sandbox_runner` any successfully tested orphans are merged into
the module map so subsequent cycles treat them like normal modules.
