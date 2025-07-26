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
python -m menace.self_test_service run --workers 4 tests/unit
```

Options include:

- `--workers` – number of pytest workers
- `--container-image` – Docker image used with `--use-container`
- `--use-container` – execute tests inside a container when available
- `--container-runtime` – container runtime executable (e.g. `docker` or `podman`)
- `--docker-host` – Docker/Podman host or URL for remote engines

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
python -m menace.self_test_service run-scheduled --interval 3600 --history test_history.json
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
