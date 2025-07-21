# Self Test Service

`SelfTestService` periodically runs the project's unit tests to catch regressions.
Specify custom pytest arguments via `SELF_TEST_ARGS` or the constructor. By
default tests run directly on the host, but enabling `use_container=True`
executes each provided test path inside a Docker container when available. Each
container is removed on completion so test runs are isolated.

Coverage percentage and total runtime are recorded in `MetricsDB` when a
`DataBot` is supplied. Results can also be streamed incrementally by providing
`result_callback` which receives a dictionary after each file and a final
summary.
