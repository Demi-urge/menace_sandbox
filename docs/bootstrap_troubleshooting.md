# Bootstrap staged readiness

Menace now surfaces bootstrap progress in **three phases** (critical, provisioning, optional) so downstream services and watchdogs can react before background warmup completes.

## Readiness gates

* Each phase registers a readiness gate and emits `bootstrap-readiness` events when it starts, degrades, fails, or completes.
* The supervisor marks Menace **online** once the `critical` and `provisioning` gates are ready. Optional/background loops may continue after this point without blocking the service start.
* Optional phase deadline overruns trigger a `degraded` readiness event instead of a fatal exit. The watchdog should treat this as "online with optional work pending" and continue polling the gate until it reports `ready`.

## Metrics and logs

* Readiness events include the gate snapshot (`readiness_gates`) and legacy boolean tokens. Look for `bootstrap-readiness-summary` right after `service_supervisor` finishes bootstrapping to capture the final state.
* Background tasks are reported separately so you can distinguish active warmup threads from failed gates.

## Systemd/service wrappers

* Wrappers should no longer wait for a single "all ready" signal. The default `menace.service` unit exports `MENACE_BOOTSTRAP_READINESS_MODE=staged` so watchdogs can opt into staged readiness handling.
* Treat `critical` + `provisioning` readiness as sufficient to accept traffic; continue to monitor the optional gate for warnings instead of failing the unit.

## Timeouts

* Increase the env var `MENACE_BOOTSTRAP_WAIT_SECS` to a higher value such as `360` to handle more complex development environments.
* If you need more fine-grained control, increase `BOOTSTRAP_STEP_TIMEOUT`, `PREPARE_PIPELINE_VECTORIZER_BUDGET_SECS`, `PREPARE_PIPELINE_RETRIEVER_BUDGET_SECS`, `PREPARE_PIPELINE_DB_WARMUP_BUDGET_SECS`, and `PREPARE_PIPELINE_ORCHESTRATOR_BUDGET_SECS`.
* High-load or vector-heavy hosts may raise component floors automatically using the persisted heartbeat data. Inspect the state at `~/.menace_bootstrap_timeout_state.json` to confirm the elevated floors and stagger additional cluster bootstraps when the watchdog reports contention.

