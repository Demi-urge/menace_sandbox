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

