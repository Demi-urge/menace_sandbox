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

* Increase the env var `MENACE_BOOTSTRAP_WAIT_SECS` to a higher value such as `360` (the enforced floor) to handle more complex development environments; vector-heavy runs should respect the `MENACE_BOOTSTRAP_VECTOR_WAIT_SECS` floor of `540`.
* If you need more fine-grained control, increase `BOOTSTRAP_STEP_TIMEOUT`, `PREPARE_PIPELINE_VECTORIZER_BUDGET_SECS`, `PREPARE_PIPELINE_RETRIEVER_BUDGET_SECS`, `PREPARE_PIPELINE_DB_WARMUP_BUDGET_SECS`, and `PREPARE_PIPELINE_ORCHESTRATOR_BUDGET_SECS` (floors: vectorizers **720s**, retrievers/DB warmup **480s**, orchestrator/config **420s**).
* High-load or vector-heavy hosts may raise component floors automatically using the persisted heartbeat data. Inspect the state at `~/.menace_bootstrap_timeout_state.json` to confirm the elevated floors and stagger additional cluster bootstraps when the watchdog reports contention.

## Cascade failures and the dependency broker

When a new helper spins up a `ModelAutomationPipeline` during bootstrap without advertising the placeholder manager, each import can recursively trigger `prepare_pipeline_for_bootstrap`, saturating the recursion guard while none of the attempts progress. This cascade presents as rapid-fire `prepare_pipeline_for_bootstrap` single-flight owner/reuse messages or `prepare_pipeline.bootstrap.recursion_refused` entries without the watchdog ever reporting real progress. To break the loop:

1. Ensure the dependency broker is active at the start of bootstrap by calling `advertise_bootstrap_placeholder` (or routing through `prepare_pipeline_for_bootstrap`, which advertises automatically) before importing modules that may lazily construct pipelines.
2. Confirm new helpers are using the global single-flight guard instead of building their own pipeline: importing `prepare_pipeline_for_bootstrap` and leaving `bootstrap_guard=True` prevents parallel or nested attempts from piling up.
3. If recursion logs persist, identify the module that is importing the pipeline before the broker placeholder exists (the `prepare_pipeline_for_bootstrap` spam usually includes the caller in structured logs). Move that import behind the broker advertisement or thread the sentinel manager into the new helper so it reuses the active bootstrap instance.

## Adding modules to the bootstrap chain

When introducing a new module that constructs or depends on `ModelAutomationPipeline` during bootstrap, wire it into the existing chain to avoid recursion:

1. Export the bootstrap placeholder early by calling `advertise_bootstrap_placeholder` in the entry script that orchestrates imports for the new module, or delegate pipeline construction to `prepare_pipeline_for_bootstrap`.
2. Keep `bootstrap_guard` enabled so the single-flight coordinator serialises concurrent callers; only disable it for tightly controlled unit tests.
3. Pass the promoted manager back into helpers (for example via the `manager` kwarg) once `prepare_pipeline_for_bootstrap` returns so subsequent imports skip sentinel creation entirely.
4. Add a regression note in the module's docstring or README linking to this troubleshooting section, making it clear that future imports must reuse the brokered sentinel instead of constructing their own pipeline.

Following this pattern keeps the guardrails intact and prevents new helpers from triggering the cascade failure mode during bootstrap.

