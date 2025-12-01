# Bootstrap contract and readiness dependencies

Bootstrap-sensitive modules share a single entrypoint and readiness contract so they do not race each other or attempt partial setup while the pipeline is still warming up.

## Single-entry bootstrap contract

* **Advertise a placeholder immediately.** Entry scripts call `coding_bot_interface.advertise_bootstrap_placeholder()` during import so later modules can reuse the sentinel manager/pipeline instead of kicking off a new bootstrap attempt.
* **Wait on the gate.** All callers block on `bootstrap_gate.wait_for_bootstrap_gate()` (or `resolve_bootstrap_placeholders()` when they need the returned handles) before invoking heavy helpers. The gate serialises contenders and coordinates host-level backoff so only one bootstrap chain runs at a time.
* **Claim the slot via the helper.** Use `coding_bot_interface.prepare_pipeline_for_bootstrap(..., bootstrap_guard=True)` to reuse the gate placeholders and claim the single-flight slot. Direct calls into `_bootstrap_manager`, bespoke pipeline constructors, or ad-hoc sentinels are forbidden because they bypass idempotency tracking and can deadlock the gate.

Following this contract keeps every entrypoint—GUI preflight, CLI launchers, orchestration surfaces, and import-time warmers—on the same single-flight path and ensures each one inherits the shared heartbeats, watchdog budgets, and telemetry.

## Depending on readiness

Modules that need the bootstrap pipeline should register their dependency on readiness explicitly:

* **Block until ready:** Call `bootstrap_gate.wait_for_bootstrap_gate()` before wiring dependent services so late imports do not assume the pipeline is available prematurely.
* **Reuse placeholders:** When a module only needs to hand out handles, reuse `resolve_bootstrap_placeholders()` instead of constructing fresh pipelines; this keeps the advertised sentinel and broker aligned with the active bootstrap.
* **Gate optional work:** If a helper can run in a degraded mode, branch on `bootstrap_gate.wait_for_bootstrap_gate` outcomes or heartbeat presence so optional features defer until the readiness signal is published.

Treat readiness as a prerequisite for any service that interacts with the self-coding pipeline, vector warm-up, or orchestrator state so that new modules do not bypass the gate and trigger re-entrant bootstrap attempts.
