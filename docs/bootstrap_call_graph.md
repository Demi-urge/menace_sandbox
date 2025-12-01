# Bootstrap dispatch call graph

## Environment bootstrap entrypoints
- `EnvironmentBootstrapper.bootstrap()` orchestrates the critical → provisioning → optional phases and delegates to `_critical_prerequisites`, `_provisioning_phase`, and `_optional_tail` based on resolved budgets and background handling preferences.【F:environment_bootstrap.py†L1806-L1868】
- `_critical_prerequisites` fans out into config discovery and dependency checks: `ensure_config`, secret export via either `SecretsManager` or `VaultSecretProvider`, command/driver/package probes, optional remote dependency checks, and opportunistic Python dependency installation through `startup_checks`/`SystemProvisioner`.【F:environment_bootstrap.py†L1669-L1709】
- `_provisioning_phase` performs systemd timer enablement, extra package installs, migrations, the Terraform-backed `InfrastructureBootstrapper.bootstrap()`, optional remote host bootstraps, and schedules the vector service scheduler (`vector_service.embedding_scheduler.start_scheduler_from_env`) once provisioning gates open.【F:environment_bootstrap.py†L1711-L1797】
- `_optional_tail` and `bootstrap_vector_assets` warm optional assets by downloading embedding bundles, seeding `VectorMetricsDB`, and initializing ROI history files when policy allows.【F:environment_bootstrap.py†L1602-L1632】【F:environment_bootstrap.py†L1799-L1803】

## Readiness stage mapping
- `bootstrap_readiness.READINESS_STAGES` defines step→stage groupings (DB/index load, retriever hydration, vector seeding, orchestrator promotion, background loops) used by phase schedulers, while `build_stage_deadlines()` scales per-stage budgets/floors from component minima and adaptive telemetry before feeding them to the bootstrap manager.【F:bootstrap_readiness.py†L23-L189】【F:bootstrap_readiness.py†L108-L208】

## Pipeline bootstrap single‑flight entry
- `coding_bot_interface.prepare_pipeline_for_bootstrap()` is the shared entry that wraps pipeline construction with a bootstrap sentinel manager. It first consults the dependency broker and global bootstrap coordinator, short-circuiting when placeholders or active promises exist and raising if a broker placeholder lacks an owner—preventing recursive bootstrap chains before delegating to the underlying implementation.【F:coding_bot_interface.py†L5659-L5739】

### End-to-end bootstrap call graph (entrypoints → re-entrancy sinks)
```
environment_bootstrap.ensure_bootstrapped
  └── bootstrap_helpers.ensure_bootstrapped (import-deferral shim)
        ├── cognition_layer._get_layer → advertises gate placeholders before constructing CognitionLayer caches【F:cognition_layer.py†L60-L90】
        ├── prediction_manager_bot._get_registry/_get_data_bot → waits on readiness then advertises placeholders prior to registry hydration【F:prediction_manager_bot.py†L243-L264】
        ├── research_aggregator_bot._ensure_runtime_dependencies → gates heavy helper creation on readiness and broker placeholders【F:research_aggregator_bot.py†L260-L330】
        └── orchestrator_loader._bootstrap_placeholders → reuses active bootstrap pipeline/manager or resolves gate placeholders for orchestrator promotion【F:orchestrator_loader.py†L32-L63】

coding_bot_interface.prepare_pipeline_for_bootstrap (single-flight + recursion guard)
  └── consumers above eventually request pipelines via dependency-broker placeholders rather than direct construction; re-entry is short-circuited when the broker already exposes a placeholder/promise or when active depth exceeds the configured guard cap.【F:coding_bot_interface.py†L5659-L5739】
```

**Allowed entrypoints (keep calls on the single-flight path)**
- Use `bootstrap_helpers.ensure_bootstrapped(...)` (or `environment_bootstrap.ensure_bootstrapped(...)`) to initialise readiness once per process; avoid ad-hoc bootstraps from module initialisers.
- Use `coding_bot_interface.prepare_pipeline_for_bootstrap(...)` with the bootstrap guard enabled to obtain pipelines/managers via the dependency broker.

**Forbidden recursive patterns**
- Do **not** instantiate pipelines directly inside bootstrap-sensitive modules; always route through the broker-aware placeholder helpers above so repeated imports reuse the active bootstrap promise.
- Avoid bypassing the broker/guard (e.g., calling `_prepare_pipeline_for_bootstrap_impl` directly or wiring bespoke sentinels) because doing so skips the recursion cap and can deadlock when placeholders lack an owner.

## High-level call flow
- External callers invoke `EnvironmentBootstrapper.bootstrap()` → dispatches critical/provisioning/optional tasks and schedules vector services.
- Vector/cognition/prediction/research orchestration modules then call into `prepare_pipeline_for_bootstrap()` (directly or via placeholder advertisement helpers) guarded by the bootstrap gate and dependency broker, preventing re-entrant pipeline spins while bootstrap phases are still active.
