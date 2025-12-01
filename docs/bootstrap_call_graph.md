# Bootstrap dispatch call graph

## Environment bootstrap entrypoints
- `EnvironmentBootstrapper.bootstrap()` orchestrates the critical → provisioning → optional phases and delegates to `_critical_prerequisites`, `_provisioning_phase`, and `_optional_tail` based on resolved budgets and background handling preferences.【F:environment_bootstrap.py†L1806-L1868】
- `_critical_prerequisites` fans out into config discovery and dependency checks: `ensure_config`, secret export via either `SecretsManager` or `VaultSecretProvider`, command/driver/package probes, optional remote dependency checks, and opportunistic Python dependency installation through `startup_checks`/`SystemProvisioner`.【F:environment_bootstrap.py†L1669-L1709】
- `_provisioning_phase` performs systemd timer enablement, extra package installs, migrations, the Terraform-backed `InfrastructureBootstrapper.bootstrap()`, optional remote host bootstraps, and schedules the vector service scheduler (`vector_service.embedding_scheduler.start_scheduler_from_env`) once provisioning gates open.【F:environment_bootstrap.py†L1711-L1797】
- `_optional_tail` and `bootstrap_vector_assets` warm optional assets by downloading embedding bundles, seeding `VectorMetricsDB`, and initializing ROI history files when policy allows.【F:environment_bootstrap.py†L1602-L1632】【F:environment_bootstrap.py†L1799-L1803】

## Readiness stage mapping
- `bootstrap_readiness.READINESS_STAGES` defines step→stage groupings (DB/index load, retriever hydration, vector seeding, orchestrator promotion, background loops) used by phase schedulers, while `build_stage_deadlines()` scales per-stage budgets/floors from component minima and adaptive telemetry before feeding them to the bootstrap manager.【F:bootstrap_readiness.py†L23-L189】【F:bootstrap_readiness.py†L108-L208】

## Pipeline bootstrap single‑flight entry
- `coding_bot_interface.prepare_pipeline_for_bootstrap()` is the shared entry that wraps pipeline construction with a bootstrap sentinel manager. It first consults the dependency broker and global bootstrap coordinator, short-circuiting when placeholders or active promises exist and raising if a broker placeholder lacks an owner—preventing recursive bootstrap chains before delegating to the underlying implementation.【F:coding_bot_interface.py†L5608-L5705】

## Recipients that re-enter bootstrap coordination
- `research_aggregator_bot._bootstrap_placeholders()` waits on `bootstrap_gate.resolve_bootstrap_placeholders()` and then advertises pipeline/manager placeholders through the dependency broker so downstream helpers reuse the active bootstrap chain instead of creating new ones.【F:research_aggregator_bot.py†L99-L116】
- `prediction_manager_bot` imports the bootstrap gate and advertises placeholders from either the active pipeline (`get_active_bootstrap_pipeline`) or the resolved gate output before its registry/data bot proxies hydrate, keeping prediction orchestration on the shared bootstrap promise.【F:prediction_manager_bot.py†L11-L33】【F:prediction_manager_bot.py†L197-L200】
- `cognition_layer` resolves placeholders on import and advertises them before constructing the vector `CognitionLayer`, ensuring cognition warm-ups reuse the broker-published bootstrap sentinel.【F:cognition_layer.py†L22-L46】
- `orchestrator_loader._bootstrap_placeholders()` checks for active bootstrap pipeline/manager instances and otherwise waits on the bootstrap gate before advertising broker placeholders, keeping orchestrator construction aligned with the single-flight bootstrap state.【F:orchestrator_loader.py†L14-L39】

## High-level call flow
- External callers invoke `EnvironmentBootstrapper.bootstrap()` → dispatches critical/provisioning/optional tasks and schedules vector services.
- Vector/cognition/prediction/research orchestration modules then call into `prepare_pipeline_for_bootstrap()` (directly or via placeholder advertisement helpers) guarded by the bootstrap gate and dependency broker, preventing re-entrant pipeline spins while bootstrap phases are still active.
