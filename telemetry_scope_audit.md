# Telemetry Scope Audit

The repository was searched for SQL queries touching the shared `telemetry` table using:

```
rg "FROM telemetry"
```

All matching locations were reviewed. Functions lacking menace scope filtering were updated to use the `db_scope` helpers. The following files required changes:

- `monitoring_dashboard.py` – `error_data` route now accepts scope parameters.
- `menace_gui.py` – `_bot_telemetry` now supports scope.
- `watchdog.py` – telemetry lookups in `_consecutive_failures`, debug proxy and dossier now respect scope.
- `launch_menace_bots.py` – `_TelemProxy` uses scope for recent errors and patterns.
- `sandbox_runner.py` – sandbox telemetry proxy uses scope.
- `adaptive_roi_dataset.py` – `_collect_error_history` accepts scope.
- `knowledge_graph.py` – `ingest_error_db` scopes telemetry ingestion.

Other occurrences already included scope handling (e.g., `error_bot.py`, `telemetry_feedback.py`, `error_logger.py`, `error_ontology_dashboard.py`, `micro_models/error_dataset.py`).

This audit ensures all direct access to the shared `telemetry` table supports menace scoping.
