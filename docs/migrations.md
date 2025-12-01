# Database Migrations

Menace uses Alembic to manage schema changes. Run migrations with:

```bash
alembic upgrade head
```

The `DATABASE_URL` environment variable controls which database is upgraded.

When updating an existing installation, run the migrations after pulling the
latest code:

```bash
alembic upgrade head
```

This will create any missing tables such as those defined in
`menace.databases.MenaceDB`. As of version 0.2 a `memory_embeddings` table is
created to store vector representations of memory entries for semantic search.
Databases that inherit from `EmbeddableDBMixin` (bots, workflows, errors,
enhancements and research info) now also maintain a shared `embeddings` table
for vector search.  After migrating, run `backfill_embeddings()` on each
database to populate vectors.

### Content hash column

Version 0.3 adds a `content_hash` column to the `bots`, `workflows`,
`enhancements`, and `errors` tables for deduplication. New installations create
the column automatically. For existing databases run the migration or add the
column manually:

```sql
ALTER TABLE bots ADD COLUMN content_hash TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_bots_content_hash ON bots(content_hash);
ALTER TABLE workflows ADD COLUMN content_hash TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_workflows_content_hash ON workflows(content_hash);
ALTER TABLE enhancements ADD COLUMN content_hash TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_enhancements_content_hash ON enhancements(content_hash);
ALTER TABLE errors ADD COLUMN content_hash TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_errors_content_hash ON errors(content_hash);
```

Re-run `alembic upgrade head` after adding the columns so Alembic can create any
missing indices.

## Bootstrap readiness API migration

Modules that previously polled ad-hoc booleans (for example `MENACE_BOOTSTRAP_READY`) should migrate to the staged readiness
API exposed by `EnvironmentBootstrapper` and the bootstrap heartbeat:

- Query `EnvironmentBootstrapper.readiness_state()` or `bootstrap_timeout_policy.read_bootstrap_heartbeat()` to obtain the
  structured readiness gates (`online`, `online_partial`, `full_ready`, and per-stage gates such as `vector_seeding` or
  `db_index_load`). Health handlers and supervisors should surface these snapshots directly instead of synthesising their own
  flags.
- When a module must wait for bootstrap, block on `bootstrap_gate.wait_for_bootstrap_gate()` or
  `bootstrap_manager.wait_until_ready(check=...)` rather than spinning on custom timers. The gate honours the shared queue and
  emits backoff telemetry so late arrivals do not starve existing bootstraps.
- If a module owns a background task that should influence readiness (for example, warm caches or hydrate retrievers), queue it
  through `EnvironmentBootstrapper` so `_track_background_task` can emit `bootstrap-readiness` events and update the
  `background_tasks` gate. Avoid emitting bespoke booleans; the staged readiness map already distinguishes optional and
  critical phases.

These steps keep readiness reporting consistent across entrypoints and ensure new modules appear in the central readiness
snapshots used by watchdogs and dashboards.
