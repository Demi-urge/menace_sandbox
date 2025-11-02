# Database Router

The `DBRouter` routes SQLite queries to either a local database specific to a
Menace instance or to a shared database used by all instances.

Buffered writes destined for the shared database are described in
[write_queue.md](write_queue.md); queue layout and synchroniser behaviour
are detailed in [shared_db_queue.md](shared_db_queue.md).

The maintenance script `sync_shared_db.py` performs low-level database
synchronisation and therefore calls `sqlite3.connect` directly. It is explicitly
allowlisted by the repository's `forbid-sqlite3-connect` pre-commit hook.

## Instantiation

Initialise a router via `init_db_router(menace_id)` or directly by creating a
`DBRouter(menace_id, local_db_path, shared_db_path)`. Local databases are stored
at `./menace_<menace_id>_local.db` and shared data lives in `./shared/global.db`
by default. `init_db_router` also stores the router in `GLOBAL_ROUTER` for
modules that rely on a globally accessible instance.

### Context manager usage

`DBRouter` acts as a context manager to automatically close both database
connections when the block exits:

```python
from db_router import DBRouter

with DBRouter("alpha", "local.db", "shared.db") as router:
    conn = router.get_connection("bots")
    conn.execute("SELECT 1")
# exiting the block closes the local and shared connections
```

## Startup initialisation

Application entry points (for example `main.py`, `sandbox_runner.py` or other
service launch scripts) **must** invoke `init_db_router(menace_id)` before any
database operations occur. Initialising the router early ensures
`GLOBAL_ROUTER` is populated so imported modules can retrieve a router without
explicit dependency injection.

Modules that may be executed standalone should include a safe fallback to
initialise the router when `GLOBAL_ROUTER` has not been set:

```python
from db_router import GLOBAL_ROUTER, init_db_router

router = GLOBAL_ROUTER or init_db_router("alpha")
```

This pattern guarantees that a router is always available regardless of how the
module is invoked.

### Typical runtime script

Entry-point scripts should initialise the router before importing modules that
perform database operations so those imports can rely on `GLOBAL_ROUTER` or an
explicitly passed router:

```python
import os
import uuid
from db_router import init_db_router
from dynamic_path_router import resolve_path

MENACE_ID = uuid.uuid4().hex
LOCAL_DB_PATH = os.getenv(
    "MENACE_LOCAL_DB_PATH", str(resolve_path(f"menace_{MENACE_ID}_local.db"))
)
SHARED_DB_PATH = os.getenv(
    "MENACE_SHARED_DB_PATH", str(resolve_path("shared/global.db"))
)
DB_ROUTER = init_db_router(MENACE_ID, LOCAL_DB_PATH, SHARED_DB_PATH)

from some_module import Service

service = Service(router=DB_ROUTER)  # or rely on GLOBAL_ROUTER
```

Passing `DB_ROUTER` to components is optional; modules can also access
`GLOBAL_ROUTER` directly when explicit dependency injection is unnecessary.

## Scope filtering

Queries against shared tables accept a ``scope`` argument to control cross-
instance visibility. ``global`` retrieves entries from other Menace instances,
while ``all`` removes filtering entirely. The parameter accepts:

- ``"local"`` – only records created by the current menace
- ``"global"`` – records from other menace instances
- ``"all"`` – no menace ID filtering

```python
from workflow_summary_db import WorkflowSummaryDB

db = WorkflowSummaryDB()
db.get_summary(1, scope="local")   # -> WorkflowSummary for current menace
db.get_summary(1, scope="global")  # -> records from other menace instances
db.get_summary(1, scope="all")     # -> records from all menaces
```

This ``scope`` parameter replaces the older ``include_cross_instance`` and
``all_instances`` flags.

Lower-level SQL helpers are available for custom queries:

```python
from scope_utils import Scope, build_scope_clause, apply_scope

clause, params = build_scope_clause("bots", Scope.GLOBAL, router.menace_id)
sql = apply_scope("SELECT * FROM bots", clause)
rows = conn.execute(sql, params).fetchall()
```

## Shared vs. local tables

Tables listed in `SHARED_TABLES` are written to the shared database while
`LOCAL_TABLES` entries reside in the local database. Table names must be
explicitly declared; unlisted tables raise a `ValueError` to keep routing
explicit. The default configuration assigns:

**Shared tables**

- `bots`
- `code`
- `discrepancies`
- `enhancements`
- `errors`
- `information`
- `overrides`
- `workflow_summaries`
- `telemetry`

**Local tables**

- `action_audit`
- `action_roi`
- `allocation_weights`
- `allocations`
- `bot_performance`
- `bot_trials`
- `capital_summary`
- `churn`
- `db_roi_metrics`
- `decisions`
- `deployments`
- `efficiency`
- `embedding_stats`
- `evaluation_history`
- `events`
- `evolution_history`
- `evolutions`
- `experiment_history`
- `experiment_tests`
- `failures`
- `feedback`
- `flakiness_history`
- `healing_actions`
- `history`
- `investments`
- `leads`
- `ledger`
- `maintenance`
- `memory`
- `menace_config`
- `messages`
- `metadata`
- `metrics`
- `mirror_logs`
- `models`
- `module_metrics`
- `patch_history`
- `patch_outcomes`
- `patch_provenance`
- `patches`
- `profit`
- `profit_history`
- `resolutions`
- `results`
- `retrieval_cache`
- `retrieval_stats`
- `retriever_kpi`
- `retriever_stats`
- `revenue`
- `risk_summaries`
- `roi`
- `roi_events`
- `roi_history`
- `roi_logs`
- `roi_prediction_events`
- `roi_telemetry`
- `sandbox_metrics`
- `saturation`
- `subs`
- `synergy_history`
- `tasks`
- `test_history`
- `update_history`
- `variants`
- `vector_metrics`
- `weight_override`
- `workflows`

Update these sets to route additional tables:

1. **Import the routing sets.**
2. **Add or remove table names** from the appropriate collection.

```python
from db_router import SHARED_TABLES, LOCAL_TABLES, DENY_TABLES

SHARED_TABLES.add("alerts")      # visible to all Menace instances
LOCAL_TABLES.add("session")      # stored in each instance's local DB
DENY_TABLES.add("legacy_data")   # block unapproved access entirely
```

For larger or frequently changing inventories the table lists may be maintained
centrally.  A configuration file at `config/db_router_tables.json` is loaded
automatically on import and merged with the built‑in defaults.  It should
provide `shared`, `local` and optional `deny` arrays:

```json
{
  "shared": ["enhancements", "bots"],
  "local": ["models", "sandbox_metrics"],
  "deny": []
}
```

### Environment variable overrides

Environment variables offer ad‑hoc customisation without editing the
configuration file. The router inspects the following on import:

- `DB_ROUTER_SHARED_TABLES` – comma‑separated tables forced into the shared
  database.
- `DB_ROUTER_LOCAL_TABLES` – additional tables kept local to each instance.
- `DB_ROUTER_DENY_TABLES` – tables to block entirely. Entries here are removed
  from both routing sets and any access attempts raise a `ValueError`.

These values are merged with the JSON configuration. Alternatively, point
`DB_ROUTER_CONFIG` to a different JSON file following the structure above to
extend or restrict the loaded lists.

#### Deployment examples

```bash
# Development: keep experimental tables isolated locally
export DB_ROUTER_LOCAL_TABLES="session,debug_stats"

# Staging: share alerts and metrics but block finance tables
export DB_ROUTER_SHARED_TABLES="alerts,metrics"
export DB_ROUTER_DENY_TABLES="capital_ledger,finance_logs"
```

## Retrieving connections

Use `DBRouter.get_connection(table_name, operation="read")` to obtain a
`LoggedConnection` for a given table. The router returns the shared connection
for names in `SHARED_TABLES` and the local connection for `LOCAL_TABLES`
entries. Every `execute` call on the returned connection records the row count
via `audit.log_db_access`. The optional `operation` argument is
recorded in audit logs and metrics and can be set to values such as `"read"` or
`"write"`:

```python
from db_router import init_db_router

router = init_db_router("alpha")
conn = router.get_connection("bots", operation="write")
cursor = conn.execute("SELECT * FROM bots")
```

## Replacing direct `sqlite3.connect` calls

Modules should avoid calling `sqlite3.connect()` directly. Instead, initialise
the router and obtain connections via `get_connection` so table placement
remains centralised and access is logged automatically. Services must call
`init_db_router(menace_id)` once at startup before any database operations. A
typical refactor looks like:

```python
# Before
import sqlite3
conn = sqlite3.connect("bots.db")

# After
from db_router import GLOBAL_ROUTER, init_db_router

router = GLOBAL_ROUTER or init_db_router("alpha")
conn = router.get_connection("bots")
```

### Shared table example

```python
from db_router import GLOBAL_ROUTER, init_db_router

router = GLOBAL_ROUTER or init_db_router("alpha")
conn = router.get_connection("bots")  # routes to shared/global.db
```

### Local table example

```python
from db_router import GLOBAL_ROUTER, init_db_router

router = GLOBAL_ROUTER or init_db_router("alpha")
conn = router.get_connection("memory")  # routes to menace_<id>_local.db
```

### Environment or configuration overrides

If a table is not covered by the default routing lists, its placement can be
changed through environment variables or a JSON config:

```bash
export DB_ROUTER_SHARED_TABLES="analytics"   # force into shared database
export DB_ROUTER_LOCAL_TABLES="session"      # keep table local
export DB_ROUTER_CONFIG="/etc/menace/db_router_tables.json"
```

After setting the overrides, calls such as
`router.get_connection("analytics")` will target the shared database while
`router.get_connection("session")` remains local. See *Environment variable
overrides* above for more details.

A pre-commit hook enforces this rule by rejecting commits that introduce
`sqlite3.connect()` outside `db_router.py`.  Obtaining connections through the
router keeps database access centralised.

### Approved direct connections

A pre-commit hook runs `scripts/check_sqlite_connections.py` to block direct
`sqlite3.connect` calls. A handful of utility scripts interact with SQLite
before `DBRouter` can be initialised; these files are allowlisted and may call
`sqlite3.connect` directly:

- `scripts/new_db.py` – scaffolds a minimal database module.
- `scripts/new_db_template.py` – templated database scaffolding with FTS and
  safety hooks.
- `scripts/scaffold_db.py` – legacy scaffolding helper.
- `scripts/new_vector_module.py` – generates a vector database module.
- `db_router.py` – the router itself constructs connections.

Any future exceptions require explicit approval and must be documented here and
added to the pre-commit allowlist.

This pattern ensures shared data resides in the global database while
instance-specific tables use the local database.

## Thread safety

Connections are created with `check_same_thread=False` and guarded by
`threading.Lock` instances so multiple threads can safely interact with the
router.

## Logging and metrics

`DBRouter` emits structured logs for shared table accesses. Because each entry
includes the `menace_id`, table name, operation and timestamp, these logs can be
aggregated across deployments to monitor cross‑Menace behaviour. Local table
accesses are silent unless audit logging is enabled. Configure the verbosity via
the `DB_ROUTER_LOG_LEVEL` environment variable. The output format defaults to
JSON but can be set to key-value pairs by defining `DB_ROUTER_LOG_FORMAT=kv`.
Connections obtained via `get_connection` automatically call
`audit.log_db_access` with the number of rows read or written.

### Access log helper

The lightweight `log_db_access` utility records summary information about each
database operation. Entries are written to the path defined by the
`DB_ROUTER_AUDIT_LOG` environment variable or the `log_path` argument. If
neither is provided, logs default to `logs/shared_db_access.log`. Each line in
the file is a JSONL object capturing the timestamp, action, table and affected
row count:

```json
{"timestamp": "2024-05-14T12:00:00Z", "action": "write", "table": "bots", "rows": 1, "menace_id": "alpha"}
```

Once row counts have been collected, summarise the log with:

```bash
python -m analysis.db_router_log_analysis logs/shared_db_access.log
```

### Audit log

Enable detailed auditing of every table access by following these steps:

1. **Choose a log path.** Set `DB_ROUTER_AUDIT_LOG` to the destination file or
   provide an `"audit_log"` entry in the JSON file referenced by
   `DB_ROUTER_CONFIG`.
2. **Run your application.** Each database access is appended to the log as a
   JSON line containing the menace ID, table name, operation and timestamp. A
   typical entry looks like:

   ```json
   {"menace_id": "alpha", "table_name": "bots", "operation": "write", "timestamp": "2024-05-14T12:00:00Z"}
   ```
3. **Review or rotate logs as needed.** Unset the environment variable to
   disable auditing.

To persist the same entries into the `shared_db_audit` table set the
`DB_ROUTER_AUDIT_TO_DB` environment variable to `1` (or `true`) **before** the
application imports `db_router`. The SQLite mirror is disabled by default so
regular write traffic is not blocked when the audit database is locked.

#### Deployment examples

```bash
# Development: write audit entries to a temporary file
export DB_ROUTER_AUDIT_LOG="/tmp/menace_audit.log"

# Production: store audit logs centrally
export DB_ROUTER_AUDIT_LOG="/var/log/menace/db_router_audit.log"
```

### Analysing audit logs

Use the `analysis/db_router_log_analysis.py` helper to summarise the log:

```bash
python -m analysis.db_router_log_analysis /var/log/menace/db_router_audit.log
```

Example output:

```
menace  table  op    count
alpha   bots   write 12

menace  table  count
alpha   bots   12

Top shared-table writers:
alpha   12
```

Interpretation:

1. The first block groups counts by menace ID, table name and operation.
2. The second block aggregates counts across operations for each menace/table
   pair.
3. The final block highlights menace IDs writing to tables that are shared with
   other instances, helping to spot misuse or hotspots.

Use these summaries to identify unexpected writes to shared tables or excessive
cross-instance activity.

The router also aggregates simple access counts per table, accessible via
`DBRouter.get_access_counts()`, and exposes access counts via the
`table_access_total` telemetry gauge.

## Table‑sharing policy

- **Shared tables**: `SHARED_TABLES` entries persist across different
  `menace_id` instances. Writes to these tables are visible to all routers
  pointing at the same shared database.
- **Local tables**: `LOCAL_TABLES` entries are stored in each instance's local
  database, keeping data isolated between different `menace_id` values. Table
  names outside these sets are rejected.

## Shared DB Access Audit

The router can emit a separate audit stream for operations against shared
tables. These logs allow infrastructure or security teams to trace how instances
interact with globally visible data.

### Configuration options

- **`log_path`** – absolute or relative destination for the audit log. Define it
  with the `DB_ROUTER_AUDIT_LOG` environment variable or by adding an
  `"audit_log"` entry to the JSON file referenced by `DB_ROUTER_CONFIG`.
- The regular router log level and format can be tuned via
  `DB_ROUTER_LOG_LEVEL` and `DB_ROUTER_LOG_FORMAT` (``json`` or ``kv``).

### Log format

Each audit entry is a single JSON object written on its own line. Fields
include the menace identifier, table, operation type and timestamp, along with
an optional `rows` count when emitted through `log_db_access`:

```json
{"timestamp": "2024-05-14T12:00:00Z", "menace_id": "alpha", "table": "bots", "operation": "write", "rows": 1}
```

### Consumption by IGI/Security AI

Because the log is structured and time‑stamped, IGI or Security AI pipelines can
tail the file, ship it to central observability platforms, or ingest it directly
for anomaly detection. Metrics such as writes per menace or unexpected table
touches can be aggregated to flag suspicious cross‑instance behaviour.

## Tests

The automated test suite enforces correct routing behaviour:

- `test_patch_safety_uses_router` confirms `PatchSafety` loads failure records
  through the router and that the `failures` table remains local.
- `test_connect_uses_router` exercises the synergy history helpers to ensure
  connections for `synergy_history` are obtained via `DBRouter`.
- `test_roidb_routes_through_router` verifies the resource allocation
  optimiser's `ROIDB` accesses `roi`, `action_roi` and `allocation_weights`
  tables through the router.
- `test_get_connection_returns_thread_local_connections` spawns multiple
  threads calling `DBRouter.get_connection` to validate thread safety.
- `test_denied_tables_raise` asserts attempts to access tables listed in
  `DENY_TABLES` (for example `capital_ledger`) raise `ValueError`.

Run the relevant tests with `pytest` to validate new services adhere to the
shared/local table policy and denial rules.
