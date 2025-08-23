# Database Router

The `DBRouter` routes SQLite queries to either a local database specific to a
Menace instance or to a shared database used by all instances.

## Instantiation

Initialise a router via `init_db_router(menace_id)` or directly by creating a
`DBRouter(menace_id, local_db_path, shared_db_path)`. Local databases are stored
at `./menace_<menace_id>_local.db` and shared data lives in `./shared/global.db`
by default. `init_db_router` also stores the router in `GLOBAL_ROUTER` for
modules that rely on a globally accessible instance.

## Startup initialisation

Application entry points (for example `main.py`, `sandbox_runner.py` or other
service launch scripts) should invoke `init_db_router(menace_id)` as early as
possible during startup. This ensures `GLOBAL_ROUTER` is populated so imported
modules can retrieve a router without explicit dependency injection.

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
import uuid
from db_router import init_db_router

DB_ROUTER = init_db_router(uuid.uuid4().hex)

from some_module import Service

service = Service(router=DB_ROUTER)  # or rely on GLOBAL_ROUTER
```

Passing `DB_ROUTER` to components is optional; modules can also access
`GLOBAL_ROUTER` directly when explicit dependency injection is unnecessary.

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

- `events`
- `memory`
- `menace_config`
- `models`
- `patch_history`
- `roi_logs`
- `sandbox_metrics`
- `variants`
- `revenue`
- `subs`
- `churn`
- `leads`
- `profit`
- `history`
- `patches`
- `healing_actions`
- `tasks`
- `metadata`
- `vector_metrics`
- `roi_telemetry`
- `roi_prediction_events`
- `results`
- `resolutions`
- `deployments`
- `bot_trials`
- `update_history`
- `roi_events`

Update these sets to route additional tables:

```python
from db_router import SHARED_TABLES, LOCAL_TABLES

SHARED_TABLES.add("alerts")
LOCAL_TABLES.add("session")
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

Use `DBRouter.get_connection(table_name, operation="read")` to obtain an
`sqlite3.Connection` for a given table. The router returns the shared
connection for names in `SHARED_TABLES` and the local connection for
`LOCAL_TABLES` entries. The optional `operation` argument is recorded in audit
logs and metrics and can be set to values such as `"read"` or `"write"`:

```python
from db_router import init_db_router

router = init_db_router("alpha")
conn = router.get_connection("bots", operation="write")
cursor = conn.execute("SELECT * FROM bots")
```

## Replacing direct `sqlite3.connect` calls

Modules should avoid calling `sqlite3.connect()` directly. Instead, retrieve a
connection through the router so table placement remains centralised:

```python
# Legacy code
conn = sqlite3.connect("bots.db")

# Updated approach
from db_router import GLOBAL_ROUTER, init_db_router

router = GLOBAL_ROUTER or init_db_router("alpha")
conn = router.get_connection("bots")
```

A pre-commit hook enforces this rule by rejecting commits that introduce
`sqlite3.connect()` outside `db_router.py`.  Obtaining connections through the
router keeps database access centralised.

This pattern ensures shared data resides in the global database while
instance-specific tables use the local database.

## Thread safety

Connections are created with `check_same_thread=False` and guarded by
`threading.Lock` instances so multiple threads can safely interact with the
router.

## Logging and metrics

`DBRouter` emits structured logs for shared table accesses. Each entry contains
the menace ID, table name, operation type and an ISO timestamp. Local table
accesses are silent unless audit logging is enabled. Configure the verbosity via
the `DB_ROUTER_LOG_LEVEL` environment variable. The output format defaults to
JSON but can be set to key-value pairs by defining `DB_ROUTER_LOG_FORMAT=kv`.

### Audit log

The router can emit an audit trail of every table access. Configure a file path
via the `DB_ROUTER_AUDIT_LOG` environment variable or provide an `"audit_log"`
entry in the JSON configuration referenced by `DB_ROUTER_CONFIG`. When
enabled, each access is appended to that file as a JSON object containing the
menace ID, table name, operation and timestamp. Leave the configuration unset to
disable auditing.

#### Deployment examples

```bash
# Development: write audit entries to a temporary file
export DB_ROUTER_AUDIT_LOG="/tmp/menace_audit.log"

# Production: store audit logs centrally
export DB_ROUTER_AUDIT_LOG="/var/log/menace/db_router_audit.log"
```

### Analysing audit logs

An `analysis/db_router_log_analysis.py` script is included to aggregate these
logs for cross-instance auditing.  Run the tool with the path to the audit log
file:

```bash
python -m analysis.db_router_log_analysis path/to/audit.log
```

The script outputs three sections:

1. Per-operation counts grouped by menace ID, table name and operation.
2. Aggregated counts by menace ID and table across all operations.
3. A summary of the top menace IDs writing to tables that are shared with other
   menaces, helping to highlight potential misuse or hotspots.

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
