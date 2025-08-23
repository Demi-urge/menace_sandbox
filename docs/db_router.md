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

**Local tables**

- `events`
- `memory`
- `menace_config`
- `models`
- `patch_history`
- `roi_logs`
- `sandbox_metrics`
- `variants`

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

Environment variables remain available for ad‑hoc overrides.  Set
`DB_ROUTER_SHARED_TABLES`, `DB_ROUTER_LOCAL_TABLES` or `DB_ROUTER_DENY_TABLES`
to comma separated table lists.  Alternatively, point `DB_ROUTER_CONFIG` to a
different JSON file following the structure above to extend or restrict the
loaded lists.

## Retrieving connections

Use `DBRouter.get_connection(table_name)` to obtain an `sqlite3.Connection` for
a given table. The router returns the shared connection for names in
`SHARED_TABLES` and the local connection for `LOCAL_TABLES` entries:

```python
from db_router import init_db_router

router = init_db_router("alpha")
conn = router.get_connection("bots")
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

This pattern ensures shared data resides in the global database while
instance-specific tables use the local database.

## Thread safety

Connections are created with `check_same_thread=False` and guarded by
`threading.Lock` instances so multiple threads can safely interact with the
router.

## Logging and metrics

`DBRouter` emits structured logs for shared table accesses. Each entry contains
the menace ID, table name and an ISO timestamp. Local table accesses are
silent. Configure the verbosity via the `DB_ROUTER_LOG_LEVEL` environment
variable. The output format defaults to JSON but can be set to key-value pairs
by defining `DB_ROUTER_LOG_FORMAT=kv`.

### Audit log

For additional auditing, set the `DB_ROUTER_AUDIT_LOG` environment variable to a
file path. When configured, every shared-table access is appended to this file
as a JSON object containing the menace ID, table name and timestamp. Leave the
variable unset to disable audit logging.

The router also aggregates simple access counts per table, accessible via
`DBRouter.get_access_counts()`, and exposes shared table counts via the
`shared_table_access_total` telemetry gauge.

## Table‑sharing policy

- **Shared tables**: `SHARED_TABLES` entries persist across different
  `menace_id` instances. Writes to these tables are visible to all routers
  pointing at the same shared database.
- **Local tables**: `LOCAL_TABLES` entries are stored in each instance's local
  database, keeping data isolated between different `menace_id` values. Table
  names outside these sets are rejected.
