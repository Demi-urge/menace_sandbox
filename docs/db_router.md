# Database Router

The `DBRouter` routes SQLite queries to either a local database specific to a
Menace instance or to a shared database used by all instances.

## Instantiation

Initialise a router via `init_db_router(menace_id)` or directly by creating a
`DBRouter(menace_id, local_db_path, shared_db_path)`. Local databases are stored
at `./menace_<menace_id>_local.db` and shared data lives in `./shared/global.db`
by default. `init_db_router` also stores the router in `GLOBAL_ROUTER` for
modules that rely on a globally accessible instance.

## Shared vs. local tables

Tables listed in `SHARED_TABLES` are written to the shared database while
`LOCAL_TABLES` entries reside in the local database. Table names must be
explicitly declared; unlisted tables raise a `ValueError` to keep routing
explicit. The default configuration assigns:

**Shared tables**

- `enhancements`
- `bots`
- `errors`
- `code`
- `discrepancies`
- `workflow_summaries`

**Local tables**

- `models`
- `patch_history`
- `variants`
- `memory`
- `events`
- `sandbox_metrics`
- `roi_logs`
- `menace_config`

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

`DBRouter` logs every invocation of `get_connection()` including whether the
table was shared or local, the menace ID and the current thread name. These
messages aid observability and can be silenced by increasing the logger level or
setting `log_level=None`.

The router also aggregates simple access counts per table, accessible via
`DBRouter.get_access_counts()`, which can be used for lightweight monitoring.

## Table‑sharing policy

- **Shared tables**: `SHARED_TABLES` entries persist across different
  `menace_id` instances. Writes to these tables are visible to all routers
  pointing at the same shared database.
- **Local tables**: `LOCAL_TABLES` entries are stored in each instance's local
  database, keeping data isolated between different `menace_id` values. Table
  names outside these sets are rejected.
