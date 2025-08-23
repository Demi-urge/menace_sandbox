# Database Router

The `DBRouter` routes SQLite queries to either a local database specific to a
Menace instance or to a shared database used by all instances.

## Configuration

Initialise a router via `init_db_router(menace_id)` or directly by creating a
`DBRouter(menace_id, local_db_path, shared_db_path)`.  Local databases are stored
at `./menace_<menace_id>_local.db` and shared data lives in `./shared/global.db`
by default.

Tables listed in `SHARED_TABLES` are written to the shared database while
`LOCAL_TABLES` and any unlisted tables reside in the local database.

## Thread safety

Connections are created with `check_same_thread=False` and guarded by
`threading.Lock` instances so multiple threads can safely interact with the
router.

## Table‑sharing policy

- **Shared tables**: `SHARED_TABLES` entries persist across different
  `menace_id` instances. Writes to these tables are visible to all routers
  pointing at the same shared database.
- **Local tables**: `LOCAL_TABLES` (and tables not listed in `SHARED_TABLES`) are
  stored in each instance's local database, keeping data isolated between
  different `menace_id` values.
