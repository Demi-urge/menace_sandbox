# Database auditing

`audit.log_db_access` captures a tamper-resistant trail of database actions. Each
entry records what happened and which Menace instance performed it so access to
shared resources can be monitored. The legacy
`audit_db_access.log_db_access` wrapper now simply forwards to this unified API
and will be removed in a future release.

## Log format

Events are appended as JSON lines with the following fields:

- `timestamp` – UTC time when the operation occurred
- `action` – `"read"`, `"write"`, etc.
- `table` – table name that was touched
- `rows` – number of rows affected
- `menace_id` – identifier of the Menace instance

Example log line:

```json
{"timestamp": "2024-05-01T12:00:00Z", "action": "write", "table": "telemetry", "rows": 1, "menace_id": "alpha"}
```

## Environment configuration

`DBRouter` writes audit entries to the file specified by the
`DB_ROUTER_AUDIT_LOG` environment variable. If unset, logs default to
`logs/shared_db_access.log`. Callers of `audit.log_db_access` may also override
the destination by passing the `log_path` argument directly.

## Querying the audit table

If `log_db_access` receives a SQLite connection via the `db_conn` parameter the
same event is inserted into the `shared_db_audit` table. Example:

```python
import sqlite3
from audit import log_db_access

conn = sqlite3.connect("audit.db")
log_db_access("write", "telemetry", 1, "alpha", db_conn=conn)
```

To inspect the table directly:

```bash
sqlite3 audit.db "SELECT action, \"table\", rows, menace_id, timestamp FROM shared_db_audit ORDER BY timestamp DESC LIMIT 10;"
```

The stored rows mirror the JSON log and provide a convenient way to query recent
activity.
