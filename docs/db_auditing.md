# Database auditing

`audit.log_db_access` captures a tamper-resistant trail of database actions. Each
entry records what happened and which Menace instance performed it so access to
shared resources can be monitored.

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

Two environment variables control where audit records are stored:

- `DB_ROUTER_AUDIT_LOG` – when set, `DBRouter` writes audit entries to this
  path. If unset, logs default to `logs/shared_db_access.log`.
- `DB_ACCESS_LOG_PATH` – legacy modules that call `audit_db_access.log_db_access`
  honour this variable for their log file location.

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
