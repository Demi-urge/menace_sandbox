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
- `hash` – SHA256 of the previous hash plus this entry's contents

Example log line:

```json
{
  "timestamp": "2024-05-01T12:00:00Z",
  "action": "write",
  "table": "telemetry",
  "rows": 1,
  "menace_id": "alpha",
  "hash": "e3b0c44298fc1c149afbf4c8996fb924..."
}
```

## Environment configuration

`DBRouter` writes audit entries to the file specified by the
`DB_ROUTER_AUDIT_LOG` environment variable. If unset, logs default to
`logs/shared_db_access.log`. Callers of `audit.log_db_access` may also override
the destination by passing the `log_path` argument directly.

For each log file a companion state file `<log_path>.state` stores the last
emitted hash so the chain can resume after restarts.

## Log rotation

`audit.log_db_access` uses Python's `RotatingFileHandler` to automatically
rotate the JSONL log. Rotation thresholds can be configured via environment
variables:

- `DB_AUDIT_LOG_MAX_BYTES` – maximum size in bytes before a new file is created
  (defaults to 10MB).
- `DB_AUDIT_LOG_BACKUPS` – number of rotated log files to retain (defaults to 5).

Both settings are optional; sensible defaults are applied when the variables are
unset or invalid. Rotation occurs with the same file-locking semantics as
regular writes, so multiple processes can safely append to the log without
interleaving.

## Verifying the log

Use `tools/verify_audit_log.py` to recompute and validate the hash chain:

```bash
python tools/verify_audit_log.py logs/shared_db_access.log
```

The script checks each entry's hash and compares the final value with the
associated `.state` file.

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
