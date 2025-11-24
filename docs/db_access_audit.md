# DB access audit logging

`log_db_access` captures a compact audit trail for each database operation. It
records the timestamp, action, table name, affected row count and menace ID so
instances can be monitored across deployments.

## Log storage

Entries are written as JSON lines to `logs/shared_db_access.log` by default.
Override the destination by passing a `log_path` argument to
`audit.log_db_access` or by setting the `DB_ROUTER_AUDIT_LOG` environment
variable when using `DBRouter`.

## Configuration

The helper writes only to the file by default. Pass `log_to_db=True` to also
insert the entry into the local SQLite `db_access_audit` table via `DBRouter`.
The `log_to_db` flag effectively switches between file-only and combined
file/database logging.

Set `DB_AUDIT_DISABLE_FILE=1` to skip writing the JSONL log entirely. This is
useful in environments where the default log location is read-only or slow to
access; database logging remains available when `log_to_db=True`.

## Analysis

Collected logs can be aggregated for metrics or anomaly detection. The
`analysis.db_router_log_analysis` module summarises operations per menace and
table and highlights tables written by multiple menace IDs:

```bash
python -m analysis.db_router_log_analysis logs/shared_db_access.log
```

