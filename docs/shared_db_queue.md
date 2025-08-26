# Shared Database Queue

## Rationale
SQLite allows only one writer at a time because it uses coarse file locks.\
Multiple Menace instances writing directly to the shared database can block
or fail with `database is locked` errors.  Queueing writes to disk lets each
instance append new records quickly and flush them later without holding the
shared database write lock.

## Queue file format and location
Queued writes are stored as JSON Lines files.  Each line contains a mapping::

    {"table": "<table>", "data": {...}, "source_menace_id": "<id>"}

Files are grouped by menace under the directory defined by `SHARED_QUEUE_DIR`
(defaults to `logs/queue`). The `env_config` module ensures this directory
exists. Each instance appends records to `<menace_id>.jsonl`.
`DBRouter.queue_insert` respects an optional `DB_ROUTER_QUEUE_DIR` which
overrides the base location when routing writes.

## How `sync_shared_db.py` processes queues
The synchroniser scans the queue directory for `*.jsonl` files and handles
each line independently:

1. Parse the JSON payload and insert the record into the target table using
   `db_dedup.insert_if_unique`.
2. On success or duplicate detection the line is removed from the queue.
3. Failures are logged to `queue.failed.jsonl` with error details.

Processed lines are trimmed from the queue and backed up to `queue.log.bak`
for manual rollback.

## Environment variables and daemon options
- `SHARED_QUEUE_DIR` – base directory for queue files (defaults to
  `logs/queue` and created automatically by `env_config`).
- `DB_ROUTER_QUEUE_DIR` – override used by `DBRouter.queue_insert` when
  queuing shared writes.
- `sync_shared_db.py` options:
  - `--queue-dir` – directory to scan for queues (defaults to
    `SHARED_QUEUE_DIR`).
  - `--db-path` – path to the shared SQLite database.
  - `--interval` – polling interval in seconds when running continuously.
  - `--once` – process queues once and exit.
  - `--watch` – watch for filesystem events using `watchdog` if available.
  - `--replay-failed` – requeue entries from `queue.failed.jsonl`.

## Rollback and recovery
- Review `queue.failed.jsonl` for records that could not be inserted.
- Requeue failed entries by running:

  ```bash
  python sync_shared_db.py --queue-dir <dir> --db-path <db> --once --replay-failed
  ```

- `queue.log.bak` retains processed lines.  Copy relevant lines back into the
  original queue file and rerun the synchroniser to roll back a previous
  sync.
