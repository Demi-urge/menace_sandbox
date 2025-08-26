# SQLite Write Queue

For queue layout, environment variables and recovery procedures see
[shared_db_queue.md](shared_db_queue.md).

## Why SQLite needs buffering
SQLite uses file-level locks and permits only one writer at a time. Multiple Menace
instances writing directly to the shared database can contend for locks or block
each other. Buffering writes to disk lets processes append new records quickly and
replay them later without holding a database write lock.

## Queue file format
Each shared table receives a queue file named `<table>_queue.jsonl` under the
configured queue directory. Every line is a JSON object containing:

- `table` – the destination table name
- `data` or `values` – column mapping for the row
- `source_menace_id` – identifier of the Menace instance that generated the entry
- `hash_fields` or `content_hash` – fields used to deduplicate queued rows

## How `sync_shared_db.py` processes queues
`sync_shared_db.py` scans the queue directory for pending files produced by
`db_write_queue.queue_insert` and attempts to insert each record into the shared
database. Successful inserts, or rows that already exist, are removed from the
queue. Malformed entries are moved to `<table>_queue.error.jsonl` while failures
are appended to `<table>_queue.failed.jsonl` with error details. The script may
run once or loop continuously based on its polling interval.

## Configuration variables
- `MENACE_QUEUE_DIR` – base directory for queue files. Defaults to `./queue` but
  can be overridden via the environment variable of the same name.
- `--interval` – polling interval in seconds for `sync_shared_db.py` when running
  continuously.

## Monitoring and cleanup
1. Inspect the queue directory for `<table>_queue.failed.jsonl` files to identify
   records that could not be written to the shared database.
2. Review their contents using tools like `cat`, `jq` or `tail -f` to view error
   details.
3. Periodically run `python queue_cleanup.py --queue-dir <dir> --days <retention>`
   to prune stale `.failed.jsonl` and temporary queue files.

## Manual rollback

Processed lines removed from a queue file are appended to `queue.log.bak` in the
same directory. To roll back a previous sync, copy the relevant lines from this
backup back into the original queue file and rerun the synchroniser.

Failed inserts accumulate in `queue.failed.jsonl`. Requeue them automatically
with:

```bash
python sync_shared_db.py --queue-dir <dir> --db-path <db> --once --replay-failed
```

This command restores entries from the failed log into their queue files. The
original failed log is preserved as `queue.failed.jsonl.bak` for manual
inspection.
