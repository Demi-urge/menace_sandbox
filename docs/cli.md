# Menace CLI

The `menace` command groups common workflows for development and operations.  Each
subcommand below includes a short description and an example invocation.

## setup
Install dependencies and bootstrap the environment.
```bash
menace setup
```

## test
Run the test suite, forwarding any extra arguments to `pytest`.
```bash
menace test -k retrieval
```

## improve
Run a single self‑improvement cycle.
```bash
menace improve --runs 1
```

## benchmark
Benchmark registered workflows.
```bash
menace benchmark
```

## deploy
Install Menace as a service.
```bash
menace deploy
```

## sandbox run
Utilities for the autonomous sandbox.
```bash
menace sandbox run --extra-flag
```

## patch
Apply a quick fix to a module.
```bash
menace patch my_module --desc "Fix bug" --context '{"priority": "high"}'
```

## patches
Tools for inspecting patch history.
```bash
menace patches list --limit 10
menace patches ancestry 42
menace patches search --vector my_vector
```

## cache
Manage the SQLite retrieval cache. Entries expire after one hour (TTL = 3600s).
```bash
menace cache show
menace cache stats
menace cache clear
```

## retrieve
Semantic code retrieval with automatic fallback to full‑text search when no
vector results are available. Results are cached for one hour unless `--no-cache`
is used. Use `--rebuild-cache` to refresh stale entries.
```bash
menace retrieve "rate limiter" --db errors --top-k 5
```

## embed
Backfill vector embeddings.
```bash
menace embed --db errors --batch-size 100
```

### Automatic vs. manual runs

Most databases trigger embedding backfills automatically when records are
inserted or updated, so the `embed` subcommand is rarely needed during normal
operation. Use `menace embed --db <name>` to manually regenerate embeddings when
automatic hooks are disabled or a bulk rebuild is required.

## new-db
Scaffold a new database module. Hooks allow custom registration logic.
```bash
menace new-db feature_usage
```

## new-vector
Scaffold a new `vector_service` module with optional hooks for registering the
router or creating an Alembic migration.
```bash
menace new-vector sentiment --register-router --create-migration
```

See the [README](../README.md) for an overview of the project.
