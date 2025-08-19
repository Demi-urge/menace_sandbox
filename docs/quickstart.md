# Quickstart

This guide walks you through the common Menace workflows using the new `menace` CLI.

## Step-by-step setup

1. **Clone the repository**
   ```bash
   git clone https://example.com/menace_sandbox.git
   cd menace_sandbox
   ```
2. **Install dependencies**
   ```bash
   menace setup
   ```
   The command wraps `scripts/setup_autonomous.sh` and creates a basic `.env` file.
3. **Run the tests** (optional)
   ```bash
   menace test
   ```
4. **Start the self‑optimisation loop**
   ```bash
   menace sandbox run --preset-count 3
   ```
5. **Benchmark workflows**
   ```bash
   menace benchmark
   ```
6. **Deploy the updated bots**
   ```bash
   menace deploy
   ```

### Utility commands

```bash
# Semantic search with caching and full-text fallback
menace retrieve "missing config" --db code

# Apply a patch using the self-coding engine
menace patch bots/example.py --desc "handle edge case"

# Backfill embeddings for a specific database
menace embed --db workflows

# Scaffold a new database module and test
menace new-db demo
```

`retrieve` caches results on disk and reuses them until the underlying databases change. When the vector retriever raises an error, database-specific full-text helpers are consulted as a fallback.

### Recursive module discovery

Passing modules are merged into the sandbox automatically and a simple
workflow is generated for each one. To discover orphaned modules and their
dependencies recursively, run:

```bash
menace sandbox run --discover-orphans --auto-include-isolated \
    --recursive-include --recursive-isolated --clean-orphans
```

`--auto-include-isolated` sets `SANDBOX_AUTO_INCLUDE_ISOLATED=1` and
`--recursive-include` enables `SANDBOX_RECURSIVE_ORPHANS=1`, causing
`sandbox_runner.discover_recursive_orphans` to follow import chains and append
non‑redundant modules to `sandbox_data/module_map.json`. One‑step workflows are
created automatically so the new code participates in subsequent runs. The
[recursive orphan workflow](recursive_orphan_workflow.md) document covers these
environment variables in more detail.

## Troubleshooting

- **Missing packages** – run `menace setup` again to reinstall requirements.
- **Tests fail** – ensure optional tools like `ffmpeg` and Docker are installed.
- **Dashboard not loading** – check that `AUTO_DASHBOARD_PORT` is free.
- **Visual agent 401 errors** – verify `VISUAL_AGENT_TOKEN` matches your `.env`.

## Self‑Optimisation Loop

```mermaid
flowchart TD
    A[Monitor metrics] --> B{ROI drop?}
    B -- yes --> C[SelfImprovementEngine]
    C --> D{ROI improved?}
    D -- yes --> E[Deploy]
    D -- no --> F[SystemEvolutionManager]
    F --> E
    B -- no --> A
    E --> A
```

The loop runs continuously until stopped. Metrics are collected each cycle and the
system decides whether to apply a simple improvement or perform a structural evolution.
