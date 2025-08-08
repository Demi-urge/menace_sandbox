# Recursive orphan workflow

The sandbox automatically discovers modules that have no inbound references and
recursively follows their imports to ensure helper files are evaluated. This
**recursive orphan workflow** keeps detached code paths from drifting out of
sync with the rest of the repository.

## Environment variables

These environment variables control the behaviour:

| Variable | Default | Purpose |
|---------|---------|---------|
| `SANDBOX_RECURSIVE_ORPHANS` | `1` | Recurse through orphan dependencies when scanning. |
| `SANDBOX_AUTO_INCLUDE_ISOLATED` | `1` | Automatically include isolated modules returned by `discover_isolated_modules`. |
| `SANDBOX_RECURSIVE_ISOLATED` | `1` | Recursively resolve local imports when auto‑including isolated modules. |
| `SANDBOX_TEST_REDUNDANT` | `1` | Execute modules marked as redundant before deciding whether to integrate them. |

All flags are enabled by default via `SandboxSettings` so operators receive the
full orphan scan without additional configuration. Set any variable to `0` to
disable that feature.

### Examples

Enable the full recursive inclusion flow:

```bash
export SANDBOX_AUTO_INCLUDE_ISOLATED=1
export SANDBOX_RECURSIVE_ORPHANS=1
export SANDBOX_RECURSIVE_ISOLATED=1
export SANDBOX_TEST_REDUNDANT=1
```

Disable recursion but still include isolated modules:

```bash
export SANDBOX_RECURSIVE_ORPHANS=0
export SANDBOX_RECURSIVE_ISOLATED=0
export SANDBOX_AUTO_INCLUDE_ISOLATED=1
```

These variables are also surfaced as CLI flags:

- `--recursive-orphans` / `--no-recursive-orphans`
- `--recursive-isolated` / `--no-recursive-isolated`
- `--auto-include-isolated`
- `--include-redundant` / `--no-include-redundant`

Refer to [`sandbox_settings.py`](../sandbox_settings.py) for more details on the
default configuration.

## Interpreting `sandbox_data/orphan_modules.json`

`include_orphan_modules` loads candidate paths from
`sandbox_data/orphan_modules.json` and queues them for validation. Each key is a
module path relative to the repository root and the value provides metadata such
as the discovery source:

```json
{
  "detached/util.py": {"source": "static-analysis"},
  "legacy/task.py":   {"source": "runtime"}
}
```

Passing entries are merged into the module map via
`auto_include_modules`. Review or edit the JSON file to influence which modules
are reconsidered on the next run.

See [`include_orphan_modules`](../sandbox_runner/cycle.py) and
[`auto_include_modules`](../sandbox_runner/environment.py) for full integration
details.
