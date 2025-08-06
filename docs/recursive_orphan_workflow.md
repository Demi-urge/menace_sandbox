# Recursive orphan workflow

The sandbox automatically discovers modules that have no inbound references and
recursively follows their imports to ensure helper files are evaluated. This
**recursive orphan workflow** keeps detached code paths from drifting out of
sync with the rest of the repository.

## Environment variables

Two environment variables control the behaviour:

| Variable | Default | Purpose |
|---------|---------|---------|
| `SANDBOX_RECURSIVE_ORPHANS` | `1` | Recurse through orphan dependencies when scanning. |
| `SANDBOX_AUTO_INCLUDE_ISOLATED` | `1` | Automatically include isolated modules returned by `discover_isolated_modules`. |

Both flags are enabled by default via `SandboxSettings` so operators receive the
full orphan scan without additional configuration. Set either variable to `0` to
disable the feature.

### Example

```bash
# disable recursion but still include isolated modules
export SANDBOX_RECURSIVE_ORPHANS=0
export SANDBOX_AUTO_INCLUDE_ISOLATED=1
```

These variables are also surfaced as CLI flags:

- `--recursive-orphans` / `--no-recursive-orphans`
- `--auto-include-isolated`

Refer to [`sandbox_settings.py`](../sandbox_settings.py) for more details on the
default configuration.
