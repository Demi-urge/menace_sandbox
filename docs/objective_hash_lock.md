# Objective hash-lock operator workflow

`objective_inventory.OBJECTIVE_ADJACENT_HASH_PATHS` is the canonical objective hash target list.
`ObjectiveGuard` defaults and `config/objective_hash_lock.json` are expected to stay in sync with that list.

## Intentional objective-surface rotation

When an approved objective-adjacent change lands, rotate the manifest intentionally:

```bash
python tools/objective_guard_manifest_cli.py \
  --operator <your_name> \
  --reason "<approved change summary>" \
  refresh
```

This regenerates `config/objective_hash_lock.json` from the configured hash specs and appends an audit entry.

## Drift detection

Startup and cycle checks fail closed when the manifest file set differs from configured objective hash specs.
A `manifest_file_set_mismatch` means one of:

- objective hash specs changed without rotating the manifest, or
- manifest was edited/rotated from a different file set.

Resolution: verify the intended hash target list in `objective_inventory.py`, then run the refresh command above with operator context.
