# Upgrade Notes

## Recursive orphan scanning default

Recursive discovery of orphan module dependencies is now enabled by default.
To restore the previous behaviour, pass `--no-recursive-include` or set
`SELF_TEST_RECURSIVE_ORPHANS=0` or `SANDBOX_RECURSIVE_ORPHANS=0`.
