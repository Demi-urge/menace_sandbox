# Upgrade Notes

## vector_service API

The `vector_service` package provides retrieval helpers and HTTP endpoints
(`/search`, `/build-context`, `/track-contributors`,
`/backfill-embeddings`). Ensure imports use `vector_service` helpers and
consult `vector_service_api.py` for usage details.

## Recursive orphan scanning default

Recursive discovery of orphan module dependencies is now enabled by default.
To restore the previous behaviour, pass `--no-recursive-include` or set
`SELF_TEST_RECURSIVE_ORPHANS=0` or `SANDBOX_RECURSIVE_ORPHANS=0`.
