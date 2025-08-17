# Upgrade Notes

## semantic_service renamed to vector_service

The `semantic_service` package has been replaced by `vector_service`. Update
imports to use `vector_service` helpers and consult `vector_service_api.py` for
HTTP endpoints (`/search`, `/build-context`, `/track-contributors`,
`/backfill-embeddings`).

## Recursive orphan scanning default

Recursive discovery of orphan module dependencies is now enabled by default.
To restore the previous behaviour, pass `--no-recursive-include` or set
`SELF_TEST_RECURSIVE_ORPHANS=0` or `SANDBOX_RECURSIVE_ORPHANS=0`.
