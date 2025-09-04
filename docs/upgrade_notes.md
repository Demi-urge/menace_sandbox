# Upgrade Notes

## Stripe per-bot keys removed

Legacy per-bot Stripe secret and public keys have been retired. Bots now route
charges and balance lookups through `stripe_billing_router`, which loads a
single set of API keys and maps bots to products and prices via the
hierarchical `ROUTING_MAP`. Remove any `STRIPE_*` entries from existing `.env`
files or vault records and configure routes in `stripe_billing_router.py`
instead.

## vector_service API

The `vector_service` package provides retrieval helpers and HTTP endpoints
(`/search`, `/build-context`, `/track-contributors`,
`/backfill-embeddings`). Ensure imports use `vector_service` helpers and
consult `vector_service_api.py` for usage details.

## Recursive orphan scanning default

Recursive discovery of orphan module dependencies is now enabled by default.
To restore the previous behaviour, pass `--no-recursive-include` or set
`SELF_TEST_RECURSIVE_ORPHANS=0` or `SANDBOX_RECURSIVE_ORPHANS=0`.

## Workflow evolution in autonomous cycles

`ImprovementEngineRegistry.run_all_cycles()` now triggers workflow evolution.
`WorkflowEvolutionManager` benchmarks the baseline with
`CompositeWorkflowScorer`, generates variants via
`WorkflowEvolutionBot.generate_variants(limit)` and promotes those with a higher
ROI. The number of variants is controlled by the `limit` argument while
`ROI_GATING_THRESHOLD` and `ROI_GATING_CONSECUTIVE` gate further evolution once
ROI gains stagnate.
