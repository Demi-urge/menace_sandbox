# Universal Retriever

`UniversalRetriever` performs cross-database similarity search across the
vector-enabled stores used throughout the project. It queries each configured
database, scores the returned candidates and optionally boosts results that are
linked through shared bots.

## API

```python
from menace.universal_retriever import UniversalRetriever

retriever = UniversalRetriever(
    bot_db=BotDB(),
    workflow_db=WorkflowDB(),
    error_db=ErrorDB(),
    enhancement_db=EnhancementDB(),
    information_db=InformationDB(),
)

hits = retriever.retrieve("upload failed", top_k=5, link_multiplier=1.2)
```

`retrieve()` accepts a raw string, a record instance or an explicit vector. It
returns a list of :class:`ResultBundle` objects containing the origin database,
metadata for the matched record, final score and an explanatory reason.

## Scoring formula

For each candidate the retriever gathers available metric signals, normalises
them against the maximum value observed and averages the metrics with the
converted vector similarity component `1 / (1 + distance)`. Metrics currently
considered are:

- **Error frequency** – how often an error has been seen.
- **Enhancement ROI** – uplift from the latest deployment.
- **Workflow usage** – run counts or assigned bots.
- **Bot deployment** – number of dependent workflows or enhancements.

The resulting average produces a `confidence` score in the range ``0..1``.

Candidates that share bot relationships are further adjusted by
`boost_linked_candidates`. This helper inspects the cross reference tables
``bot_workflow``, ``bot_enhancement``, ``error_bot`` and the
``information_*`` tables to detect connectivity between results. Connected
groups have each member's confidence multiplied by ``link_multiplier``
(``final = base * link_multiplier``). The multiplier is applied once per
retrieval and capped internally to keep scores reproducible. The linkage
path and related record identifiers are returned in each result's metadata for
downstream inspection.

## Integration steps

1. Ensure each participating database has embeddings generated via
   `backfill_embeddings`.
2. Instantiate `UniversalRetriever` with any combination of the supported
   databases.
3. Call `retrieve()` with either a text query or an existing record to obtain
   cross-database matches.
4. Adjust ``link_multiplier`` to emphasise connectivity between bots, workflows,
   enhancements, errors and information records when desired.

The helper provides a unified interface over the various vector stores while
maintaining rich scoring explanations for ranking and troubleshooting.

## Feedback weighting

`UniversalRetriever` can bias ranking using aggregated feedback metrics stored
in ``metrics.db``. The :class:`~menace.data_bot.MetricsDB` collects win rates,
regret rates and embedding staleness for each origin database. During
``retrieve()`` the latest KPI values are fetched and converted into a bias
factor:

- **win_rate** – higher historical success increases rank.
- **regret_rate** – high regret reduces confidence.
- **stale_penalty** – older embeddings are down-weighted.

These signals complement similarity and contextual metrics to encourage results
that have performed well in past experiments.

## Tuning parameters

Ranking behaviour can be customised via the :class:`RetrievalWeights` dataclass
passed to ``UniversalRetriever``:

```python
from menace.universal_retriever import UniversalRetriever, RetrievalWeights

weights = RetrievalWeights(similarity=0.6, context=0.4, win=0.2,
                           regret=0.1, stale_cost=0.02)
retriever = UniversalRetriever(bot_db=BotDB(), weights=weights)
```

Adjusting these parameters allows future experiments to explore different
balances between vector similarity, contextual metrics and feedback signals.

