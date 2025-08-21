# Vectorized Cognition

This guide walks through the vector pipeline used across the project.  It
covers embedding backfills, semantic retrieval, context assembly, patch logging,
ROI feedback and ranking.  Each step builds on the previous one so LLM driven
modules can reason about code history and prioritise improvements.

## Embedding backfill

````python
from vector_service import EmbeddingBackfill

# discover all EmbeddableDBMixin subclasses and populate their vector stores
EmbeddingBackfill().run(session_id="bootstrap")
````

`EmbeddingBackfill` iterates over every database that implements
`EmbeddableDBMixin` and calls `backfill_embeddings`.  Records flagged by the
lightweight license check are skipped and logged.  Backfills can be restricted to
specific databases via the `db`/`dbs` arguments.

## Retrieval

````python
from vector_service import Retriever

retriever = Retriever()
results = retriever.search("upload failed", session_id="abc123")
````

`Retriever` normalises results from `universal_retriever`, applies license and
alignment filtering and returns plain dictionaries.  A fallback to heuristic
full‑text search is used when the semantic lookup yields no confident hits.

## Context building

````python
from vector_service import ContextBuilder

builder = ContextBuilder()
context = builder.build("fix failing tests", session_id="abc123")
````

`ContextBuilder` summarises related bots, workflows, errors and code records,
applying ranking, ROI and safety weights when available.  Win/regret rates and
alignment severity are pulled from `VectorMetricsDB` and normalised so risky
vectors are down‑ranked.  The compact JSON blob is ready for prompt injection or
further processing.

## Patch logging

````python
from vector_service import PatchLogger

logger = PatchLogger()
logger.track_contributors(["bot:7", "workflow:2"], True,
                          patch_id="42", session_id="abc123")
````

`PatchLogger` records which vectors influenced a patch.  Outcomes are persisted
in `PatchHistoryDB`/`VectorMetricsDB` and optional ROI trackers, providing
training data for later ranking and ROI analysis.

## ROI feedback

````python
from roi_tracker import ROITracker

tracker = ROITracker()
tracker.update(0.15, retrieval_metrics=[{"origin_db": "bot", "hit": True}])
````

`ROITracker` maintains ROI and risk‑adjusted ROI (RAROI) histories.  Retrieval
metrics allow it to attribute ROI changes to specific databases which later
influence ranking.

## Ranking

````python
from ranking_model_scheduler import RankingModelScheduler

sched = RankingModelScheduler([], roi_tracker=tracker)
# retrain the model and notify any dependent services
sched.retrain_and_reload()
````

`RankingModelScheduler` periodically rebuilds the retrieval ranking model and
refreshes reliability KPIs.  When an `ROITracker` is supplied, strong ROI signals
can trigger an earlier retrain.

## CognitionLayer

````python
from roi_tracker import ROITracker
from vector_service import CognitionLayer
from ranking_model_scheduler import RankingModelScheduler

tracker = ROITracker()
layer = CognitionLayer(roi_tracker=tracker)

ctx, sid = layer.query("optimise database indexes", session_id="abc123")
# ... apply patch ...
layer.record_patch_outcome(sid, True)

# metrics now feed ranking updates
sched = RankingModelScheduler([], roi_tracker=tracker)
sched.retrain_and_reload()
````

`CognitionLayer` bundles retrieval, context assembly, ranking and patch logging
behind two calls.  Each `query` logs token, rank and hit metrics to
`VectorMetricsDB`; these records train the ranking model the next time
`RankingModelScheduler` retrains.  When `record_patch_outcome` is invoked the
same vectors are forwarded to `PatchLogger` and `ROITracker`, updating success
rates and per‑database ROI.  Subsequent queries will surface higher ROI vectors
earlier as both the ranker and ROI weights adapt to the accumulated metrics.

## Scheduling and retraining

### Embedding scheduler

The scheduler runs `EmbeddingBackfill` on a timer over all
`EmbeddableDBMixin` subclasses. Environment variables control the interval,
batch size, backend and optional database filters:

````bash
export EMBEDDING_SCHEDULER_INTERVAL=3600   # run every hour
export EMBEDDING_SCHEDULER_BATCH_SIZE=50   # override batch size
export EMBEDDING_SCHEDULER_BACKEND=hnsw    # choose vector backend
export EMBEDDING_SCHEDULER_DBS="bots,errors"  # optional filter
python - <<'PY'
from vector_service.embedding_scheduler import start_scheduler_from_env
start_scheduler_from_env()
PY
````

### Ranking model retrainer

Use the CLI to keep the ranking model up to date:

````bash
python -m menace_sandbox.ranking_model_scheduler \
    --vector-db vector_metrics.db \
    --metrics-db metrics.db \
    --model-path retrieval_ranker.json \
    --interval 86400
````

The scheduler can also be embedded in code:

````python
from ranking_model_scheduler import RankingModelScheduler

sched = RankingModelScheduler([service], interval=86400)
sched.start()
````

Stop the scheduler with `sched.stop()` when shutting down.

## End-to-end autonomous run

The full pipeline can be exercised inside the sandbox using
`run_autonomous.py`.  This command wires together embedding backfills,
context assembly, patch logging and ROI feedback for a single cycle:

````bash
python run_autonomous.py --runs 1 --preset-file presets.json
````

The run generates embeddings, builds retrieval context for each task and
records patch outcomes.  Subsequent executions reuse the accumulated metrics
so higher ROI vectors surface earlier in the context.
