# Borderline Bucket

The **borderline bucket** is a lightweight JSONL-backed store for workflows
whose risk-adjusted ROI (RAROI) or confidence scores fall near the minimum
acceptable threshold.  Instead of immediately promoting or terminating these
workflows, the bucket preserves their recent history so that a limited
**micro-pilot** can gather more evidence.

## Data fields

Each line in `borderline_bucket.jsonl` stores a JSON object with:

| field       | type      | description |
|-------------|-----------|-------------|
| `workflow_id` | string | Unique identifier for the workflow. |
| `raroi`     | list of numbers | Recorded RAROI values from evaluations and micro‑pilots. |
| `confidence` | number | Last confidence score in `[0,1]`. |
| `status`   | string | One of `candidate`, `promoted` or `terminated`. |

## Promotion and termination

`ROITracker` adds a workflow to the bucket when its RAROI falls below the
configured `raroi_borderline_threshold` or when the tracker’s confidence score
drops under `confidence_threshold`.  Calling `borderline_bucket.process()` runs
a small micro‑pilot for each pending candidate and records the new RAROI and
confidence.  Candidates exceeding the supplied thresholds are **promoted**
while the rest are **terminated** and removed from further consideration.

## Micro‑pilot mechanics

Micro‑pilots supply a quick signal before committing to large scale changes.
During a micro‑pilot, an evaluator function produces a temporary RAROI and
optionally a confidence score for the candidate workflow.  The bucket records
these via `record_result()` and updates the status using `promote()` or
`terminate()`.  This allows risky or uncertain workflows to be tested in
isolation before wider deployment.

See [docs/roi_tracker.md](roi_tracker.md) for integration details.
