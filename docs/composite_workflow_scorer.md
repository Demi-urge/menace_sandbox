# CompositeWorkflowScorer

`CompositeWorkflowScorer` runs workflows under simulated environments and aggregates ROI metrics. It wraps `ROITracker` and persists results in `roi_results.db` via `ROIResultsDB`.

## API overview

```python
from menace_sandbox.composite_workflow_scorer import CompositeWorkflowScorer
scorer = CompositeWorkflowScorer()
result = scorer.evaluate("wf_example")
print(result.success_rate, result.roi_gain)
```

## Metric formulas

- **Workflow synergy** – average pairwise synergy value for participating modules. History loads from `synergy_history_db` or sandbox metrics:

  ```
  workflow_synergy_score = mean(synergy_scores(mod_i, mod_j) for i,j in modules)
  ```

- **Bottleneck index** – worst runtime-to-ROI ratio adjusted by workflow variance:

  ```
  bottleneck_index = max(runtime_m / roi_m) * (1 + workflow_variance(workflow_id))
  ```

- **Patchability** – slope of a linear regression fitted to the last `n` ROI deltas (`n=5`):

  ```
  patchability_score = slope(linear_fit(range(len(hist)), hist))
  ```

## Database schema

`ROIResultsDB` creates a local SQLite table `roi_results`:

| column | type | description |
| --- | --- | --- |
| `workflow_id` | TEXT | identifier of the evaluated workflow |
| `run_id` | TEXT | unique identifier for this evaluation run |
| `runtime` | REAL | total execution time in seconds |
| `success_rate` | REAL | successes divided by total module runs |
| `roi_gain` | REAL | sum of `roi_history` deltas |
| `workflow_synergy_score` | REAL | average synergy score |
| `bottleneck_index` | REAL | worst runtime/ROI ratio adjusted by variance |
| `patchability_score` | REAL | regression slope of ROI history |
| `module_deltas` | TEXT | JSON mapping `module -> {success_rate, roi_delta}` |
| `ts` | TEXT | insertion timestamp |

## CLI example

Run a workflow evaluation and fetch stored results:

```bash
python - <<'PY'
from menace_sandbox.composite_workflow_scorer import CompositeWorkflowScorer
from menace_sandbox.roi_results_db import module_impact_report

scorer = CompositeWorkflowScorer()
res = scorer.evaluate("wf_example")
cur = scorer.results_db.conn.cursor()
run_id = cur.execute(
    "SELECT run_id FROM roi_results WHERE workflow_id=? ORDER BY ts DESC LIMIT 1",
    ("wf_example",),
).fetchone()[0]
print(module_impact_report("wf_example", run_id, scorer.results_db.path))
PY
```

This snippet runs the workflow, stores metrics in `roi_results.db`, and prints a module impact report for the most recent run.
