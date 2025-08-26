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

- **Workflow synergy** – ratio of summed individual module ROI gains to the combined ROI gain:

  ```
  workflow_synergy_score = sum(roi_gain_m for m in modules) / total_roi_gain
  ```

- **Bottleneck index** – slowest module runtime divided by total workflow runtime:

  ```
  bottleneck_index = max(runtime_m) / sum(runtime_m for m in modules)
  ```

- **Patchability** – derivative of the ROI trend over recent runs adjusted by historical volatility:

  ```
  patchability_score = slope(recent_roi_history) / (1 + std(all_roi_history))
  ```

## Database schema

`ROIResultsDB` creates a local SQLite table `workflow_results`:

| column | type | description |
| --- | --- | --- |
| `workflow_id` | TEXT | identifier of the evaluated workflow |
| `run_id` | TEXT | unique identifier for this evaluation run |
| `timestamp` | TEXT | insertion timestamp |
| `runtime` | REAL | total execution time in seconds |
| `success_rate` | REAL | successes divided by total module runs |
| `roi_gain` | REAL | sum of `roi_history` deltas |
| `workflow_synergy_score` | REAL | summed module ROI gains divided by combined gain |
| `bottleneck_index` | REAL | max module runtime divided by total runtime |
| `patchability_score` | REAL | ROI slope adjusted by volatility |
| `module_deltas` | TEXT | JSON mapping `module -> {success_rate, roi_delta}` |

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
    "SELECT run_id FROM workflow_results WHERE workflow_id=? ORDER BY timestamp DESC LIMIT 1",
    ("wf_example",),
).fetchone()[0]
print(module_impact_report("wf_example", run_id, scorer.results_db.path))
PY
```

This snippet runs the workflow, stores metrics in `roi_results.db`, and prints a module impact report for the most recent run.
