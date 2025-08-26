# Workflow Benchmark CLI

The `workflow_benchmark` module provides a light command line interface for
scoring a single workflow and recording the results in local databases.

## Basic usage

```bash
python workflow_benchmark.py my_module:my_workflow --workflow-id wf_demo --run-id test1
```

The first argument is a dotted path to a callable returning `True` on success.
The CLI executes the function, computes performance metrics via
`CompositeWorkflowScorer` and stores the aggregated ROI metrics in
`roi_results.db`. The resulting JSON output includes the run identifier and
aggregate metrics:

```json
{
  "run_id": "test1",
  "runtime": 0.01,
  "success_rate": 1.0,
  "roi_gain": 0.0,
  ...
}
```

Subsequent runs with different `--run-id` values are appended under the same
`workflow_id`, allowing historical comparisons.
