# ForesightTracker

`ForesightTracker` maintains a rolling window of cycle metrics for each workflow and derives basic trend curves.  It acts as a light‑weight temporal forecaster that lets the sandbox spot deteriorating behaviour early.

## Required metrics

Each cycle should call `record_cycle_metrics(workflow_id, metrics)` with a mapping of metric names to numbers.  The tracker aggregates these values and evaluates slope and volatility across the retained window.  When used with `SelfImprovementEngine` the following metrics are expected:

- `roi_delta` – ROI change from the last cycle.
- `raroi_delta` – Risk‑adjusted ROI delta.
- `confidence` – most recent confidence score reported by the ROI tracker.
- `resilience` – latest resiliency estimate.
- `scenario_degradation` – optional scenario‑based stress indicator.

Any additional numeric metrics can be supplied; they are included in the stability calculations.

## Example with `SelfImprovementEngine`

```python
from menace_sandbox.foresight_tracker import ForesightTracker
from menace_sandbox.self_improvement_engine import SelfImprovementEngine

tracker = ForesightTracker(window=5, volatility_threshold=2.0)
engine = SelfImprovementEngine(foresight_tracker=tracker)

# within the improvement loop the engine will record metrics automatically
engine.run_once()

# later you can inspect whether a workflow remains stable
if tracker.is_stable("workflow-1"):
    print("workflow-1 metrics trending upward")
```

The tracker stores only the recent `window` cycles per workflow, keeping memory usage predictable.
