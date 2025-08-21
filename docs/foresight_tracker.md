# ForesightTracker

`ForesightTracker` maintains a rolling window of `max_cycles` cycle metrics for each workflow and derives basic trend curves. The `window` parameter controls this retention and the deprecated alias `N` is still accepted. It acts as a light-weight temporal forecaster that lets the sandbox spot deteriorating behaviour early.

## Required metrics

Each cycle should call `record_cycle_metrics(workflow_id, metrics)` with a mapping of metric names to numbers. The tracker aggregates these values and evaluates slope and volatility across the retained window. When used with `SelfImprovementEngine` the following metrics are expected:

- `roi_delta` – ROI change from the last cycle.
- `raroi_delta` – Risk-adjusted ROI delta.
- `confidence` – most recent confidence score reported by the ROI tracker.
- `resilience` – latest resiliency estimate.
- `scenario_degradation` – optional scenario-based stress indicator.

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

The tracker stores only the recent `max_cycles` cycles per workflow, keeping memory usage predictable.

## Persisting state

`ForesightTracker` can serialise its configuration and recent history for later restoration. The :meth:`to_dict` method returns a JSON-serialisable dictionary containing the tracked `history` together with the current `max_cycles` and `volatility_threshold` settings. The companion :meth:`from_dict` classmethod rebuilds an instance from this data and accepts optional overrides for the configuration values (`N` remains a supported alias for `window`).

```python
tracker = ForesightTracker(window=5, volatility_threshold=2.0)
# ... record some cycles
data = tracker.to_dict()

# Re-create an identical tracker
restored = ForesightTracker.from_dict(data)

# Or restore while changing the configuration
restored_custom = ForesightTracker.from_dict(data, window=3, volatility_threshold=1.5)
```

Only the most recent `max_cycles` entries per workflow are retained when deserialising.
