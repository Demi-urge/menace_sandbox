# ForesightTracker

`ForesightTracker` maintains a rolling window of `max_cycles` cycle metrics for each workflow and derives basic trend curves. The `max_cycles` parameter controls this retention and the deprecated aliases `window` and `N` are still accepted. It acts as a light-weight temporal forecaster that lets the sandbox spot deteriorating behaviour early.

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

tracker = ForesightTracker(max_cycles=5, volatility_threshold=2.0)
engine = SelfImprovementEngine(foresight_tracker=tracker)

# within the improvement loop the engine will record metrics automatically
engine.run_once()

# later you can inspect whether a workflow remains stable
if tracker.is_stable("workflow-1"):
    print("workflow-1 metrics trending upward")
```

The tracker stores only the recent `max_cycles` cycles per workflow, keeping memory usage predictable.

## Cold starts and template curves

Cold‑start mode keeps brand‑new workflows from triggering false volatility
signals by seeding them with a baseline ROI curve until enough real data is
available. Workflows with fewer than three recorded cycles are treated as cold
starts. The `is_cold_start(workflow_id)` helper detects this warm‑up period and
also returns `True` when no ROI metric has been captured yet. During this phase
the tracker blends real ROI measurements with synthetic template values.

The templates live at `configs/foresight_templates.yaml` in the repository's
`configs` directory. The file has two top‑level sections:

- `profiles` – maps workflow identifiers to template names.
- `templates` – lists the ROI sequences used for bootstrapping.

For example:

```yaml
profiles:
  scraper_bot: slow_riser
  trading_bot: early_volatile
templates:
  slow_riser:      [0.05, 0.1, 0.2, 0.35, 0.5]
  early_volatile:  [0.4, -0.15, 0.5, -0.1, 0.45]
```

Here `scraper_bot` uses the `slow_riser` profile while `trading_bot` follows
`early_volatile`. Additional profiles can be configured by editing
`configs/foresight_templates.yaml`.

When capturing metrics, the first five logged cycles blend real observations
with the template curve using:

```python
alpha = min(logged_cycles / 5.0, 1.0)
effective_roi = alpha * real_roi + (1.0 - alpha) * template_val
```

`alpha` grows linearly with the number of logged cycles (0.2, 0.4, …, 1.0 after
five cycles), so the template influence fades out as history accumulates. This
weighted `effective_roi` is stored as both `roi_delta` and `raroi_delta` until
enough history accrues, after which real ROI values take over completely.

```python
from menace_sandbox.foresight_tracker import ForesightTracker
from menace_sandbox.roi_tracker import ROITracker

tracker = ForesightTracker()
roi_tracker = ROITracker()

# Early cycles blend real ROI with the `early_volatile` template.
if tracker.is_cold_start("trading_bot"):
    tracker.capture_from_roi(roi_tracker, "trading_bot", "early_volatile")
```

## Persisting state

`ForesightTracker` can serialise its configuration and recent history for later restoration. The :meth:`to_dict` method returns a JSON-serialisable dictionary containing the tracked `history` together with the current `max_cycles` and `volatility_threshold` settings. The companion :meth:`from_dict` classmethod rebuilds an instance from this data and accepts optional overrides for the configuration values (`window` and `N` remain supported aliases for `max_cycles`).

```python
tracker = ForesightTracker(max_cycles=5, volatility_threshold=2.0)
# ... record some cycles
data = tracker.to_dict()

# Re-create an identical tracker
restored = ForesightTracker.from_dict(data)

# Or restore while changing the configuration
restored_custom = ForesightTracker.from_dict(data, max_cycles=3, volatility_threshold=1.5)
```

Only the most recent `max_cycles` entries per workflow are retained when deserialising.

Within the sandbox runner these dictionaries are written to `foresight_history.json`
after each cycle and loaded on the next run via `from_dict`, ensuring metric history
survives restarts of the self‑improvement process.
