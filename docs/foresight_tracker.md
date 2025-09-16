# ForesightTracker

`ForesightTracker` maintains a rolling window of `max_cycles` cycle metrics for each workflow and derives basic trend curves. The `max_cycles` parameter controls this retention and the deprecated aliases `window` and `N` are still accepted. It acts as a light-weight temporal forecaster that lets the sandbox spot deteriorating behaviour early.  In addition to stability scores the tracker can project future ROI with `predict_roi_collapse`, labelling trajectories as **Stable**, **Slow decay**, **Volatile** or **Immediate collapse risk** and flagging *brittle* workflows that crash after small entropy shifts.

``record_cycle_metrics`` now exposes a ``compute_stability`` flag. When set
to ``True`` the tracker evaluates the current window immediately and appends a
``stability`` value to the recorded metrics.

## Required metrics

Each cycle should call `record_cycle_metrics(workflow_id, metrics)` with a mapping of metric names to numbers.  When `compute_stability=True` the call also stores the current window stability under the `"stability"` key for that cycle.  The tracker aggregates these values and evaluates slope and volatility across the retained window. When used with `SelfImprovementEngine` the following metrics are expected:

- `roi_delta` – ROI change from the last cycle.
- `raroi_delta` – Risk-adjusted ROI delta.
- `confidence` – most recent confidence score reported by the ROI tracker.
- `resilience` – latest resiliency estimate.
- `scenario_degradation` – optional scenario-based stress indicator.

Any additional numeric metrics can be supplied; they are included in the stability calculations.

## Example with `SelfImprovementEngine`

```python
from menace_sandbox.foresight_tracker import ForesightTracker
from menace_sandbox.self_improvement import SelfImprovementEngine
from context_builder_util import create_context_builder

tracker = ForesightTracker(max_cycles=5, volatility_threshold=2.0)
builder = create_context_builder()
engine = SelfImprovementEngine(context_builder=builder, foresight_tracker=tracker)

# within the improvement loop the engine will record metrics automatically
engine.run_once()

# later you can inspect whether a workflow remains stable
if tracker.is_stable("workflow-1"):
    print("workflow-1 metrics trending upward")
```

The tracker stores only the recent `max_cycles` cycles per workflow, keeping memory usage predictable.

## Logging temporal simulations

Temporal simulations use :func:`simulate_temporal_trajectory` to march a
workflow through escalating entropy stages. The helper iterates through
``normal``, ``high_latency``, ``resource_strain``, ``schema_drift`` and
``chaotic_failure`` presets. It loads the workflow steps via ``WorkflowDB`` and
forwards ROI, resilience and degradation metrics for each stage to
``ForesightTracker.record_cycle_metrics`` with ``compute_stability=True``.  Each
entry is annotated with the preset name under a ``stage`` field and includes a
``stability`` value summarising the trend over the rolling window. Together
these fields let the sandbox model long‑term decay across scenarios:

```python
from sandbox_runner.environment import simulate_temporal_trajectory
from menace_sandbox.foresight_tracker import ForesightTracker

tracker = ForesightTracker()
simulate_temporal_trajectory(workflow_id, foresight_tracker=tracker)

cycle = tracker.history[str(workflow_id)][-1]
print(
    cycle["stage"],
    cycle["roi_delta"],
    cycle["resilience"],
    cycle["stability"],
    cycle["scenario_degradation"],
)
```

This captures the stage label, per-stage ROI and resilience, the degradation
from the baseline run and the computed ``stability`` score. Analysing how
stability changes between stages highlights long‑term decay in the workflow's
performance.

## Cold starts and template trajectories

Cold‑start mode keeps brand‑new workflows from triggering false volatility
signals by seeding them with a baseline ROI curve until enough real data is
available. Workflows with fewer than three recorded cycles are treated as cold
starts. The `is_cold_start(workflow_id)` helper detects this warm‑up period and
also returns `True` when no ROI metric has been captured yet. During this phase
the tracker blends real ROI measurements with synthetic template values.

The templates live at `configs/foresight_templates.yaml` in the repository's
`configs` directory. The file recognises the following top‑level keys:

- `profiles` – maps workflow identifiers to ROI template names.
- `trajectories` – ROI sequences used for bootstrapping.
- `entropy_profiles` / `risk_profiles` – optional mappings for baseline
  entropy and risk curves.
- `entropy_trajectories` / `risk_trajectories` – sample entropy and risk
  trajectories referenced by the profile mappings.

For example:

```yaml
profiles:
  scraper_bot: slow_riser
  trading_bot: early_volatile
trajectories:
  slow_riser:      [0.05, 0.1, 0.2, 0.35, 0.5]
  early_volatile:  [0.4, -0.15, 0.5, -0.1, 0.45]
# Optional entropy and risk baselines
entropy_profiles:
  scraper_bot: low_entropy
risk_profiles:
  scraper_bot: low_risk
entropy_trajectories:
  low_entropy: [0.1, 0.15, 0.2, 0.25, 0.3]
risk_trajectories:
  low_risk: [0.1, 0.1, 0.1, 0.1, 0.1]
```

Here `scraper_bot` uses the `slow_riser` profile while `trading_bot` follows
`early_volatile`. Additional profiles can be configured by editing
`configs/foresight_templates.yaml`.

When capturing metrics, the first five logged cycles blend real observations
with the template trajectory using:

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

## Predicting ROI collapse

The helper :func:`predict_roi_collapse` analyses recent ``roi_delta`` history to
compute a slope via :func:`get_trend_curve`. Cycle‑to‑cycle volatility is
calculated as the standard deviation of ROI changes. Recorded
``scenario_degradation`` is compared against the expected entropy baseline from
:func:`get_entropy_template_curve`, and both entropy deviation and volatility
reduce the projected slope before extrapolating it a few cycles into the
future. The routine then estimates when the trajectory falls below zero.

### Risk categories

Risk is classified by combining the ROI trend with observed volatility:

- **Stable** – non‑negative slope and volatility below ``volatility_threshold``.
- **Slow decay** – gently negative slope while volatility stays low.
- **Volatile** – volatility exceeding the threshold regardless of slope.
- **Immediate collapse risk** – steep negative slope or a projected drop below
  zero within the next couple of cycles.

### Brittle detection

The tracker flags a workflow as *brittle* when small entropy changes trigger
sharp ROI declines. If ``scenario_degradation`` rises by less than ``0.05``
between the last two cycles but ``roi_delta`` drops by more than ten times that
amount, the ``brittle`` field is set to ``True``.

### Inputs

- ``workflow_id`` – identifier of the workflow to analyse. Requires the
  workflow to have recorded ``roi_delta`` entries and optionally
``scenario_degradation`` metrics.

### Returned fields

- ``risk`` – one of ``Stable``, ``Slow decay``, ``Volatile`` or
  ``Immediate collapse risk`` (see risk categories above).
- ``collapse_in`` – estimated cycles remaining before ROI becomes
  negative, or ``None`` if no collapse is predicted.
- ``brittle`` – ``True`` when small entropy changes produce large ROI drops.
- ``curve`` – projected ROI values for future cycles up to the horizon or
  until collapse.

### Example

```python
tracker = ForesightTracker(max_cycles=5)
risk = tracker.predict_roi_collapse("workflow-1")
if risk["risk"] in {"Immediate collapse risk", "Volatile"} or risk["brittle"]:
    print("promotion blocked")
else:
    print("workflow is safe to promote")
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
