# UpgradeForecaster

`UpgradeForecaster` projects return on investment (ROI) and stability for upcoming self‑improvement cycles.
It combines historical trends tracked by `ForesightTracker` with short temporal simulations of the patched workflow.

## Usage

```python
from foresight_tracker import ForesightTracker
from upgrade_forecaster import UpgradeForecaster, load_record

tracker = ForesightTracker()
forecaster = UpgradeForecaster(tracker)

# ``patch`` can be a workflow identifier or iterable of workflow steps
result = forecaster.forecast("workflow-1", patch=["step_a", "step_b"], cycles=3)

for p in result.projections:
    print(p.cycle, p.roi, p.risk, p.confidence, p.decay)
print("overall confidence", result.confidence)

# Load the persisted forecast later
saved = load_record("workflow-1")
```

The helper persists each forecast under ``forecast_records/`` and optionally logs it through ``ForecastLogger``.
Previously persisted results can be retrieved with ``load_record``.
`cycles` is clamped to the range 3–5.

## Cold‑start behaviour

When ``ForesightTracker`` has fewer than three recorded cycles for a workflow, ``UpgradeForecaster`` blends template trajectories with real metrics.
Template ROI and entropy curves from :meth:`ForesightTracker.get_template_curve` (or CSSM‑provided profiles) are mixed with simulated metrics using ``alpha = samples / 5``.
Risk is approximated as ``1 - blended_roi`` while decay derives from the blended entropy curve.
Confidence is based solely on the number of collected samples, avoiding misleading projections until real metrics accumulate.

## Output schema

``forecast`` returns a :class:`ForecastResult` object and also writes a JSON document with the following structure:

```json
{
  "workflow_id": "workflow-1",
  "patch": ["step_a", "step_b"],
  "projections": [
    {"cycle": 1, "roi": 0.1, "risk": 0.9, "confidence": 0.0, "decay": 0.0},
    {"cycle": 2, "roi": 0.2, "risk": 0.8, "confidence": 0.0, "decay": 0.0}
  ],
  "confidence": 0.5,
  "timestamp": 1690000000
}
```

Each ``projection`` entry contains the projected cycle number together with estimated ``roi``, ``risk`` (0‑1),
``confidence`` (0‑1) and ``decay``. The top‑level ``confidence`` summarises the overall certainty of the forecast.
``timestamp`` stores when the record was written as a UNIX epoch value.
