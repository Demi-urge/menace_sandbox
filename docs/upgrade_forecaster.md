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
Template ROI curves from :meth:`ForesightTracker.get_template_curve` (or CSSM‑provided profiles) are mixed with simulated metrics using ``alpha = samples / 5``.
Variance across the blended ROI values is computed and combined with the sample‑based term to yield the overall forecast confidence: ``samples / (samples + 1)`` scaled by ``1 / (1 + variance)``.
This guards against volatile early projections while still reflecting the limited data available.

## Entropy and risk templates

When available, :class:`ForesightTracker` exposes baseline entropy and risk trajectories through
``get_entropy_template_curve`` and ``get_risk_template_curve``. ``UpgradeForecaster`` blends these
curves with the simulated metrics during a cold start using the same ``alpha`` weighting. Entropy
influences the per‑cycle ``decay`` value while risk templates are combined with simulated risk
indices; if no templates exist, both metrics fall back to simple ROI‑based heuristics.

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

## Record retrieval

Persisted forecasts live under ``forecast_records/``. Use :func:`load_record` to reload the most
recent result for a workflow. :func:`list_records` returns a summary of all available records with
their workflow identifiers, upgrade hashes and timestamps. The JSON record includes the top‑level
``timestamp`` field shown above in addition to ``projections`` and overall ``confidence``.

## Foresight promotion gate

The higher‑level deployment flow calls
:func:`deployment_governance.is_foresight_safe_to_promote` to decide whether a
patch can be promoted.  The helper combines ``UpgradeForecaster``,
``ForesightTracker`` and a ``WorkflowGraph`` and returns ``(ok, reason_codes,
forecast)``.  Promotion proceeds only when all four gates pass:

1. every projected ROI meets or exceeds the supplied ``roi_threshold``;
2. forecast ``confidence`` is at least ``0.6`` (customisable via
   ``confidence_threshold``);
3. :meth:`ForesightTracker.predict_roi_collapse` reports neither immediate
   collapse risk nor a ``collapse_in`` value within the forecast horizon; and
4. :meth:`WorkflowGraph.simulate_impact_wave` reports no negative downstream ROI
   deltas unless ``allow_negative_dag`` is ``True``.

All gate decisions are appended to ``forecast_records/decision_log.jsonl`` using
``ForecastLogger``. Each line stores a ``timestamp``, ``workflow_id``,
``patch_summary``, the list of ``forecast_projections``, top‑level
``forecast_confidence``, an optional ``dag_impact`` mapping, the boolean
``decision`` and any ``reason_codes``.

If ``ok`` is ``False`` the calling :class:`DeploymentGovernor` downgrades the
workflow to the borderline bucket (when available) or triggers a pilot run
instead.  Reason codes such as ``projected_roi_below_threshold``,
``low_confidence``, ``roi_collapse_risk`` and ``negative_dag_impact`` identify
which gate failed.

A minimal integration example lives at
[`examples/foresight_gate.py`](examples/foresight_gate.py).
