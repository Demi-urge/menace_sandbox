# ROI Calculator

The `ROICalculator` computes a weighted return-on-investment score for a
set of metrics. Profiles live in a YAML file and define metric weights and
veto rules that can force a hard failure when certain thresholds are
violated.

## Profile format

Each profile contains a `weights` mapping for the eight supported metrics and
a `veto` section listing hard constraints:

```yaml
scraper_bot:
  weights:
    profitability: 0.3
    efficiency: 0.2
    reliability: 0.1
    resilience: 0.1
    maintainability: 0.1
    security: 0.1
    latency: -0.1
    energy: -0.05
  veto:
    security: {min: 0.4}
    alignment_violation: {equals: true}
```

`min` and `max` apply to numeric metrics. `equals` can match booleans or
strings and triggers when the metric value exactly equals the provided
literal.

## Usage

```python
from menace_sandbox.roi_calculator import ROICalculator

calc = ROICalculator()  # loads configs/roi_profiles.yaml by default
metrics = {"profitability": 0.8, "security": 0.5}
score, vetoed, triggers = calc.calculate(metrics, "scraper_bot")
```

The tuple contains the weighted score, a boolean indicating whether any veto
fired and a list of the triggered veto descriptions.

`log_debug()` uses the standard :mod:`logging` module to emit a human readable
breakdown of each contribution and any veto triggers at ``DEBUG`` level.

## Suggesting fixes

The :func:`propose_fix` helper highlights metrics that cap ROI and offers
remediation hints. It first checks the profile's veto rules – any metric below
its ``min``, above ``max`` or matching an ``equals`` value is treated as a hard
ROI cap and included in the suggestion list. Remaining metrics are then sorted
by their weighted contribution so the weakest ones surface next.

```python
from menace_sandbox.roi_calculator import ROICalculator, propose_fix

calc = ROICalculator()
profile = calc.profiles["scraper_bot"]
metrics = {"profitability": 0.5, "efficiency": 0.1, "security": 0.2,
           "latency": 0.9, "energy": 0.2}
fixes = propose_fix(metrics, profile)
print(fixes)
```

Sample output:

```text
[('security', 'harden authentication; add input validation'),
 ('latency', 'optimise I/O; use caching'),
 ('energy', 'batch work; reduce polling')]
```

### Error logger integration

The :class:`~menace_sandbox.error_logger.ErrorLogger` calls
``propose_fix`` via :meth:`log_roi_cap`, emitting a ``ROIBottleneck`` event
with the suggestions. Downstream services can turn this telemetry into fix
tickets or craft Codex prompts for automated patches.
