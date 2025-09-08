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
result = calc.calculate(metrics, "scraper_bot")
score, vetoed, triggers = result.score, result.vetoed, result.triggers
```
The returned :class:`ROIResult` exposes the weighted score, a boolean
indicating whether any veto fired and a list of the triggered veto descriptions.
It also supports tuple unpacking for backwards compatibility.

`log_debug()` uses the standard :mod:`logging` module to emit a human readable
breakdown of each contribution and any veto triggers at ``DEBUG`` level.

## Suggesting fixes

The :func:`propose_fix` helper highlights metrics that cap ROI and offers
remediation hints. It first checks the profile's veto rules – any metric below
its ``min``, above ``max`` or matching an ``equals`` value is treated as a hard
ROI cap and included in the suggestion list. Remaining metrics are then sorted
by their weighted contribution so the weakest ones surface next. Hints come
from ``configs/roi_fix_rules.yaml`` which maps each metric to a default
remediation message.

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

### Configuring suggestion mappings

Remediation messages are loaded from ``configs/roi_fix_rules.yaml`` where each
metric maps to a human‑readable hint. The file can override or extend the
built‑in defaults:

```yaml
profitability: "optimise revenue streams; reduce costs"
latency: "cache API responses"
new_metric: "add unit tests"
```

Metrics missing from the file fall back to ``"improve <metric>"`` so additional
metrics can be introduced incrementally.

### Closing the loop

Downstream services can act on these hints to automatically raise tickets or
craft Codex prompts. For example, a simple ticket integration might look like:

```python
from menace_sandbox.roi_calculator import propose_fix
from mytracker import create_issue

fixes = propose_fix(metrics, "scraper_bot")
for metric, hint in fixes:
    create_issue(title=f"Improve {metric}", body=hint)
```

Using Codex to draft a patch can reuse the same hints:

```python
prompt = """Optimize the following aspect:
{metric}: {hint}
"""
code = openai.Completion.create(model="codex", prompt=prompt.format(metric=metric, hint=hint))
```

This feedback loop ensures that low-scoring metrics quickly result in concrete
remediation steps.

### Error logger integration

The :class:`~menace_sandbox.error_logger.ErrorLogger` calls
``propose_fix`` via :meth:`log_roi_cap`, emitting a ``ROIBottleneck`` event
with the suggestions. Consumers can translate the event into automatic fix
tickets or feed the hints into a Codex prompt.

```python
from menace_sandbox.error_logger import ErrorLogger
from vector_service.context_builder import ContextBuilder

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
elog = ErrorLogger(context_builder=builder)
elog.log_roi_cap(metrics, "scraper_bot")
```

Each event's ``suggestions`` field contains the ``(metric, hint)`` pairs shown
above, allowing downstream automation to open tracker tickets or craft patch
prompts for large language models.
