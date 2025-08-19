# Risk-Adjusted ROI (RAROI)

The risk-adjusted return on investment (RAROI) refines raw ROI values so that
workflows with dangerous side effects are penalized while stable and safe
workflows are rewarded. The adjustment consists of three multiplicative
components:

1. **Catastrophic risk** – combines the probability that a workflow must be
   rolled back with the severity of the potential impact. The probability is
   estimated from runtime metrics (error rates, test flakiness or recent
   failure rates) normalised to a 0–1 range, while the severity is looked up
   using the `impact_severity` configuration mapping. RAROI reduces the base
   ROI by this product: `1 - (rollback_probability * impact_severity)`.
2. **Stability factor** – measures how consistent recent ROI deltas have been.
   The standard deviation of the last ten deltas is subtracted from one,
   yielding a number between 0 and 1. Highly volatile histories therefore
   reduce the RAROI while stable performance keeps the factor close to one.
3. **Safety factor** – aggregates security and alignment metrics (such as
   `safety_rating` or `security_score`) and applies additional penalties for
   recorded hostile or misuse failures. The factor falls between 0 and 1,
   shrinking the RAROI when safety metrics degrade.

The final formula is:

```
raroi = base_roi * (1 - catastrophic_risk) * stability_factor * safety_factor
```

Higher RAROI values promote a workflow in ranking calculations while lower
values push it downward, allowing ranking systems to favour stable and safe
workflows.

## Configuring impact severity

Impact severity values live in `config/impact_severity.yaml`. A minimal example
looks like:

```yaml
experimental: 0.2
standard: 0.5
critical: 0.9
```

Set the `IMPACT_SEVERITY_CONFIG` environment variable to point to a custom YAML
file if different defaults are required. These values represent the perceived
impact if a workflow must be rolled back and can be tuned to match deployment
environments. Missing entries default to `standard`.
