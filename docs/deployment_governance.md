# Deployment Governance

`deployment_governance` evaluates whether a workflow should be **promoted**, **demoted**, sent to **pilot** or receive a "no_go" verdict. Decisions combine risk-adjusted ROI (RAROI), confidence scores and scenario stress tests. Optional rule files or signed overrides can adjust the behaviour.

## Policy schema

Policies are expressed as a list of rules. Each rule contains three fields:

- `decision` – one of `promote`, `demote`, `pilot` or `no_go`.
- `condition` – a Python expression evaluated with variables like `raroi`, `confidence` and `min_scenario`.
- `reason_code` – short code recorded in the output when the rule matches.

Rules loaded from YAML or JSON are prepended to the built-in defaults.

### Sample rules

```yaml
- decision: demote
  condition: "raroi < 0.5"
  reason_code: raroi_below_half
- decision: pilot
  condition: "sandbox_roi < sandbox_roi_low and adapter_roi > adapter_roi_high"
  reason_code: micro_pilot
- decision: promote
  condition: "True"
  reason_code: ""
```

## Evaluation process

1. Alignment and security statuses are checked first. Any failure immediately
   returns a demotion with an `alignment_veto` or `security_veto` reason code.
2. Unless a `bypass_micro_pilot` override is present, low sandbox ROI paired
   with high adapter ROI triggers the automatic **micro‑pilot** path.
3. Remaining rules are evaluated in order using the metrics from the supplied
   scorecard. The first rule whose `condition` evaluates truthy decides the
   verdict and contributes its `reason_code` to the output.
4. If no rule matches, the built‑in default of ``promote`` is returned.

## CLI and automation usage

Automation such as `deployment_bot` and `central_evaluation_loop` calls
`evaluate_workflow(scorecard, policy)` to obtain deployment verdicts. The helper
can also be invoked from the command line:

```bash
python - <<'PY'
from menace.deployment_governance import evaluate_workflow
scorecard = {"alignment_status": "pass", "raroi": 1.2, "confidence": 0.8}
policy = {"override_path": "override.json", "public_key_path": "key"}
print(evaluate_workflow(scorecard, policy))
PY
```

## Override procedures

Manual override files contain a `data` object and HMAC `signature`. When a
valid override is supplied via `override_path` / `public_key_path`, its entries
are merged into the returned `overrides` mapping. Overrides may specify flags
like ``bypass_micro_pilot`` or force a ``verdict`` such as ``promote``. A
validated forced verdict appends the ``manual_override`` reason code and
replaces the computed decision.
