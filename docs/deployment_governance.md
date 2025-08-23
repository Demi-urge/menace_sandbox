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

An example policy file is available at
[`examples/deployment_policy.yaml`](examples/deployment_policy.yaml). Placing a
file with the same structure at `config/deployment_governance.yaml` or
`config/deployment_governance.json` will make the rules load automatically.

## Configuration

Custom rule files are searched for in the module's `config` directory. Threshold
defaults such as `raroi_threshold`, `confidence_threshold` and
`scenario_score_min` come from `config/deployment_policy.yaml`. The policy file
also supports a `max_variance` field limiting the allowed variance between
scenario scores and a `scenario_thresholds` mapping for per‑scenario minimums.
Workflows are demoted when variance exceeds `max_variance` or any listed
scenario falls below its configured threshold. Operators may
override these values by passing a `policy` mapping to `evaluate_workflow`.

## Sample scorecard

`evaluate_workflow` expects a mapping of workflow metrics including alignment,
RAROI and confidence. A minimal example scorecard is provided in
[`examples/scorecard.json`](examples/scorecard.json).

## Evaluation process

1. Alignment and security statuses are checked first. Any failure immediately
   returns a demotion with an `alignment_veto` or `security_veto` reason code.
2. Unless a `bypass_micro_pilot` override is present, low sandbox ROI paired
   with high adapter ROI triggers the automatic **micro‑pilot** path.
3. Remaining rules are evaluated in order using the metrics from the supplied
   scorecard. The first rule whose `condition` evaluates truthy decides the
   verdict and contributes its `reason_code` to the output.
4. If no rule matches, the built‑in default of ``promote`` is returned.

## Foresight promotion gate

Before a `promote` verdict is finalised, `DeploymentGovernor` calls
`foresight_gate.is_foresight_safe_to_promote()` to simulate the patch with
`UpgradeForecaster`, `ForesightTracker` and a `WorkflowGraph`. The gate enforces
per‑cycle ROI and confidence thresholds (defaults: `roi_threshold=0.0`,
`confidence_threshold=0.6`), rejects upcoming collapse risk and, unless
`allow_negative_dag` is set, any negative downstream ROI from
`WorkflowGraph.simulate_impact_wave()`.

Each decision is appended to `forecast_records/decision_log.jsonl` via
`ForecastLogger`. Every JSON line contains a `timestamp`, `workflow_id`, a
`patch_summary`, the full list of `forecast_projections`, the overall
`forecast_confidence`, optional `dag_impact` summaries, the boolean `decision`
and the collected `reason_codes`.

When the gate fails, `evaluate()` downgrades the verdict to `borderline` if a
`BorderlineBucket` was supplied, otherwise it falls back to a `pilot` run. The
caller can then queue the workflow for borderline review or run a limited
pilot.

A minimal example of integrating the gate with :class:`DeploymentGovernor`
is available at [`examples/foresight_gate.py`](examples/foresight_gate.py).

## Governance outcomes log

`append_governance_result()` can persist each decision for later analysis. The
helper appends newline‑delimited JSON entries to
`sandbox_data/governance_outcomes.jsonl`. Each record contains the original
`scorecard`, any governance `vetoes` and, when available, the foresight gate
`forecast` details and corresponding `reasons` codes.

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
are merged into the returned `overrides` mapping. Common flags include:

- `bypass_micro_pilot` – skip the automatic micro-pilot trigger.
- `verdict` / `forced_verdict` – force the final decision (`promote`, `demote`,
  `pilot` or `no_go`).

A validated forced verdict appends the `manual_override` reason code and
replaces the computed decision.
