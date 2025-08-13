# Human Alignment Flagger Settings

The sandbox includes a lightweight checker that scans patches for potential
alignment regressions. `SandboxSettings` exposes several knobs to configure this
behaviour. Each setting can be overridden via environment variables.

| Environment variable | Default | Rationale |
| --- | --- | --- |
| `ENABLE_ALIGNMENT_FLAGGER` | `true` | Enabled to surface potential safety issues without manual review. |
| `ALIGNMENT_WARNING_THRESHOLD` | `0.5` | Warn at moderate risk scores while keeping noise manageable. |
| `ALIGNMENT_FAILURE_THRESHOLD` | `0.9` | Escalate only very high scores to avoid false positives. |
| `ALIGNMENT_BASELINE_METRICS_PATH` | `sandbox_metrics.yaml` | Path to a YAML snapshot used for baseline comparisons. |

These defaults aim to flag questionable changes early while avoiding excessive
alerts. Adjust them as needed for stricter or more permissive reviews.

## How it works

`HumanAlignmentFlagger` parses Git diffs and applies a series of heuristics.
It highlights removed docstrings or logging statements, missing tests and high
complexity blocks and also calls auxiliary detectors for ethics violations and
risk/reward misalignment.  The result is a structured report containing a list
of issues with severity scores.

After each commit the autonomous sandbox invokes the flagger and stores the
report alongside the commit hash.

## Baseline comparisons

When a baseline metrics file is supplied via `ALIGNMENT_BASELINE_METRICS_PATH`
the flagger loads values such as `tests` and `complexity` from that snapshot.
The current changes are compared against the baseline and warnings are emitted
if test counts drop or overall cyclomatic complexity rises. These checks fire
even when performance metrics improve, ensuring maintainability isn't traded
for short‑term gains.  Set `ALIGNMENT_BASELINE_METRICS_PATH` to an empty string
to skip these comparisons.

## Logging and review

Flagger results are appended to `sandbox_data/alignment_flags.jsonl` in JSON
Lines format and published on the event bus as `alignment:flag`.  When warning
scores reach `ALIGNMENT_WARNING_THRESHOLD` the engine emits an
`alignment_warning` alert and records an `alignment_flag` entry via the audit
logger.

Security AI or developers should monitor the JSON log or subscribe to the event
bus and triage any warnings.  High‑severity entries at or above
`ALIGNMENT_FAILURE_THRESHOLD` merit immediate review and remediation before
changes are merged.

