# Human Alignment Flagging and Review

The sandbox ships with a lightweight checker and background review agent to
surface potential safety regressions before changes are merged.

## Components

* `HumanAlignmentFlagger` scans Git diffs for removed documentation, rising
  complexity, risky patterns and ethics violations.  It is intentionally
  conservative and never raises, returning a structured report of issues.
* `AlignmentReviewAgent` runs in the background and polls recent alignment
  warnings.  Each unseen record is forwarded to `SecurityAuditor` so Security AI
  can triage and escalate problems.

## Configuring thresholds via `SandboxSettings`

Alignment sensitivity is controlled through `SandboxSettings` and may be
overridden either programmatically or with environment variables:

```python
from sandbox_settings import SandboxSettings

settings = SandboxSettings(
    enable_alignment_flagger=True,
    alignment_warning_threshold=0.5,
    alignment_failure_threshold=0.9,
)
```

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
of issues with severity scores.  After each commit the autonomous sandbox
invokes the flagger and stores the report alongside the commit hash.

## Baseline comparisons

When a baseline metrics file is supplied via `ALIGNMENT_BASELINE_METRICS_PATH`
the flagger loads values such as `tests` and `complexity` from that snapshot.
The current changes are compared against the baseline and warnings are emitted
if test counts drop or overall cyclomatic complexity rises. These checks fire
even when performance metrics improve, ensuring maintainability isn't traded
for short‑term gains.  Set `ALIGNMENT_BASELINE_METRICS_PATH` to an empty string
to skip these comparisons.

## Warning logs and Security AI review

Each run appends the flagger report to `sandbox_data/alignment_flags.jsonl` and
emits an `alignment:flag` event.  `violation_logger.log_violation` mirrors the
warning to `logs/violation_log.jsonl` and stores high‑severity entries in the
SQLite store `logs/alignment_warnings.db`.

`AlignmentReviewAgent` periodically reads recent warnings via
`load_recent_alignment_warnings` and hands new ones to `SecurityAuditor`.  The
auditor persists a copy in `logs/alignment_warnings.jsonl` where Security AI or
operators can review and escalate issues.  High‑severity entries at or above
`ALIGNMENT_FAILURE_THRESHOLD` merit immediate remediation before changes are
merged.

