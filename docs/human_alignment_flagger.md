# Human Alignment Flagger Settings

The sandbox includes a lightweight checker that scans patches for potential
alignment regressions. `SandboxSettings` exposes several knobs to configure this
behaviour. Each setting can be overridden via environment variables.

| Environment variable | Default | Rationale |
| --- | --- | --- |
| `ENABLE_ALIGNMENT_FLAGGER` | `true` | Enabled to surface potential safety issues without manual review. |
| `ALIGNMENT_WARNING_THRESHOLD` | `0.5` | Warn at moderate risk scores while keeping noise manageable. |
| `ALIGNMENT_FAILURE_THRESHOLD` | `0.9` | Escalate only very high scores to avoid false positives. |
| `ALIGNMENT_BASELINE_METRICS_PATH` | `sandbox_metrics.yaml` | Use the repository's metrics snapshot for comparisons. |

These defaults aim to flag questionable changes early while avoiding excessive
alerts. Adjust them as needed for stricter or more permissive reviews.

