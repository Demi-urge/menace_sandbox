# Prompt Optimizer

`PromptOptimizer` analyses experiment logs to discover effective prompt formats.
It groups prompts by structural features and tracks success rates, ROI and
runtime improvements to suggest high-performing configurations.

## Failure Fingerprint Penalties

Pass a path to `failure_fingerprints.jsonl` via the
`failure_fingerprints_path` argument when constructing `PromptOptimizer`.
Fingerprints are grouped by their prompt text and counted. If the number of
fingerprints associated with a prompt configuration exceeds the
`fingerprint_threshold`, the optimiser reduces that configuration's recorded
successes accordingly. This allows repeated failures to bias the success rate
without needing corresponding log entries.
