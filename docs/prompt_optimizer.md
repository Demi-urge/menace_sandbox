# Prompt Optimizer

`PromptOptimizer` analyses experiment logs to discover effective prompt formats.
It groups prompts by structural features and tracks success rates, ROI and
runtime improvements to suggest high-performing configurations.

## Failure Fingerprint Penalties

Pass a path to `failure_fingerprints.jsonl` via the
`failure_fingerprints_path` argument when constructing `PromptOptimizer`.
Fingerprints are grouped into similarity clusters. For each cluster whose size
exceeds `fingerprint_threshold`, the optimiser deducts the cluster size from the
configuration's successes and scales its score by `1 / (1 + size -
fingerprint_threshold)`. Larger clusters therefore impose heavier penalties,
allowing repeated failures to bias the success rate without needing
corresponding log entries.
