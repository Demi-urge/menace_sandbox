# Prompt Optimizer

`PromptOptimizer` analyses experiment logs to discover effective prompt formats.
It groups prompts by structural features and tracks success rates, ROI and
runtime improvements to suggest high-performing configurations.

## Failure Fingerprint Penalties

Pass a path to `failure_fingerprints.jsonl` via the
`failure_fingerprint_path` argument when constructing `PromptOptimizer`.
During aggregation, each fingerprint entry's `prompt_text` is matched against
the collected prompt statistics. A synthetic failure with negative ROI is
applied for the corresponding configuration, reducing its overall score. This
ensures prompts that frequently produce bad failure fingerprints rank lower in
suggestions.
