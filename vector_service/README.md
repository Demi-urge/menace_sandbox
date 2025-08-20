# Vector Service

## Safety filtering

`ContextBuilder` and `Retriever` apply configurable safety thresholds:

- `max_alignment_severity`: skip vectors with `alignment_severity` above this value.
- `max_alerts`: skip vectors with more than this number of `semantic_alerts`.
- `license_denylist`: skip vectors whose `license` or fingerprint maps to a denylisted identifier.

Filtered vectors are counted by `PatchLogger` using the metric `patch_logger_vectors_total{risk="filtered"}`.

### Examples

With `max_alignment_severity=0.5`, `max_alerts=1` and `license_denylist={"GPL-3.0"}`:

- a vector with `alignment_severity=0.8` is dropped;
- a vector containing two `semantic_alerts` is dropped;
- a vector marked with license `GPL-3.0` is dropped.
