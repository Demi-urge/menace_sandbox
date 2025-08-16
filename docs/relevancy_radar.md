# Relevancy Radar

The relevancy radar tracks how often modules are imported and executed during
sandbox runs. The collected metrics are persisted to
`sandbox_data/relevancy_metrics.json` and can be inspected or annotated using
the `relevancy_radar_cli.py` helper.

## CLI usage

```
python relevancy_radar_cli.py --retire old_mod --compress slow_mod --replace alt_mod
```

### Dependency-aware evaluation

Pass `--final` to run `evaluate_final_contribution` before listing modules.
This performs a dependency-aware analysis that walks the import graph and
attributes usage to reachable modules. Modules outside the core dependency
chains are flagged for retirement, while those with low combined import,
execution and impact scores are annotated for compression or replacement.

The evaluation runs prior to displaying metrics, ensuring that any dependency
adjustments are reflected in the output.

## Display options

- `--threshold`: score below which modules are shown (default `5`).
- `--show-impact`: include cumulative impact scores in the listing.
- `--final`: run the dependency-aware evaluation step described above.
