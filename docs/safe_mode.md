# Safe Mode and Overrides

Menace historically offered a conservative mode that disabled code changes and
structural evolution. `SelfServiceOverride` previously toggled this mode when
ROI or error metrics deteriorated. It now only records a warning when
thresholds are breached. Manual variables `MENACE_SAFE` and `EVOLUTION_PAUSED`
have been removed.

## When to enable safe mode

Watch for the same triggers&mdash;an ROI drop of more than **10%**, an error rate
above **25%** or an energy score below **0.3**. These limits still mirror the
defaults used by `AdaptiveTriggerService` and `SystemEvolutionManager`, but they
no longer toggle safe mode automatically.

## Automatic rollback

`AutoRollbackService` still watches ROI, error rates and energy score. When a
threshold is breached it logs a warning with the offending commit hash instead
of running `git revert`. Manual rollback is left to the operator.
