# Safe Mode and Overrides

Menace supports a conservative mode that disables code changes and structural
evolution. `SelfServiceOverride` monitors ROI and error metrics to toggle safe
mode automatically. Manual variables `MENACE_SAFE` and `EVOLUTION_PAUSED` are
no longer required.

## When to enable safe mode

Activate safe mode when ROI drops by more than **10%**, the overall error rate
exceeds **25%** or the energy score falls below **0.3**. These thresholds mirror
the defaults used by `AdaptiveTriggerService` and `SystemEvolutionManager`.

## Automatic rollback

`AutoRollbackService` monitors ROI drop, error rate and energy score. When any
threshold is breached it enables safe mode and runs `git revert` on the most
recent commit. Metrics continue to be collected until performance recovers.
