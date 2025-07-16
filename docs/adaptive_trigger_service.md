# Adaptive Trigger Service

`AdaptiveTriggerService` monitors metrics via `DataBot` and the energy score from
`CapitalManagementBot`. When error rates exceed a threshold or the energy score
falls below a limit it publishes events on `UnifiedEventBus`:

- `evolve:self_improve` – triggers `SelfImprovementEngine`
- `evolve:system` – triggers `EvolutionOrchestrator`

```python
from menace.adaptive_trigger_service import AdaptiveTriggerService

service = AdaptiveTriggerService(data_bot, capital_bot, event_bus,
                                 interval=30,
                                 error_threshold=0.2,
                                 energy_threshold=0.3)
service.start()
```
