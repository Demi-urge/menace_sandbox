# Distributed Benchmarking

`ModelEvaluationService` now publishes evaluation jobs on the `"evaluation:run"` topic of `UnifiedEventBus`. A separate `EvaluationWorker` subscribes to this topic, runs the evaluation for the requested engine and publishes the outcome on `"evaluation:result"`.

```python
from menace.unified_event_bus import UnifiedEventBus
from menace.model_evaluation_service import ModelEvaluationService
from menace.evaluation_worker import EvaluationWorker

bus = UnifiedEventBus()
service = ModelEvaluationService(event_bus=bus)
worker = EvaluationWorker(bus, service.manager)

service.run_cycle()  # publishes tasks and waits for results
```

Run multiple workers across different hosts using a networked bus to distribute the benchmarking load.
