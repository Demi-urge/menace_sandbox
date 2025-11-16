import types

from menace.unified_event_bus import UnifiedEventBus
from menace.evaluation_manager import EvaluationManager
from menace.evaluation_worker import EvaluationWorker


def test_worker_publishes_results():
    bus = UnifiedEventBus()
    mgr = EvaluationManager()
    mgr.engines = {"A": types.SimpleNamespace(evaluate=lambda: {"cv_score": 0.2})}
    results = []
    bus.subscribe("evaluation:result", lambda t, e: results.append(e))
    EvaluationWorker(bus, mgr)
    bus.publish("evaluation:run", {"engine": "A"})
    assert results and results[0]["result"]["cv_score"] == 0.2
