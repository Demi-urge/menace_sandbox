import types
from pathlib import Path
import sys

# Stub dependencies required by billing.sanity_consumer
class _Bus:
    def __init__(self):
        self.handlers = []

    def subscribe(self, topic, cb):
        self.handlers.append((topic, cb))

    def publish(self, topic, event):
        for t, cb in list(self.handlers):
            if t == topic:
                cb(topic, event)

bus_module = types.ModuleType("unified_event_bus")
bus_module.UnifiedEventBus = _Bus
sys.modules["unified_event_bus"] = bus_module

sf_module = types.ModuleType("sanity_feedback")

class SanityFeedback:
    def __init__(self, manager, outcome_db=None, **kwargs):
        self.manager = manager
        self.outcome_db = outcome_db
        self.kwargs = kwargs

sf_module.SanityFeedback = SanityFeedback
sys.modules["sanity_feedback"] = sf_module

msl_module = types.ModuleType("menace_sanity_layer")
msl_module._EVENT_BUS = _Bus()
msl_module.record_event = lambda *a, **k: None
sys.modules["menace_sanity_layer"] = msl_module

dpr_module = types.ModuleType("dynamic_path_router")
dpr_module.resolve_path = lambda p: Path(p)
dpr_module.resolve_dir = lambda p: Path(p)
dpr_module.get_project_root = lambda: Path('.')
dpr_module.get_project_roots = lambda: [Path('.')]
sys.modules["dynamic_path_router"] = dpr_module

db_module = types.ModuleType("discrepancy_db")
class _DummyDB:
    def __init__(self, *a, **k):
        pass

db_module.DiscrepancyDB = _DummyDB
db_module.DiscrepancyRecord = object
sys.modules["discrepancy_db"] = db_module

from unified_event_bus import UnifiedEventBus  # noqa: E402
import billing.sanity_consumer as sc  # noqa: E402


class _Builder:
    def __init__(self):
        self.refreshed = False

    def refresh_db_weights(self):
        self.refreshed = True


class DummyManager:
    def __init__(self, engine):
        self.engine = engine

    def run_patch(self, *a, **k):
        pass


def test_injected_manager_used():
    class DummyEngine:
        pass

    engine = DummyEngine()
    builder = _Builder()
    consumer = sc.SanityConsumer(
        DummyManager(engine), event_bus=UnifiedEventBus(), context_builder=builder
    )
    assert consumer._get_engine() is engine
    assert builder.refreshed


def test_builder_shared_with_feedback():
    class DummyEngine:
        pass

    builder = _Builder()
    consumer = sc.SanityConsumer(
        DummyManager(DummyEngine()), event_bus=UnifiedEventBus(), context_builder=builder
    )
    feedback = consumer._get_feedback()
    assert getattr(feedback, "context_builder", None) is builder
