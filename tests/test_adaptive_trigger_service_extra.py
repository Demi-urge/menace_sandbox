import types
import os
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
import sys
import importlib.util
import types

# create a minimal menace package to satisfy relative imports
if 'menace' not in sys.modules:
    menace_pkg = types.ModuleType('menace')
    menace_pkg.__path__ = [os.path.dirname(os.path.dirname(__file__))]
    sys.modules['menace'] = menace_pkg

# load module in isolation to avoid heavy optional dependencies
sys.modules.setdefault('menace.capital_management_bot', types.SimpleNamespace(CapitalManagementBot=object))
sys.modules.setdefault('menace.data_bot', types.SimpleNamespace(DataBot=object))
sys.modules.setdefault('menace.unified_event_bus', types.SimpleNamespace(UnifiedEventBus=object))
root_dir = os.path.dirname(os.path.dirname(__file__))
spec = importlib.util.spec_from_file_location(
    'menace.adaptive_trigger_service', os.path.join(root_dir, 'adaptive_trigger_service.py')
)
adaptive_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adaptive_mod)
AdaptiveTriggerService = adaptive_mod.AdaptiveTriggerService
spec_bus = importlib.util.spec_from_file_location(
    'menace.unified_event_bus', os.path.join(root_dir, 'unified_event_bus.py')
)
bus_mod = importlib.util.module_from_spec(spec_bus)
spec_bus.loader.exec_module(bus_mod)
UnifiedEventBus = bus_mod.UnifiedEventBus


def test_event_payload_contains_metadata(monkeypatch):
    bus = UnifiedEventBus()
    payloads = []
    bus.subscribe("evolve:self_improve", lambda t, e: payloads.append(e))
    data_bot = types.SimpleNamespace(db=types.SimpleNamespace(fetch=lambda limit=30: [{"errors": 5, "cpu": 50.0}]), patch_db=None)
    cap_bot = types.SimpleNamespace(energy_score=lambda **k: 0.2)
    svc = AdaptiveTriggerService(data_bot, cap_bot, bus, interval=0, error_threshold=0.1)
    svc.running = True
    monkeypatch.setattr("menace.adaptive_trigger_service.time.sleep", lambda x: (_ for _ in ()).throw(SystemExit))
    try:
        svc._loop()
    except SystemExit:
        pass
    assert payloads
    payload = payloads[0]
    assert "timestamp" in payload and "interval" in payload and "failure_count" in payload

def test_custom_threshold_strategy(monkeypatch):
    bus = UnifiedEventBus()
    events = []
    bus.subscribe("custom", lambda t, e: events.append(e))

    def metric_getter():
        return 5.0

    def strategy(name, base, history):
        return 4.0  # constant threshold

    extra = {"latency": (metric_getter, 1.0, "custom")}
    data_bot = types.SimpleNamespace(db=types.SimpleNamespace(fetch=lambda limit=30: []), patch_db=None)
    cap_bot = types.SimpleNamespace(energy_score=lambda **k: 1.0)
    svc = AdaptiveTriggerService(
        data_bot,
        cap_bot,
        bus,
        interval=0,
        threshold_strategy=strategy,
        extra_metrics=extra,
    )
    svc.running = True
    monkeypatch.setattr("menace.adaptive_trigger_service.time.sleep", lambda x: (_ for _ in ()).throw(SystemExit))
    try:
        svc._loop()
    except SystemExit:
        pass
    assert events
