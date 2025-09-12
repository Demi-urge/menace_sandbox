import sys
import types
from pathlib import Path

# Minimal stubs to satisfy imports
cbi = types.ModuleType("menace.coding_bot_interface")
cbi.self_coding_managed = lambda f: f
sys.modules.setdefault("menace.coding_bot_interface", cbi)

dpr = types.ModuleType("dynamic_path_router")
dpr.resolve_path = lambda p: Path(p)
sys.modules.setdefault("dynamic_path_router", dpr)

scm_mod = types.ModuleType("menace.self_coding_manager")
class SelfCodingManager:
    def __init__(self, **kwargs):
        self.bot_name = kwargs.get("bot_name", "dummy")
        self.calls = []
    def register_patch_cycle(self, description, context_meta=None):
        self.calls.append((description, context_meta))
    def run_patch(self, *a, **k):
        pass
    def should_refactor(self):
        return True
scm_mod.SelfCodingManager = SelfCodingManager
scm_mod.HelperGenerationError = RuntimeError
sys.modules.setdefault("menace.self_coding_manager", scm_mod)

from menace.data_bot import DataBot, MetricsDB
from menace.evolution_orchestrator import EvolutionOrchestrator

class DummyBus:
    def __init__(self):
        self.subs = {}
    def subscribe(self, topic, fn):
        self.subs.setdefault(topic, []).append(fn)
    def publish(self, topic, payload):
        for fn in self.subs.get(topic, []):
            fn(topic, payload)

def test_threshold_breach_registers_patch_cycle(tmp_path):
    bus = DummyBus()
    db = MetricsDB(path=tmp_path / "metrics.db")
    data_bot = DataBot(db=db, event_bus=bus)
    manager = SelfCodingManager(bot_name="dummy")
    EvolutionOrchestrator(
        data_bot,
        types.SimpleNamespace(trend_predictor=None),
        types.SimpleNamespace(),
        types.SimpleNamespace(),
        selfcoding_manager=manager,
        event_bus=bus,
        history_db=types.SimpleNamespace(add=lambda *a, **k: None),
    )
    data_bot.check_degradation("dummy", roi=1.0, errors=0.0)
    data_bot.check_degradation("dummy", roi=0.0, errors=5.0)
    assert manager.calls
