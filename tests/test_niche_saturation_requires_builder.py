import types
import sys
from pathlib import Path
import pytest

# Create a lightweight menace package to avoid heavy imports
ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("menace")
pkg.__path__ = [str(ROOT)]
sys.modules["menace"] = pkg

# stub dependencies before importing module
stub_ctx = types.ModuleType("vector_service.context_builder")
class ContextBuilder:
    def refresh_db_weights(self):
        pass
    def build(self, *a, **k):
        return ""
stub_ctx.ContextBuilder = ContextBuilder
sys.modules["vector_service.context_builder"] = stub_ctx

rab_mod = types.ModuleType("menace.resource_allocation_bot")
class ResourceAllocationBot:
    def __init__(self, *a, **k):
        self.context_builder = k.get("context_builder")
    def allocate(self, *a, **k):
        return []
rab_mod.ResourceAllocationBot = ResourceAllocationBot
sys.modules["menace.resource_allocation_bot"] = rab_mod
setattr(pkg, "resource_allocation_bot", rab_mod)

rpb_mod = types.ModuleType("menace.resource_prediction_bot")
class ResourceMetrics:
    def __init__(self, cpu=0.0, memory=0.0, disk=0.0, time=0.0):
        self.cpu = cpu
        self.memory = memory
        self.disk = disk
        self.time = time
rpb_mod.ResourceMetrics = ResourceMetrics
sys.modules["menace.resource_prediction_bot"] = rpb_mod
setattr(pkg, "resource_prediction_bot", rpb_mod)

pmb_mod = types.ModuleType("menace.prediction_manager_bot")
class PredictionManager:
    def __init__(self, *a, **k):
        self.registry = {}
    def assign_prediction_bots(self, *_):
        return []
pmb_mod.PredictionManager = PredictionManager
sys.modules["menace.prediction_manager_bot"] = pmb_mod
setattr(pkg, "prediction_manager_bot", pmb_mod)

spb_mod = types.ModuleType("menace.strategy_prediction_bot")
class StrategyPredictionBot:
    pass
spb_mod.StrategyPredictionBot = StrategyPredictionBot
sys.modules["menace.strategy_prediction_bot"] = spb_mod
setattr(pkg, "strategy_prediction_bot", spb_mod)

import menace.niche_saturation_bot as nsb


class DummyAllocBot:
    def __init__(self):
        self.context_builder = object()
    def allocate(self, *_):
        return []


class DummyDB:
    def add(self, *_):
        pass


def test_requires_context_builder():
    alloc = DummyAllocBot()
    with pytest.raises(TypeError):
        nsb.NicheSaturationBot(db=DummyDB(), alloc_bot=alloc, context_builder=None)
