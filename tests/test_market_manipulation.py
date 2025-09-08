import importlib.util
import os
import sys
import types
from pathlib import Path
from dataclasses import dataclass
import pytest

pytest.skip("optional dependencies not installed", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("menace")
pkg.__path__ = [str(ROOT)]
sys.modules["menace"] = pkg

# Minimal competitive intelligence stub
class _DummyIntel:
    def __init__(self):
        self.db = types.SimpleNamespace(fetch=lambda limit=50: [])

ci_mod = types.ModuleType("competitive_intelligence_bot")
ci_mod.CompetitiveIntelligenceBot = _DummyIntel
sys.modules["menace.competitive_intelligence_bot"] = ci_mod

# Minimal niche saturation stub
class _DummySaturation:
    def __init__(self):
        self.calls = 0

    def saturate(self, candidates):
        self.calls += 1
        return [(candidates[0].name if candidates else "", True)]

@dataclass
class _NicheCandidate:
    name: str
    demand: float
    competition: float
    trend: float = 0.0

ns_mod = types.ModuleType("niche_saturation_bot")
ns_mod.NicheSaturationBot = _DummySaturation
ns_mod.NicheCandidate = _NicheCandidate
sys.modules["menace.niche_saturation_bot"] = ns_mod
sys.modules["niche_saturation_bot"] = ns_mod
setattr(pkg, "niche_saturation_bot", ns_mod)
ns_mod.__spec__ = importlib.machinery.ModuleSpec("menace.niche_saturation_bot", loader=None)

stub_ctx = types.ModuleType("vector_service.context_builder")
class ContextBuilder:
    def __init__(self, *a, **k):
        self.calls = []

    def refresh_db_weights(self):
        self.calls.append("refresh")

    def build_context(self, query, *a, **k):
        self.calls.append(query)
        return "{}"
stub_ctx.ContextBuilder = ContextBuilder
sys.modules["vector_service.context_builder"] = stub_ctx

alloc_mod = types.ModuleType("menace.resource_allocation_bot")
class ResourceAllocationBot:
    def __init__(self, *a, **k):
        pass
alloc_mod.ResourceAllocationBot = ResourceAllocationBot
sys.modules["menace.resource_allocation_bot"] = alloc_mod
setattr(pkg, "resource_allocation_bot", alloc_mod)
sys.modules["resource_allocation_bot"] = alloc_mod
alloc_mod.__spec__ = importlib.machinery.ModuleSpec("menace.resource_allocation_bot", loader=None)

cc_mod = types.ModuleType("compliance_checker")

class AuditTrail:
    def __init__(self, path):
        pass

    def record(self, msg):
        pass


class ComplianceChecker:
    def __init__(self, log_path: str | None = None) -> None:
        self.audit = AuditTrail(log_path or "log")

    def check_trade(self, trade):
        vol = float(trade.get("volume", 0))
        limit = float(os.getenv("MAX_TRADE_VOLUME", "1000"))
        allowed = vol <= limit
        self.audit.record({"type": "volume", "trade": dict(trade), "allowed": allowed})
        return allowed

    def verify_permission(self, role, action):
        allowed = role == "trader" and action == "trade"
        self.audit.record({"type": "permission", "role": role, "action": action, "allowed": allowed})
        return allowed

cc_mod.ComplianceChecker = ComplianceChecker
cc_mod.AuditTrail = AuditTrail
sys.modules["menace.compliance_checker"] = cc_mod
cc = cc_mod
setattr(pkg, "compliance_checker", cc_mod)
cc_mod.__spec__ = importlib.machinery.ModuleSpec("menace.compliance_checker", loader=None)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"menace.{name}", ROOT / f"{name}.py", submodule_search_locations=[str(ROOT)]  # path-ignore
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"menace.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod

mm = _load("market_manipulation_bot")


DummyIntel = _DummyIntel
DummySaturation = _DummySaturation


def test_compliance_enforced(monkeypatch):
    records = []

    class DummyTrail:
        def __init__(self, path):
            pass

        def record(self, msg):
            records.append(msg)

    monkeypatch.setattr(cc, "AuditTrail", DummyTrail)
    monkeypatch.setenv("MAX_TRADE_VOLUME", "1")

    checker = cc.ComplianceChecker()
    sat1 = DummySaturation()
    bot = mm.MarketManipulationBot(
        DummyIntel(), sat1, checker=checker, role="trader", context_builder=ContextBuilder()
    )
    res = bot.saturate(["one"])
    assert sat1.calls == 1
    assert res and res[0][0] == "one"

    sat2 = DummySaturation()
    bot2 = mm.MarketManipulationBot(
        DummyIntel(), sat2, checker=checker, role="trader", context_builder=ContextBuilder()
    )
    res2 = bot2.saturate(["a", "b"])
    assert sat2.calls == 0
    assert not res2
    assert any(r.get("type") == "volume" and not r.get("allowed") for r in records)

    sat3 = DummySaturation()
    bot3 = mm.MarketManipulationBot(
        DummyIntel(), sat3, checker=checker, role="viewer", context_builder=ContextBuilder()
    )
    res3 = bot3.saturate(["x"])
    assert sat3.calls == 0
    assert not res3
    assert any(r.get("type") == "permission" and not r.get("allowed") for r in records)
