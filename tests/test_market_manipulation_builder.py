import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("menace")
pkg.__path__ = [str(ROOT)]
sys.modules["menace"] = pkg

# Minimal stubs for dependencies used during import
ci_mod = types.ModuleType("competitive_intelligence_bot")
ci_mod.CompetitiveIntelligenceBot = type("CompetitiveIntelligenceBot", (), {})
sys.modules["menace.competitive_intelligence_bot"] = ci_mod

ns_mod = types.ModuleType("niche_saturation_bot")
ns_mod.NicheSaturationBot = type("NicheSaturationBot", (), {})
ns_mod.NicheCandidate = type("NicheCandidate", (), {})
sys.modules["menace.niche_saturation_bot"] = ns_mod
sys.modules["niche_saturation_bot"] = ns_mod
setattr(pkg, "niche_saturation_bot", ns_mod)

cc_mod = types.ModuleType("compliance_checker")
cc_mod.ComplianceChecker = type("ComplianceChecker", (), {})
sys.modules["menace.compliance_checker"] = cc_mod

ctx_mod = types.ModuleType("vector_service.context_builder")
class _ContextBuilder:
    def refresh_db_weights(self):
        pass
ctx_mod.ContextBuilder = _ContextBuilder
sys.modules["vector_service.context_builder"] = ctx_mod

mm = importlib.import_module("menace.market_manipulation_bot")


def test_missing_builder_raises():
    with pytest.raises(TypeError):
        mm.MarketManipulationBot()
