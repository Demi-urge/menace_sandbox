import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.market_manipulation_bot as mm
from vector_service.context_builder import ContextBuilder


def test_saturate_simple():
    bot = mm.MarketManipulationBot(context_builder=ContextBuilder())
    res = bot.saturate(["niche"])
    assert res and res[0][0] == "niche"
