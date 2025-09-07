import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.niche_saturation_bot as nsb
import menace.resource_allocation_bot as rab
import menace.resource_prediction_bot as rpb


class _DummyBuilder:
    def build(self, *_: object, **__: object) -> str:
        return "ctx"

    def refresh_db_weights(self):
        pass


def test_saturate_logs(tmp_path):
    db = nsb.NicheDB(tmp_path / "niche.db")
    alloc_db = rab.AllocationDB(tmp_path / "alloc.db")
    alloc_bot = rab.ResourceAllocationBot(
        alloc_db, rpb.TemplateDB(tmp_path / "t.csv"), context_builder=_DummyBuilder()
    )
    bot = nsb.NicheSaturationBot(db, alloc_bot, context_builder=_DummyBuilder())
    cand = nsb.NicheCandidate(name="ai-tools", demand=5.0, competition=1.0, trend=1.0)
    actions = bot.saturate([cand])
    hist = db.history()
    assert actions and hist.iloc[0]["niche"] == "ai-tools"

