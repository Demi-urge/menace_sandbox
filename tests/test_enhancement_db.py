import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.chatgpt_enhancement_bot as ceb


def test_add_and_link(tmp_path):
    db = ceb.EnhancementDB(tmp_path / "e.db")
    enh = ceb.Enhancement(
        idea="i",
        rationale="r",
        model_ids=[1],
        bot_ids=[2],
        workflow_ids=[3],
    )
    eid = db.add(enh)
    assert eid
    assert db.models_for(eid) == [1]
    assert db.bots_for(eid) == [2]
    assert db.workflows_for(eid) == [3]
    items = db.fetch()
    assert items and items[0].model_ids == [1]
