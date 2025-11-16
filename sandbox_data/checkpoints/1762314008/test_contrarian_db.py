import menace.contrarian_db as cdb


def test_add_and_fetch(tmp_path):
    db = cdb.ContrarianDB(tmp_path / "c.db")
    rec = cdb.ContrarianRecord(
        innovation_name="hybrid",
        innovation_type="hybridization",
        risk_score=0.5,
        reward_score=1.2,
        activation_trigger="low ROI",
        resource_allocation={"cpu": 1.0},
        status="active",
    )
    cid = db.add(rec)
    db.link_model(cid, 1)
    db.link_workflow(cid, 2)
    db.link_enhancement(cid, 3)
    db.link_error(cid, 4)
    db.link_discrepancy(cid, 5)
    items = db.fetch()
    assert items and items[0].contrarian_id == cid
    assert items[0].resource_allocation["cpu"] == 1.0
    assert db.models_for(cid) == [1]
    assert db.workflows_for(cid) == [2]
    assert db.enhancements_for(cid) == [3]
    assert db.errors_for(cid) == [4]
    assert db.discrepancies_for(cid) == [5]
