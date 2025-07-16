import menace.customer_acquisition_memory as cam


def test_log_and_query(tmp_path):
    db = cam.CustomerAcquisitionDB(tmp_path / "cad.db")
    rec1 = cam.InteractionRecord(
        platform="Reddit",
        user_id="u1",
        pitch_script="p1",
        language_style="casual",
        emotional_strategy="validation",
        conversion=True,
        age=25,
        gender="M",
        location="US",
    )
    db.log(rec1)
    rows = db.list_interactions("Reddit")
    assert len(rows) == 1
    assert rows[0].pitch_script == "p1"
    rates = db.conversion_rates()
    assert rates["p1"] == 1.0
    assert db.best_pitch_by_platform("Reddit") == "p1"
