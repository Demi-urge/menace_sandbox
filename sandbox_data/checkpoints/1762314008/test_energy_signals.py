import pytest
import menace.neuroplasticity as neu
import menace.data_bot as db


def test_energy_signals(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    data_bot = db.DataBot(mdb)
    mdb.add(db.MetricRecord(bot="a", cpu=1.0, memory=1.0, response_time=0.0, disk_io=0.0, net_io=0.0, errors=0, revenue=10.0, expense=5.0))
    mdb.add(db.MetricRecord(bot="a", cpu=1.0, memory=1.0, response_time=0.0, disk_io=0.0, net_io=0.0, errors=0, revenue=5.0, expense=5.0))
    delta = data_bot.engagement_delta(limit=10)
    assert delta == pytest.approx((15.0 - 10.0) / 10.0)

    pdb = neu.PathwayDB(tmp_path / "p.db")
    rec = neu.PathwayRecord(actions="A", inputs="", outputs="", exec_time=1.0, resources="", outcome=neu.Outcome.SUCCESS, roi=1.0)
    pdb.log(rec)
    assert pdb.highest_myelination_score() > 0.0
