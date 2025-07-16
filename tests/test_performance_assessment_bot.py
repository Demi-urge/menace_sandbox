import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.performance_assessment_bot as pab
import menace.data_bot as db


def test_rl_update():
    rl = pab.SimpleRL()
    state = (1, 1, 1, 0)
    before = rl.score(state)
    rl.update(state, -5)
    after = rl.score(state)
    assert after != before and after < 0


def test_self_assess_and_advise(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    rec = db.MetricRecord(
        bot="bot1", cpu=80.0, memory=70.0, response_time=0.5, disk_io=1.0, net_io=1.0, errors=2
    )
    for _ in range(3):
        mdb.add(rec)
    hist_db = pab.BotPerformanceHistoryDB(tmp_path / "h.db")
    bot = pab.PerformanceAssessmentBot(mdb, history_db=hist_db)
    for _ in range(3):
        bot.self_assess("bot1")
    advice = bot.advise("bot1")
    assert "upgrade" in advice.lower() or "optimise" in advice.lower()
    assert hist_db.history("bot1")


def test_hypothetical_projection():
    bot = pab.PerformanceAssessmentBot(db.MetricsDB(":memory:"))
    proj = bot.hypothetical_projection(pab.KPI(cpu=10.0, memory=20.0, response_time=1.0, errors=1))
    assert isinstance(proj, float)
