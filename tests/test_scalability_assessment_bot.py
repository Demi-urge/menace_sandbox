import json
import types
import menace.scalability_assessment_bot as sab

BLUEPRINT = json.dumps({"tasks": [{"name": "a"}, {"name": "b"}]})


def test_parse_blueprint():
    bot = sab.ScalabilityAssessmentBot()
    bp = bot.parse_blueprint(BLUEPRINT)
    assert [t["name"] for t in bp.get("tasks", [])] == ["a", "b"]


def test_analyse_and_send(monkeypatch):
    sent = {}

    def fake_post(url, json=None, timeout=3):
        sent["data"] = json

    router = types.SimpleNamespace(terms=[])
    router.query_all = lambda term: router.terms.append(term) or {}
    bot = sab.ScalabilityAssessmentBot(rp_url="http://x", db_router=router)
    if sab.requests:
        monkeypatch.setattr(sab.requests, "post", fake_post)
    report = bot.analyse(BLUEPRINT)
    bot.send_report(report)
    if sab.requests:
        assert sent["data"]["bottlenecks"] == []
    assert "scalability" in router.terms
