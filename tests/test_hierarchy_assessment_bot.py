import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import json
import menace.hierarchy_assessment_bot as hab


def test_redundancy_analysis(monkeypatch):
    bot = hab.HierarchyAssessmentBot(planning_api="http://x")
    bot.register("a", "t1")
    bot.register("b", "t1")
    sent = {}

    def fake_post(url, json=None, timeout=3):
        sent['data'] = json

    monkeypatch.setattr(hab.requests, "post", fake_post)
    overlaps = bot.redundancy_analysis()
    assert overlaps == ["t1"]
    assert sent['data'] == {"overlaps": ["t1"]}


def test_complete_triggers(monkeypatch):
    bot = hab.HierarchyAssessmentBot()
    bot.register("a", "t1")
    captured = {}

    def fake_send_json(msg, flags=0):
        captured['msg'] = msg

    monkeypatch.setattr(bot.socket, "send_json", fake_send_json)
    bot.complete("a", "t1")
    assert captured['msg'] == {"bot": "a", "task": "t1"}


def test_monitor_system(monkeypatch):
    bot = hab.HierarchyAssessmentBot()
    called = False

    def fake_publish(exc, queue, body):
        nonlocal called
        called = True

    if bot.channel:
        monkeypatch.setattr(bot.channel, "basic_publish", fake_publish)
    monkeypatch.setattr(hab.psutil, "cpu_percent", lambda: 95.0)
    triggered = bot.monitor_system(limit=90.0)
    assert triggered
    if bot.channel:
        assert called
