import json
import time

import menace.central_evaluation_loop as cel


def _audit_path(tmp_path):
    return tmp_path / f"audit_{time.strftime('%Y-%m-%d')}.jsonl"


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(cel, "AUDIT_DIR", str(tmp_path))
    monkeypatch.setattr(cel.reward_dispatcher, "dispatch_reward", lambda _action: None)
    monkeypatch.setattr(cel, "TRUTH_ADAPTER", None)
    import menace.governance as gov

    class _DummyTracker:
        def calculate_raroi(self, roi, rollback_prob=0.0, metrics=None):
            return roi, roi, {}

    monkeypatch.setattr(gov, "ROITracker", _DummyTracker)


def _read_last(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.loads(fh.readlines()[-1])


def test_roi_veto(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    action = {
        "domain": "social_media",
        "risk_score": 0.0,
        "alignment_score": 1.0,
        "metrics": {"security": 0.9},
        "flags": {"alignment_violation": True},
        "roi_profile": "scraper_bot",
    }
    ok = cel.process_action(json.dumps(action))
    assert not ok
    record = _read_last(_audit_path(tmp_path))
    assert record["error"] == "roi_veto"


def test_roi_logged_without_veto(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    action = {
        "domain": "social_media",
        "risk_score": 0.1,
        "alignment_score": 1.0,
        "metrics": {
            "profitability": 0.5,
            "efficiency": 0.6,
            "reliability": 0.7,
            "resilience": 0.8,
            "maintainability": 0.9,
            "security": 0.9,
            "latency": 0.1,
            "energy": 0.1,
        },
        "roi_profile": "scraper_bot",
    }
    ok = cel.process_action(json.dumps(action))
    assert ok
    record = _read_last(_audit_path(tmp_path))
    assert "error" not in record
    assert isinstance(record.get("roi"), float)


def test_alignment_veto(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    action = {
        "decision": "ship",
        "alignment_status": "fail",
        "metrics": {
            "profitability": 0.5,
            "efficiency": 0.6,
            "reliability": 0.7,
            "resilience": 0.8,
            "maintainability": 0.9,
            "security": 0.9,
            "latency": 0.1,
            "energy": 0.1,
        },
        "roi_profile": "scraper_bot",
    }
    ok = cel.process_action(json.dumps(action))
    assert not ok
    record = _read_last(_audit_path(tmp_path))
    assert record["error"] == "governance_veto"
    assert record["decision"] == "ship"


def test_raroi_increase_veto(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    scorecard = {
        "scenarios": {
            "normal": {"roi": 1.0},
            "a": {"roi": 2.0},
            "b": {"roi": 3.0},
            "c": {"roi": 4.0},
        }
    }
    action = {
        "decision": "rollback",
        "alignment_status": "pass",
        "scorecard": scorecard,
        "metrics": {
            "profitability": 0.5,
            "efficiency": 0.6,
            "reliability": 0.7,
            "resilience": 0.8,
            "maintainability": 0.9,
            "security": 0.9,
            "latency": 0.1,
            "energy": 0.1,
        },
        "roi_profile": "scraper_bot",
    }
    ok = cel.process_action(json.dumps(action))
    assert not ok
    record = _read_last(_audit_path(tmp_path))
    assert record["error"] == "governance_veto"
    assert record["decision"] == "rollback"
