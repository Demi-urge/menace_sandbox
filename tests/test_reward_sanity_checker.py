import json
from pathlib import Path

import menace.reward_sanity_checker as rsc


def test_load_recent_rewards_jsonl(tmp_path: Path):
    log = tmp_path / "rewards.jsonl"
    lines = [json.dumps({"reward": v}) for v in [1.0, 2.0, 3.0, 4.0]]
    log.write_text("\n".join(lines))
    rewards = rsc.load_recent_rewards(str(log), window_size=2)
    assert rewards == [3.0, 4.0]


def test_detect_outliers():
    rewards = [1.0, 1.0, 1.0, 50.0]
    out = rsc.detect_outliers(rewards, threshold_stddev=1)
    assert len(out) == 1
    assert out[0][1] == 50.0


def test_check_risk_reward_alignment():
    actions = [
        {"id": 1, "reward": 10.0, "risk_score": 2.0},
        {"id": 2, "reward": 90.0, "risk_score": 8.0},
    ]
    mis = rsc.check_risk_reward_alignment(actions, reward_threshold=80.0, risk_threshold=7.0)
    assert len(mis) == 1
    assert mis[0]["action_id"] == 2


def test_generate_sanity_report(tmp_path: Path):
    outliers = [(0, 100.0)]
    misalign = [{"index": 0, "action_id": 1, "reward": 100.0, "risk": 9.0}]
    report_path = tmp_path / "report.jsonl"
    rsc.generate_sanity_report(outliers, misalign, str(report_path))
    data = report_path.read_text().strip()
    assert data
    rec = json.loads(data)
    assert rec["outliers"]
    assert rec["risk_reward_misalignments"]


def test_is_reward_sane():
    ctx = {"risk_score": 9.0}
    assert not rsc.is_reward_sane(90.0, ctx)
    assert rsc.is_reward_sane(10.0, ctx)
