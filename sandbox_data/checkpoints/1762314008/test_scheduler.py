import json
import logging
import menace.clipped.scheduler as sched


def test_compute_moving_average_empty():
    assert sched.compute_moving_average([]) == 0.0


def test_compute_moving_average_basic():
    values = [1, 2, 3, 4, 5]
    assert sched.compute_moving_average(values, window=3) == (3 + 4 + 5) / 3


def test_is_balolos_clip():
    clip = {"tags": ["balolos"], "category": "test"}
    assert sched.is_balolos_clip(clip)
    clip = {"tags": [], "category": "balolos"}
    assert sched.is_balolos_clip(clip)
    clip = {"tags": ["other"], "category": "none"}
    assert not sched.is_balolos_clip(clip)


def test_load_accounts_nested(tmp_path):
    data = {
        "accounts": [
            {
                "platform": "yt",
                "username": "u",
                "destination": "",
                "topics": ["t"],
            }
        ]
    }
    (tmp_path / "accounts.json").write_text(json.dumps(data))
    (tmp_path / "clips.json").write_text("[]")
    (tmp_path / "clip_topics.json").write_text("{}")
    sched_inst = sched.Scheduler(
        clips_file=tmp_path / "clips.json",
        topics_file=tmp_path / "clip_topics.json",
        accounts_file=tmp_path / "accounts.json",
        history_file=tmp_path / "hist.json",
    )
    assert sched_inst.accounts == data["accounts"]


def test_schedule_logs_bad_timestamp(tmp_path, caplog):
    clips = {
        "1": {
            "topic": "t",
            "stats": [{"views": 10}],
            "created": "bad-timestamp",
        }
    }
    accounts = {
        "accounts": [
            {"id": "a1", "platform": "yt", "topics": ["t"]}
        ]
    }
    topics = {"t": {}}
    (tmp_path / "clips.json").write_text(json.dumps(clips))
    (tmp_path / "accounts.json").write_text(json.dumps(accounts))
    (tmp_path / "clip_topics.json").write_text(json.dumps(topics))
    caplog.set_level(logging.ERROR)
    sched_inst = sched.Scheduler(
        clips_file=tmp_path / "clips.json",
        topics_file=tmp_path / "clip_topics.json",
        accounts_file=tmp_path / "accounts.json",
        history_file=tmp_path / "hist.json",
    )
    result = sched_inst.run()
    assert result == [{"account": "a1", "clip": "1"}]
    assert "invalid timestamp" in caplog.text


def test_compute_schedule_selects_best(tmp_path):
    clips = {
        "1": {"topic": "t", "stats": [{"views": 1}]},
        "2": {"topic": "t", "stats": [{"views": 10}]},
    }
    accounts = {"accounts": [{"id": "a1", "platform": "yt", "topics": ["t"]}]}
    topics = {"t": {}}
    (tmp_path / "clips.json").write_text(json.dumps(clips))
    (tmp_path / "accounts.json").write_text(json.dumps(accounts))
    (tmp_path / "clip_topics.json").write_text(json.dumps(topics))
    sched_inst = sched.Scheduler(
        clips_file=tmp_path / "clips.json",
        topics_file=tmp_path / "clip_topics.json",
        accounts_file=tmp_path / "accounts.json",
        history_file=tmp_path / "hist.json",
    )
    result = sched_inst.compute_schedule()
    assert result == [{"account": "a1", "clip": "2"}]


def test_run_writes_history(tmp_path):
    clips = {"1": {"topic": "t", "stats": [{"views": 5}]}}
    accounts = {"accounts": [{"id": "a1", "platform": "yt", "topics": ["t"]}]}
    topics = {"t": {}}
    (tmp_path / "clips.json").write_text(json.dumps(clips))
    (tmp_path / "accounts.json").write_text(json.dumps(accounts))
    (tmp_path / "clip_topics.json").write_text(json.dumps(topics))
    hist = tmp_path / "hist.json"
    sched_inst = sched.Scheduler(
        clips_file=tmp_path / "clips.json",
        topics_file=tmp_path / "clip_topics.json",
        accounts_file=tmp_path / "accounts.json",
        history_file=hist,
    )
    result = sched_inst.run()
    assert hist.exists()
    assert json.loads(hist.read_text())["schedule"] == result

