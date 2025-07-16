from pathlib import Path
import json
from menace.clipped.profit_manager import ProfitManager

def write_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)

def test_balolos_clip_protected(tmp_path):
    clips = {
        "1": {
            "file_size_mb": 5.0,
            "profit": 0.0,
            "account": "acc1",
            "topic": "Balolos",
            "confidence": 80.0,
            "created": "2025-01-01T00:00:00",
        }
    }
    accounts = {"accounts": [{"id": "acc1", "platform": "yt", "topics": ["Balolos"]}]}
    topics = {"Balolos": {}}
    chamber = {"path": "bal.mp4", "duration_min": 1, "file_size_mb": 20, "clip_count": 0}

    write_json(tmp_path / "clips.json", clips)
    write_json(tmp_path / "accounts.json", accounts)
    write_json(tmp_path / "clip_topics.json", topics)
    write_json(tmp_path / "chamber.json", chamber)

    pm = ProfitManager(
        clips_file=tmp_path / "clips.json",
        accounts_file=tmp_path / "accounts.json",
        topics_file=tmp_path / "clip_topics.json",
        chamber_file=tmp_path / "chamber.json",
    )
    pm.run()
    result = json.load(open(tmp_path / "clips.json"))
    assert "1" in result

def test_replacement_on_delete(tmp_path):
    clips = {
        "1": {
            "file_size_mb": 10.0,
            "profit": 0.0,
            "account": "acc1",
            "topic": "tech",
            "confidence": 50.0,
            "created": "2025-01-01T00:00:00",
        },
        "2": {
            "file_size_mb": 10.0,
            "profit": 10.0,
            "account": "acc1",
            "topic": "tech",
            "confidence": 80.0,
            "created": "2025-01-01T00:00:00",
        },
    }
    accounts = {"accounts": [{"id": "acc1", "platform": "yt", "topics": ["tech"]}]}
    topics = {"tech": {}}
    chamber = {"path": "vid.mp4", "duration_min": 1, "file_size_mb": 20, "clip_count": 0}

    write_json(tmp_path / "clips.json", clips)
    write_json(tmp_path / "accounts.json", accounts)
    write_json(tmp_path / "clip_topics.json", topics)
    write_json(tmp_path / "chamber.json", chamber)

    pm = ProfitManager(
        clips_file=tmp_path / "clips.json",
        accounts_file=tmp_path / "accounts.json",
        topics_file=tmp_path / "clip_topics.json",
        chamber_file=tmp_path / "chamber.json",
    )
    deleted = pm.run()
    result = json.load(open(tmp_path / "clips.json"))
    assert "1" not in result
    assert len(result) == 2
    assert "3" in result
    assert "1" in deleted

def test_platform_account_limit(tmp_path):
    accounts = {"accounts": []}
    for i in range(205):
        accounts["accounts"].append({"id": f"a{i}", "platform": "yt", "topics": ["tech"], "confidence": 80.0})
    write_json(tmp_path / "clips.json", {})
    write_json(tmp_path / "accounts.json", accounts)
    write_json(tmp_path / "clip_topics.json", {})
    write_json(tmp_path / "chamber.json", {})

    pm = ProfitManager(
        clips_file=tmp_path / "clips.json",
        accounts_file=tmp_path / "accounts.json",
        topics_file=tmp_path / "clip_topics.json",
        chamber_file=tmp_path / "chamber.json",
    )
    pm.run()
    result = json.load(open(tmp_path / "accounts.json"))
    active = [a for a in result["accounts"] if a.get("platform") == "yt" and a.get("status") != "removed"]
    assert len(active) <= 200
