import json
from pathlib import Path

from preset_logger import PresetLogger


def test_preset_logger(tmp_path: Path) -> None:
    log_file = tmp_path / "pre.jsonl"
    logger = PresetLogger(log_file)
    logger.log(1, "static file", ["a", "b"])
    logger.log(2, "RL agent", [])
    logger.close()

    entries = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
    assert entries[0]["run"] == 1
    assert entries[0]["preset_source"] == "static file"
    assert entries[0]["actions"] == ["a", "b"]
    assert entries[1]["preset_source"] == "RL agent"
