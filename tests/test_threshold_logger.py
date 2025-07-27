import json
from pathlib import Path
from threshold_logger import ThresholdLogger

def test_threshold_logger(tmp_path: Path) -> None:
    log_file = tmp_path / "thr.jsonl"
    logger = ThresholdLogger(log_file)
    logger.log(1, 0.1, 0.2, True)
    logger.close()
    data = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
    assert data and data[0]["run"] == 1
    assert data[0]["roi_threshold"] == 0.1
    assert data[0]["synergy_threshold"] == 0.2
    assert data[0]["converged"] is True
