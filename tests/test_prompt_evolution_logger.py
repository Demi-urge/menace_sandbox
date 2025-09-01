import json
import threading
from pathlib import Path

from prompt_evolution_logger import PromptEvolutionLogger
from prompt_types import Prompt


class DummyEngine:
    def __init__(self) -> None:
        self.last_metadata = {"tone": "neutral", "structured_sections": ["intro"]}


def test_prompt_evolution_logger_records(tmp_path: Path) -> None:
    logger = PromptEvolutionLogger(
        success_path=tmp_path / "success.jsonl",
        failure_path=tmp_path / "failure.jsonl",
    )
    prompt = Prompt(system="sys", user="usr", examples=["ex"])
    engine = DummyEngine()

    logger.log_success(
        "p1",
        prompt,
        "ok",
        roi_delta=1.5,
        coverage=0.9,
        prompt_engine=engine,
    )
    logger.log_failure(
        "p2",
        prompt,
        "fail",
        roi_delta=-0.3,
        coverage=0.1,
        prompt_engine=engine,
    )

    success_record = json.loads((tmp_path / "success.jsonl").read_text().splitlines()[0])
    failure_record = json.loads((tmp_path / "failure.jsonl").read_text().splitlines()[0])

    assert success_record["prompt"] == {
        "system": "sys",
        "user": "usr",
        "examples": ["ex"],
    }
    assert success_record["metadata"] == engine.last_metadata
    assert success_record["roi"] == {"roi_delta": 1.5, "coverage": 0.9}

    assert failure_record["prompt"]["user"] == "usr"
    assert failure_record["roi"] == {"roi_delta": -0.3, "coverage": 0.1}


def test_prompt_evolution_logger_locking(tmp_path: Path) -> None:
    logger = PromptEvolutionLogger(success_path=tmp_path / "success.jsonl")
    prompt = Prompt(user="hi")

    def worker(i: int) -> None:
        logger.log_success(f"p{i}", prompt, "ok", roi_delta=1.0, coverage=1.0)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = (tmp_path / "success.jsonl").read_text().splitlines()
    assert len(lines) == 10
    for line in lines:
        json.loads(line)
