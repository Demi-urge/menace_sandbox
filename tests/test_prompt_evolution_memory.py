import json
import threading
from pathlib import Path

from prompt_evolution_memory import PromptEvolutionMemory
from llm_interface import Prompt


def test_prompt_evolution_memory_records(tmp_path: Path) -> None:
    logger = PromptEvolutionMemory(
        success_path=tmp_path / "success.json",
        failure_path=tmp_path / "failure.json",
    )
    prompt = Prompt(
        system="sys",
        user="usr",
        examples=["ex"],
        metadata={"tone": "neutral", "structured_sections": ["intro"]},
    )

    logger.log(prompt, True, {"summary": "ok"}, {"roi_delta": 1.5, "coverage": 0.9})
    logger.log(prompt, False, {"summary": "fail"}, {"roi_delta": -0.3, "coverage": 0.1})

    success_record = json.loads((tmp_path / "success.json").read_text().splitlines()[0])
    failure_record = json.loads((tmp_path / "failure.json").read_text().splitlines()[0])

    assert success_record["prompt"] == {
        "system": "sys",
        "user": "usr",
        "examples": ["ex"],
        "metadata": {"tone": "neutral", "structured_sections": ["intro"]},
    }
    assert success_record["roi"] == {"roi_delta": 1.5, "coverage": 0.9}

    assert failure_record["prompt"]["user"] == "usr"
    assert failure_record["roi"] == {"roi_delta": -0.3, "coverage": 0.1}


def test_prompt_evolution_memory_locking(tmp_path: Path) -> None:
    logger = PromptEvolutionMemory(success_path=tmp_path / "success.json")
    prompt = Prompt(user="hi")

    def worker(i: int) -> None:
        logger.log(prompt, True, {"i": i}, {"roi_delta": 1.0, "coverage": 1.0})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = (tmp_path / "success.json").read_text().splitlines()
    assert len(lines) == 10
    for line in lines:
        json.loads(line)
