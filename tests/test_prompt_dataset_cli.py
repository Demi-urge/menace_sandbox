import json
import os
import sys

from prompt_db import PromptDB
from llm_interface import Prompt, LLMResult
from tools.prompt_dataset_cli import main


def _seed_db(path):
    os.environ["PROMPT_DB_PATH"] = str(path)
    db = PromptDB(model="test")
    db.log(
        Prompt(text="hi", outcome_tags=["keep"], vector_confidence=0.9),
        LLMResult(text="hello"),
    )
    db.log(
        Prompt(text="bye", outcome_tags=["drop"], vector_confidence=0.1),
        LLMResult(text="later"),
    )


def test_jsonl_and_csv(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "prompts.db"
    _seed_db(db_path)

    out_jsonl = tmp_path / "out.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", str(out_jsonl), "--tag", "keep", "--min-confidence", "0.5"],
    )
    main()
    data = out_jsonl.read_text().strip().splitlines()
    assert len(data) == 1
    rec = json.loads(data[0])
    assert rec["prompt"] == "hi"
    assert rec["completion"] == "hello"
    assert json.loads(capsys.readouterr().out)["written"] == 1

    out_csv = tmp_path / "out.csv"
    monkeypatch.setattr(sys, "argv", ["prog", str(out_csv), "--format", "csv"])
    main()
    csv_lines = out_csv.read_text().strip().splitlines()
    assert csv_lines[0] == "prompt,completion"
    assert any("hi" in line for line in csv_lines[1:])
