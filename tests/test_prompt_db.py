import json
import os

from prompt_db import PromptDB
from llm_interface import Prompt, LLMResult
from openai_client import OpenAILLMClient


def test_log(tmp_path):
    os.environ["PROMPT_DB_PATH"] = str(tmp_path / "prompts.db")
    db = PromptDB(model="gpt-test")
    prompt = Prompt(
        text="hi", examples=["ex1"], outcome_tags=["test"], vector_confidences=[0.5]
    )
    result = LLMResult(
        raw={"data": 1},
        text="hello",
        prompt_tokens=5,
        completion_tokens=7,
        latency_ms=12.5,
    )
    db.log(prompt, result)
    row = db.conn.execute(
        (
            "SELECT text, examples, vector_confidences, outcome_tags, "
            "response_text, model, prompt_tokens, completion_tokens, latency_ms "
            "FROM prompts"
        )
    ).fetchone()
    assert row[0] == "hi"
    assert json.loads(row[1]) == ["ex1"]
    assert json.loads(row[2]) == [0.5]
    assert json.loads(row[3]) == ["test"]
    assert row[4] == "hello"
    assert row[5] == "gpt-test"
    assert row[6:] == (5, 7, 12.5)


def test_openai_client_logs_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("PROMPT_DB_PATH", str(tmp_path / "prompts.db"))
    client = OpenAILLMClient(model="gpt-test")

    def fake_request(self, payload):
        return {"choices": [{"message": {"content": "world"}}]}

    monkeypatch.setattr(OpenAILLMClient, "_request", fake_request)
    prompt = Prompt(text="hi", metadata={"tags": ["t"], "vector_confidences": [0.9]})
    res = client.generate(prompt)
    assert res.text == "world"
    row = client.db.conn.execute(
        "SELECT text, outcome_tags, vector_confidences, response_text, model "
        "FROM prompts",
    ).fetchone()
    assert row[0] == "hi"
    assert json.loads(row[1]) == ["t"]
    assert json.loads(row[2]) == [0.9]
    assert row[3] == "world"
    assert row[4] == "gpt-test"
