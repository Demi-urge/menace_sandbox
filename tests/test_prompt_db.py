import json
import os

from prompt_db import PromptDB
from llm_interface import Prompt, LLMResult
from openai_client import OpenAILLMClient


def test_log_prompt(tmp_path):
    os.environ["PROMPT_DB_PATH"] = str(tmp_path / "prompts.db")
    db = PromptDB(model="gpt-test")
    prompt = Prompt(text="hi", examples=["ex1"])
    result = LLMResult(raw={"data": 1}, text="hello")
    db.log_prompt(prompt, result, ["test"], 0.5)
    row = db.conn.execute(
        "SELECT text, examples, confidence, tags, response_text, model FROM prompts"
    ).fetchone()
    assert row[0] == "hi"
    assert json.loads(row[1]) == ["ex1"]
    assert row[2] == 0.5
    assert json.loads(row[3]) == ["test"]
    assert row[4] == "hello"
    assert row[5] == "gpt-test"


def test_openai_client_logs_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("PROMPT_DB_PATH", str(tmp_path / "prompts.db"))
    client = OpenAILLMClient(model="gpt-test")

    def fake_request(self, payload):
        return {"choices": [{"message": {"content": "world"}}]}

    monkeypatch.setattr(OpenAILLMClient, "_request", fake_request)
    prompt = Prompt(text="hi", metadata={"tags": ["t"], "confidence": 0.9})
    res = client.generate(prompt)
    assert res.text == "world"
    row = client.db.conn.execute(
        "SELECT text, tags, confidence, response_text, model FROM prompts"
    ).fetchone()
    assert row[0] == "hi"
    assert json.loads(row[1]) == ["t"]
    assert row[2] == 0.9
    assert row[3] == "world"
    assert row[4] == "gpt-test"
