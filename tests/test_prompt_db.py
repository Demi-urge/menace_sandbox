
import json
import os

from prompt_db import PromptDB
from llm_interface import Prompt, LLMResult, OpenAIProvider, LLMClient
from context_builder_util import create_context_builder


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
            "response_text, model, prompt_tokens, completion_tokens, latency_ms, "
            "input_tokens, output_tokens, cost, backend FROM prompts"
        )
    ).fetchone()
    assert row[0] == "hi"
    assert json.loads(row[1]) == ["ex1"]
    assert json.loads(row[2]) == [0.5]
    assert json.loads(row[3]) == ["test"]
    assert row[4] == "hello"
    assert row[5] == "gpt-test"
    assert row[6:] == (5, 7, 12.5, 5, 7, None, None)


def test_openai_client_logs_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("PROMPT_DB_PATH", str(tmp_path / "prompts.db"))
    monkeypatch.setattr(OpenAIProvider, "generate", LLMClient.generate)
    client = OpenAIProvider(model="gpt-test")

    def fake_generate(self, prompt, *, context_builder):
        return LLMResult(text="world", raw={"choices": [{"message": {"content": "world"}}]})

    monkeypatch.setattr(OpenAIProvider, "_generate", fake_generate)
    prompt = Prompt(text="hi", metadata={"tags": ["t"], "vector_confidences": [0.9]})
    builder = create_context_builder()
    res = client.generate(prompt, context_builder=builder)
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
