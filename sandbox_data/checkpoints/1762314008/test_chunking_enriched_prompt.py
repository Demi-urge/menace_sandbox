import types
import chunking
class StubLLM:
    def __init__(self):
        self.last_prompt = None

    def generate(self, prompt, context_builder=None):
        self.last_prompt = prompt
        return types.SimpleNamespace(text="summary")


class StubBuilder:
    def __init__(self):
        self.prompt_calls = []

    def build_prompt(self, text, *, intent=None, top_k=5, **kwargs):
        self.prompt_calls.append((text, intent, top_k))
        from prompt_types import Prompt

        meta = {
            "vector_confidences": [0.6],
            "vectors": [("code", "v1", 0.6)],
            "retrieval_metadata": {"code": [{"desc": "ctx1", "score": 0.6}]},
        }
        prompt = Prompt(user=text, metadata=meta)
        prompt.vector_confidence = 0.6
        return prompt


def test_summarize_snippet_enriched_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(chunking, "SNIPPET_CACHE_DIR", tmp_path)
    llm = StubLLM()
    builder = StubBuilder()
    result = chunking.summarize_code("print('x')", llm, context_builder=builder)
    assert result == "summary"
    assert builder.prompt_calls[0][0] == "print('x')"
    assert builder.prompt_calls[0][2] == 5
    intent = builder.prompt_calls[0][1]
    assert intent["instruction"].startswith("Summarise")
    prompt = llm.last_prompt
    assert prompt.vector_confidence == 0.6
    assert prompt.metadata["vector_confidences"] == [0.6]
    assert prompt.metadata["vectors"] == [("code", "v1", 0.6)]
    assert prompt.metadata["retrieval_metadata"]["code"][0]["desc"] == "ctx1"
