from llm_interface import LLMClient, LLMResult, Prompt
from prompt_engine import PromptEngine


class SimpleRetriever:
    """Return a single high-confidence record for any query."""

    def search(self, _query: str, top_k: int):  # pragma: no cover - trivial
        return [
            {
                "score": 0.9,
                "metadata": {"summary": "example", "tests_passed": True, "ts": 1},
            }
        ]


class CapturingClient(LLMClient):
    """LLM client that records the prompt it receives."""

    def __init__(self):
        super().__init__("capture", log_prompts=False)
        self.seen: list[Prompt] = []

    def _generate(self, prompt: Prompt) -> LLMResult:
        self.seen.append(prompt)
        # Return JSON to exercise parsed handling
        return LLMResult(text='{"result": "ok"}', parsed={"result": "ok"})


def test_prompt_engine_to_llm_client_flow():
    engine = PromptEngine(
        retriever=SimpleRetriever(),
        patch_retriever=SimpleRetriever(),
        top_n=1,
        confidence_threshold=0,
        context_builder=object(),
    )

    prompt = engine.build_prompt("do things", context_builder=engine.context_builder)
    client = CapturingClient()
    result = client.generate(prompt)

    assert isinstance(prompt, Prompt)
    assert client.seen == [prompt]
    assert result.parsed == {"result": "ok"}
