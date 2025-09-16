import types
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import llm_config
import rate_limit
from llm_interface import LLMClient, LLMResult, Prompt


def test_clients_share_token_bucket(monkeypatch):
    cfg = types.SimpleNamespace(
        model="m",
        api_key="key",
        max_retries=1,
        tokens_per_minute=100,
    )
    monkeypatch.setattr(llm_config, "get_config", lambda: cfg)

    bucket = rate_limit.SHARED_TOKEN_BUCKET
    bucket.update_rate(100)
    bucket.tokens = 100

    class Dummy(LLMClient):
        def __init__(self, tokens: int):
            super().__init__("dummy", log_prompts=False)
            self._tokens = tokens

        def _generate(self, prompt: Prompt, *, context_builder) -> LLMResult:
            self._rate_limiter.consume(self._tokens)
            return LLMResult(text="", raw={"backend": "dummy"})

    c1 = Dummy(30)
    c2 = Dummy(50)
    builder = types.SimpleNamespace(roi_tracker=None)

    meta = {"vector_confidences": [0.5]}
    c1.generate(
        Prompt("hi", origin="context_builder", metadata=meta), context_builder=builder
    )
    assert bucket.tokens == 70

    c2.generate(
        Prompt("hi", origin="context_builder", metadata=meta), context_builder=builder
    )
    assert bucket.tokens == 20

    assert c1._rate_limiter is bucket and c2._rate_limiter is bucket

    # Reset global bucket to avoid interference with other tests.  Use a large
    # value so subsequent ``update_rate`` calls refill the allowance fully.
    bucket.tokens = 10**9
    bucket.capacity = 10**9

