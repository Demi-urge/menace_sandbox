import pytest

from rate_limit import estimate_tokens

pytest.importorskip("tiktoken")


def test_estimate_tokens_known_models():
    assert estimate_tokens("hello world", model="gpt-3.5-turbo") == 2
    assert estimate_tokens("This is a test.", model="gpt-4o") == 5
