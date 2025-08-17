import pytest
from vector_service import ContextBuilder, ErrorResult, VectorServiceError

class DummyBuilder:
    def __init__(self, exc=None):
        self.exc = exc
    def build_context(self, task_description, **kwargs):
        if self.exc:
            raise self.exc
        return "ok"


def test_context_builder_returns_value_error_object():
    cb = ContextBuilder(builder=DummyBuilder(exc=ValueError("bad")))
    res = cb.build("task")
    assert isinstance(res, ErrorResult)
    assert res.error == "value_error"


def test_context_builder_returns_rate_limit_object():
    cb = ContextBuilder(builder=DummyBuilder(exc=Exception("429 rate limit")))
    res = cb.build("task")
    assert isinstance(res, ErrorResult)
    assert res.error == "rate_limited"


def test_context_builder_wraps_other_errors():
    cb = ContextBuilder(builder=DummyBuilder(exc=RuntimeError("boom")))
    with pytest.raises(VectorServiceError):
        cb.build("task")
