import pytest

from vector_service.exceptions import VectorServiceError
from vector_service.retriever import Retriever


class _DummyBuilder:
    roi_tag_penalties = {}


class _DummyRetriever:
    def __init__(self) -> None:
        self.search_called = False

    def retrieve(self, *_args, **_kwargs):
        raise AttributeError("retrieve boom")

    def search(self, *_args, **_kwargs):
        self.search_called = True
        raise AssertionError("search should not be called")


def test_retriever_attribute_error_surfaces_as_vector_service_error():
    retriever = _DummyRetriever()
    wrapper = Retriever(
        context_builder=_DummyBuilder(),
        retriever=retriever,
        cache=None,
    )

    with pytest.raises(VectorServiceError):
        wrapper.search("query", dbs=["code"])

    assert retriever.search_called is False
