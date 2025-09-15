from dataclasses import dataclass
import sys
import types

th_stub = types.ModuleType("sandbox_runner.test_harness")
th_stub.run_tests = lambda *a, **k: None
th_stub.TestHarnessResult = object
sys.modules.setdefault("sandbox_runner.test_harness", th_stub)
import sandbox_runner.generative_stub_provider as gsp


def _disable_generation(monkeypatch):
    async def _noop():
        return None

    monkeypatch.setattr(gsp, "_aload_generator", _noop)
    gsp._CACHE.clear()


class DummyBuilder:
    def build_prompt(self, query, *, intent_metadata=None, **kwargs):
        return query


def test_docstring_samples(monkeypatch):
    """Values from docstrings are used when available."""

    def sample(name: str, count: int) -> None:
        """Docstring with examples.

        Args:
            name: user name, e.g. 'widget'.
            count: number of widgets, e.g. 5.
        """

    _disable_generation(monkeypatch)
    stubs = gsp.generate_stubs([{}], {"target": sample}, context_builder=DummyBuilder())
    assert stubs == [{"name": "widget", "count": 5}]
    assert gsp._type_matches(stubs[0]["name"], str)
    assert gsp._type_matches(stubs[0]["count"], int)


@dataclass
class User:
    id: int
    tags: list[str]


def test_nested_and_dataclass(monkeypatch):
    """Container types and dataclasses receive structured defaults."""

    def process(user: User, data: dict[str, list[int]]) -> None:
        """Process a user.

        Args:
            data: mapping of values, e.g. {'nums': [1, 2]}.
        """

    _disable_generation(monkeypatch)
    stubs = gsp.generate_stubs([{}], {"target": process}, context_builder=DummyBuilder())
    stub = stubs[0]
    assert isinstance(stub["user"], User)
    assert isinstance(stub["user"].id, int)
    assert isinstance(stub["user"].tags, list)
    assert all(isinstance(t, str) for t in stub["user"].tags)
    assert stub["data"] == {"nums": [1, 2]}
    assert gsp._type_matches(stub["data"], dict[str, list[int]])
