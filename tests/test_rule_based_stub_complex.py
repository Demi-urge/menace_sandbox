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


class Config:
    def __init__(self, name: str, thresholds: dict[str, list[int]]):
        self.name = name
        self.thresholds = thresholds


def complex_func(configs: list[Config], mapping: dict[str, tuple[int, float]]) -> None:
    """Function with nested and custom class annotations."""


def test_custom_class_and_nested(monkeypatch):
    _disable_generation(monkeypatch)
    stub = gsp.generate_stubs([{}], {"target": complex_func}, context_builder=DummyBuilder())[0]
    assert isinstance(stub["configs"], list)
    assert len(stub["configs"]) == 1
    cfg = stub["configs"][0]
    assert isinstance(cfg, Config)
    assert isinstance(cfg.name, str)
    assert isinstance(cfg.thresholds, dict)
    for k, v in cfg.thresholds.items():
        assert isinstance(k, str)
        assert isinstance(v, list)
        assert all(isinstance(i, int) for i in v)
    mapping = stub["mapping"]
    assert isinstance(mapping, dict)
    for k, v in mapping.items():
        assert isinstance(k, str)
        assert isinstance(v, tuple) and len(v) == 2
        assert isinstance(v[0], int)
        assert isinstance(v[1], float)


def test_defaults_used(monkeypatch):
    _disable_generation(monkeypatch)

    def with_defaults(a: int = 5, flag: bool = False, name: str = "x") -> None:
        pass

    stub = gsp.generate_stubs([{}], {"target": with_defaults}, context_builder=DummyBuilder())[0]
    assert stub == {"a": 5, "flag": False, "name": "x"}
