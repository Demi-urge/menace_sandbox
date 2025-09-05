import workflow_synthesizer as ws
from workflow_synthesizer import generate_workflow_variants


class DummyIntent:
    class Match:
        def __init__(self, path: str) -> None:
            self.path = path
            self.members = None

    def _search_related(self, prompt: str, top_k: int = 5):  # pragma: no cover - simple stub
        return [self.Match("extra_mod.py")]  # path-ignore


def test_generate_workflow_variants_basic(monkeypatch):
    def fake_cluster(mod, threshold=0.7, bfs=False):  # pragma: no cover - used in test
        return {mod, f"{mod}_alt", "bad_mod"}

    monkeypatch.setattr(ws, "get_synergy_cluster", fake_cluster)

    spec = [
        {"module": "foo", "inputs": [], "outputs": ["x"]},
        {"module": "bar", "inputs": ["x"], "outputs": []},
    ]

    def validator(seq):
        return "bad_mod" not in seq

    variants = generate_workflow_variants(
        spec, limit=5, validator=validator, intent_clusterer=DummyIntent()
    )

    assert variants
    assert len(variants) <= 5
    assert all(isinstance(v, list) for v in variants)
    assert all(tuple(v) != ("foo", "bar") for v in variants)
    assert all("bad_mod" not in v for v in variants)
    assert any("extra_mod" in v for v in variants)
    assert not any(v[0] == "bar" and v[1] == "foo" for v in variants)
