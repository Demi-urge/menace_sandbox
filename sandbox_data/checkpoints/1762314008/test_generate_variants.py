import types
from workflow_synthesizer import generate_variants


class DummySynergy:
    def get_synergy_cluster(self, module_name, threshold=0.7, bfs=False):
        return {module_name, f"{module_name}_alt"}


class DummyIntent:
    class Match:
        def __init__(self, path):
            self.path = path
            self.members = None

    def _search_related(self, prompt, top_k=5):
        return [self.Match("extra_mod.py")]  # path-ignore


def test_generate_variants_basic():
    base = ["foo", "bar"]
    variants = generate_variants(base, 5, DummySynergy(), DummyIntent())
    assert variants
    assert all(isinstance(v, list) for v in variants)
    assert all(tuple(v) != tuple(base) for v in variants)
    assert any("extra_mod" in v for v in variants)
    assert len(variants) <= 5
