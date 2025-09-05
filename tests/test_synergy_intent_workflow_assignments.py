from workflow_synthesizer import generate_variants


class SwapGraph:
    def get_synergy_cluster(self, module_name, threshold=0.7, bfs=False):
        return {module_name, f"{module_name}_alt"}


class IntentSuggester:
    class Match:
        def __init__(self, path):
            self.path = path
            self.members = None

    def _search_related(self, prompt, top_k=5):
        return [self.Match("extra_step.py")]  # path-ignore


def test_synergy_and_intent_cluster_assignments():
    base = ["step_a", "step_b"]
    variants = generate_variants(base, 5, SwapGraph(), IntentSuggester())
    assert ["step_a_alt", "step_b"] in variants
    assert any("extra_step" in v for v in variants)
    assert all(len(v) in {2, 3} for v in variants)
