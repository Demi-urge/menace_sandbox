import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.prompt_cache import TriggerPromptCache, FewShotPromptEngine


def test_cache_rollover():
    cache = TriggerPromptCache(max_prompts=3)
    for i in range(5):
        cache.add("scarcity", f"p{i}")
    items = cache.get("scarcity")
    assert len(items) == 3
    assert items[0] == "p2"


def test_seed_new_slots_and_variants():
    engine = FewShotPromptEngine(threshold=0.2, max_prompts=3)
    labels = engine.seed_new_slots(["buy coins now", "limited offer", "join tribe", "exclusive access"])
    assert labels
    for label in labels:
        assert label in engine.classifier.categories
    engine.ingest(labels[0], "special promo")
    vars = engine.generate_variants(labels[0], ["crypto"])
    assert vars
