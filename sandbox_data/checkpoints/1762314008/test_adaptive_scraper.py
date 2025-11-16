import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.adaptive_scraper import AdaptiveWebScraper


def test_novel_terms_expand_depth_and_keywords():
    scraper = AdaptiveWebScraper(["brain", "neuron"], crawl_depth=1)
    scraper.entropy = scraper._entropy(scraper._tokenize("brain neuron study"))
    new_abs = ["Study explores insula-valence coupling in detail"]
    scraper.nightly_update("s1", new_abs, [0.2])
    assert scraper.crawl_depth > 1
    assert any("insula" in kw for kw in scraper.keywords)
    assert scraper.source_priority["s1"] > 1.0


def test_low_reward_decreases_priority():
    scraper = AdaptiveWebScraper(["brain"], crawl_depth=1)
    scraper.entropy = scraper._entropy(scraper._tokenize("brain study"))
    scraper.nightly_update("s2", ["boring text"], [0.0])
    assert scraper.source_priority["s2"] < 1.0
