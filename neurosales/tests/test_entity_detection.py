import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.entity_detection import EntityDetector


def test_basic_detection_with_coref():
    det = EntityDetector()
    text = "Elon Musk founded SpaceX. He also leads Tesla."
    ents = det.detect(text)
    names = [e.canonical for e in ents]
    assert "elon musk" in names
    # pronoun coreference should map "He" to elon musk
    assert any(e.canonical == "elon musk" and e.text.lower() == "he" for e in ents)


def test_fuzzy_matching():
    det = EntityDetector()
    det.detect("Apple released a new phone")
    match = det.fuzzy_match("appel")
    assert match and match.canonical == "apple"


def test_unsupervised_expansion():
    det = EntityDetector()
    det.detect("Met Foobaz yesterday")
    det.detect("Foobaz was excited")
    det.detect("Discussed with Foobaz again")
    added = det.expand_unknowns(min_count=2)
    assert "foobaz" in added
    assert det.entities["foobaz"].label != ""
