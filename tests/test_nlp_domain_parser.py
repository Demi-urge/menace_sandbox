import menace.nlp_domain_parser as ndp


def test_classify_basic():
    ndp.load_model()
    res = ndp.classify_text("military operations")
    assert res
    assert res[0][0] == "military"
    assert ndp.flag_if_similar("military base", threshold=0.5)


def test_classify_entry():
    entry = {"target_domain": "military", "action_description": "strategy"}
    res = ndp.classify_text(entry)
    assert res
    assert res[0][0] == "military"

