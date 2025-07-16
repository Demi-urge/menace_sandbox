import menace.indirect_domain_reference_detector as idrd


def test_detect_direct_phrase():
    text = "The project involves ballistic analysis and other tasks"
    assert "military" in idrd.detect_indirect_domains(text)


def test_detect_fuzzy_phrase():
    text = "This includes ballistik analysis of new systems"
    assert "military" in idrd.detect_indirect_domains(text)


def test_flag_indirect_risk():
    entry = {"target_domain": "unknown", "action_description": "studying genetic optimization for treatments"}
    res = idrd.flag_indirect_risk(entry)
    assert "pharma" in res["matched_domains"]
    assert any("genetic optimization" in p for p in res["matched_phrases"])
    assert res["risk_bonus"] > 0
