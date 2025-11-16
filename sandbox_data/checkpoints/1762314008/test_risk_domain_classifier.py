import pytest

from menace import risk_domain_classifier as rdc


def test_get_domain_risk():
    assert rdc.get_domain_risk("military") == 5
    assert rdc.get_domain_risk("pharma") == 4
    assert rdc.get_domain_risk("unknown") == 0


def test_is_forbidden_domain():
    assert rdc.is_forbidden_domain("military")
    assert not rdc.is_forbidden_domain("politics")


def test_classify_action():
    entry = {"target_domain": "pharma"}
    result = rdc.classify_action(entry)
    assert result == {"domain": "pharma", "risk_score": 4, "forbidden": False}
    assert rdc.classify_action({"target_domain": "military"})["forbidden"]
