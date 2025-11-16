import sys
import types
import plan_validation as pv


def test_validate_plan_with_forbidden_tokens(monkeypatch):
    """Validation flags plans with forbidden tokens."""
    ethics = types.ModuleType("ethics_violation_detector")
    ethics.flag_violations = lambda data: {"violates_ethics": False}
    monkeypatch.setitem(sys.modules, "ethics_violation_detector", ethics)

    unsafe = types.ModuleType("unsafe_patterns")
    unsafe.find_matches = lambda text: []
    monkeypatch.setitem(sys.modules, "unsafe_patterns", unsafe)

    safe = pv.validate_plan('{"steps": ["check logs"]}')
    assert safe == ["check logs"]

    bad = pv.validate_plan('{"steps": ["sudo rm /"]}')
    assert bad["error"] == "plan_contains_violations"
