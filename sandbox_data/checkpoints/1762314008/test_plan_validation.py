import plan_validation as pv


def test_parse_plan_json_and_yaml():
    assert pv.parse_plan('["a", "b"]') == ["a", "b"]
    yaml_plan = "steps:\n  - x\n  - y\n"
    assert pv.parse_plan(yaml_plan) == ["x", "y"]


def test_validate_plan_flags():
    bad_plan = '["use sudo rm -rf /"]'
    result = pv.validate_plan(bad_plan)
    assert isinstance(result, dict)
    assert result["error"] == "plan_contains_violations"


def test_validate_plan_success():
    good_plan = '["echo hello"]'
    assert pv.validate_plan(good_plan) == ["echo hello"]
