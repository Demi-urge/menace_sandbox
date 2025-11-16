from analysis.semantic_diff_filter import find_semantic_risks
from unsafe_patterns import find_matches


def test_find_matches_detects_new_patterns():
    code = """
import pickle
pickle.loads(data)
import yaml
yaml.load(data)
"""
    matches = find_matches(code)
    assert "untrusted pickle load" in matches
    assert "yaml load without safe loader" in matches


def test_semantic_diff_filter_detects_new_patterns():
    lines = ["pickle.loads(data)", "yaml.load(data)"]
    results = find_semantic_risks(lines)
    messages = [m for _, m, _ in results]
    assert any("untrusted pickle load" in m for m in messages)
    assert any("yaml load without safe loader" in m for m in messages)
