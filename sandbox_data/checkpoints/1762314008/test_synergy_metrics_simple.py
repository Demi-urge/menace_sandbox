import math
import pytest
from pathlib import Path

from menace_sandbox import workflow_synergy_comparator as wsc


def _mock_optional(monkeypatch):
    monkeypatch.setattr(wsc, "_HAS_NX", False, raising=False)
    monkeypatch.setattr(wsc, "WorkflowVectorizer", None, raising=False)
    monkeypatch.setattr(wsc, "ROITracker", None, raising=False)
    monkeypatch.setattr(
        wsc.WorkflowSynergyComparator,
        "best_practices_file",
        Path("/tmp/wsc_best.json"),
        raising=False,
    )


def test_similarity_and_entropy(monkeypatch):
    _mock_optional(monkeypatch)
    spec_a = {"steps": [{"module": "a"}, {"module": "b"}, {"module": "c"}]}
    spec_b = {
        "steps": [
            {"module": "a"},
            {"module": "b"},
            {"module": "a"},
            {"module": "b"},
        ]
    }
    scores = wsc.WorkflowSynergyComparator.compare(spec_a, spec_b)
    expected_sim = 4 / (math.sqrt(3) * math.sqrt(8))
    assert scores.similarity == pytest.approx(expected_sim)
    assert scores.entropy_a == pytest.approx(math.log2(3))
    assert scores.entropy_b == pytest.approx(1.0)


def test_duplicate_detection_thresholds(monkeypatch):
    _mock_optional(monkeypatch)
    spec_a = {"steps": [{"module": "x"}, {"module": "x"}]}
    spec_b = {"steps": [{"module": "x"}, {"module": "y"}]}
    scores = wsc.WorkflowSynergyComparator.compare(spec_a, spec_b)
    assert not wsc.WorkflowSynergyComparator.is_duplicate(scores)
    assert wsc.WorkflowSynergyComparator.is_duplicate(
        scores, similarity_threshold=0.7, entropy_threshold=1.0
    )
    # direct specification invocation
    assert not wsc.WorkflowSynergyComparator.is_duplicate(spec_a, spec_b)
    assert wsc.WorkflowSynergyComparator.is_duplicate(
        spec_a,
        spec_b,
        similarity_threshold=0.7,
        entropy_threshold=1.0,
    )
