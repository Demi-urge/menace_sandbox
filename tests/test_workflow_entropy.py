import math

from menace.workflow_metrics import compute_workflow_entropy
from menace.workflow_synergy_comparator import WorkflowSynergyComparator
from pathlib import Path


def test_compute_workflow_entropy_simple_spec():
    spec = {
        "steps": [
            {"module": "a"},
            {"module": "a"},
            {"module": "b"},
        ]
    }
    ent = compute_workflow_entropy(spec)
    expected = -(2/3)*math.log2(2/3)-(1/3)*math.log2(1/3)
    assert abs(ent - expected) < 1e-9


def test_workflow_synergy_comparator_uses_entropy():
    spec = {
        "steps": [
            {"module": "x"},
            {"module": "y"},
        ]
    }
    WorkflowSynergyComparator.best_practices_file = Path("/tmp/wsc_best.json")
    result = WorkflowSynergyComparator.compare(spec, spec)
    expected_entropy = compute_workflow_entropy(spec)
    assert result.entropy_a == expected_entropy
    assert result.entropy_b == expected_entropy
