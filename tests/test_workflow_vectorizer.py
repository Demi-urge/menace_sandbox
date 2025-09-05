import json
from pathlib import Path

import json

import pytest

from workflow_vectorizer import WorkflowVectorizer
from dynamic_path_router import resolve_path


@pytest.fixture
def sample_workflows():
    with resolve_path("tests/fixtures/workflow_examples.json").open() as fh:
        return json.load(fh)


def test_unseen_category_and_status(sample_workflows):
    vec = WorkflowVectorizer().fit(sample_workflows)
    dim = vec.dim
    cat_len = len(vec.category_index)
    status_len = len(vec.status_index)
    wf = {"category": "marketing", "status": "queued", "workflow": ["a"]}
    out = vec.transform(wf)
    assert len(out) == dim == vec.dim
    assert len(vec.category_index) == cat_len + 1
    assert len(vec.status_index) == status_len + 1
    struct_len = 6 + vec.roi_window
    cat_slice = out[struct_len: struct_len + vec.max_categories]
    status_slice = out[
        struct_len + vec.max_categories: struct_len + vec.max_categories + vec.max_status
    ]
    assert sum(cat_slice) == 1.0
    assert sum(status_slice) == 1.0
    assert cat_slice[vec.category_index["marketing"]] == 1.0
    assert status_slice[vec.status_index["queued"]] == 1.0


def test_category_status_overflow_defaults_to_other(sample_workflows):
    vec = WorkflowVectorizer(max_categories=1, max_status=1).fit(sample_workflows)
    dim = vec.dim
    wf = {"category": "marketing", "status": "queued", "workflow": []}
    out = vec.transform(wf)
    assert len(out) == dim == vec.dim
    struct_len = 6 + vec.roi_window
    cat_slice = out[struct_len: struct_len + vec.max_categories]
    status_slice = out[
        struct_len + vec.max_categories: struct_len + vec.max_categories + vec.max_status
    ]
    assert cat_slice == [1.0]
    assert status_slice == [1.0]
    assert vec.category_index == {"other": 0}
    assert vec.status_index == {"other": 0}
