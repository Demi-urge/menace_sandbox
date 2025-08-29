import json
from pathlib import Path

import networkx as nx
import pytest

from meta_workflow_planner import MetaWorkflowPlanner


class DummyGraph:
    """Minimal graph wrapper exposing ``graph`` attribute."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()


class DummyROI:
    """Stub ROI database returning no trends."""

    def fetch_trends(self, workflow_id: str):  # pragma: no cover - simple stub
        return []


class StubCodeDatabase:
    """Return predefined categories and tags for modules and functions."""

    def __init__(self, module_meta, function_meta=None):
        self.module_meta = module_meta
        self.function_meta = function_meta or {}

    def get_module_categories(self, module):
        return self.module_meta.get(module, {}).get("categories", [])

    def get_context_tags(self, name):
        if name in self.module_meta:
            return self.module_meta[name].get("tags", [])
        return self.function_meta.get(name, {}).get("tags", [])

    def search(self, name):
        data = self.function_meta.get(name)
        if not data:
            return []
        return [
            {
                "template_type": data.get("template_type"),
                "summary": data.get("summary", ""),
                "context_tags": data.get("tags", []),
                "dependency_depth": data.get("depth", 0.0),
                "branching_factor": data.get("branching", 0.0),
                "roi_curve": data.get("curve", []),
            }
        ]


MODULE_META = {
    "a": {"categories": ["Alpha"], "tags": ["TagA"]},
    "b": {"categories": ["Beta"], "tags": ["TagB"]},
    "c": {"categories": ["Gamma"], "tags": ["TagC"]},
}

FUNCTION_META = {
    "f1": {"tags": ["FuncTag"]},
}


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "workflows"


@pytest.mark.parametrize(
    "workflow_file, expected_modules, expected_tags",
    [
        ("simple_ab.json", ["alpha", "beta"], ["taga", "tagb"]),
        ("simple_bc.json", ["beta", "gamma"], ["tagb", "tagc"]),
    ],
)
def test_code_db_tags_in_embedding(workflow_file, expected_modules, expected_tags):
    planner = MetaWorkflowPlanner(
        graph=DummyGraph(),
        roi_db=DummyROI(),
        code_db=StubCodeDatabase(MODULE_META),
    )
    with (FIXTURES / workflow_file).open() as fh:
        data = json.load(fh)

    workflow = {"workflow": [{"module": step["module"]} for step in data.get("steps", [])]}

    vec = planner.encode(workflow_file, workflow)

    base = 2 + planner.roi_window + 2 + planner.roi_window
    module_start = base + 3 * planner.max_functions
    tag_start = module_start + planner.max_modules

    for mod in expected_modules:
        idx = planner.module_index[mod]
        assert vec[module_start + idx] == 1.0

    for tag in expected_tags:
        idx = planner.tag_index[tag]
        assert vec[tag_start + idx] > 0.0


def test_function_context_tags_in_embedding():
    planner = MetaWorkflowPlanner(
        graph=DummyGraph(),
        roi_db=DummyROI(),
        code_db=StubCodeDatabase(MODULE_META, FUNCTION_META),
    )

    workflow = {"workflow": [{"function": "f1"}]}

    vec = planner.encode_workflow("wf_func", workflow)

    base = 2 + planner.roi_window + 2 + planner.roi_window
    module_start = base + 3 * planner.max_functions
    tag_start = module_start + planner.max_modules

    idx = planner.tag_index["functag"]
    assert vec[tag_start + idx] > 0.0


def test_platform_token_in_embedding():
    planner = MetaWorkflowPlanner(
        graph=DummyGraph(),
        roi_db=DummyROI(),
        code_db=StubCodeDatabase(MODULE_META),
    )
    workflow = {"platform": "YouTube", "workflow": []}
    vec = planner.encode_workflow("wf_platform", workflow)

    base = 2 + planner.roi_window + 2 + planner.roi_window
    module_start = base + 3 * planner.max_functions

    idx = planner.module_index["youtube"]
    assert vec[module_start + idx] == 1.0

