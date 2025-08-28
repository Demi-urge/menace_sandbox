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
    """Return predefined categories and tags for modules."""

    def __init__(self, data):
        self.data = data

    def get_module_categories(self, module):
        return self.data.get(module, {}).get("categories", [])

    def get_context_tags(self, module):
        return self.data.get(module, {}).get("tags", [])


MODULE_META = {
    "a": {"categories": ["Alpha"], "tags": ["TagA"]},
    "b": {"categories": ["Beta"], "tags": ["TagB"]},
    "c": {"categories": ["Gamma"], "tags": ["TagC"]},
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
    module_start = base + planner.max_functions
    tag_start = module_start + planner.max_modules

    for mod in expected_modules:
        idx = planner.module_index[mod]
        assert vec[module_start + idx] == 1.0

    for tag in expected_tags:
        idx = planner.tag_index[tag]
        assert vec[tag_start + idx] == 1.0

