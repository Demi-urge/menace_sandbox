import json
import pytest
import workflow_lineage as wl


def test_lineage_handles_bad_specs_and_missing_parents(tmp_path):
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir()

    # valid parent with summary
    (wf_dir / "parent.workflow.json").write_text(
        json.dumps({"metadata": {"workflow_id": "parent", "mutation_description": "root"}})
    )
    (wf_dir / "parent.summary.json").write_text(json.dumps({"roi": 1.0}))

    # child referencing existing parent
    (wf_dir / "child.workflow.json").write_text(
        json.dumps({
            "metadata": {
                "workflow_id": "child",
                "parent_id": "parent",
                "mutation_description": "child",
            }
        })
    )
    (wf_dir / "child.summary.json").write_text(json.dumps({"roi": 2.0}))

    # child referencing missing parent
    (wf_dir / "orphan.workflow.json").write_text(
        json.dumps({"metadata": {"workflow_id": "orphan", "parent_id": "ghost"}})
    )

    # malformed spec files that should be ignored
    (wf_dir / "bad.workflow.json").write_text("{broken")
    (wf_dir / "noid.workflow.json").write_text(json.dumps({"metadata": {"parent_id": "parent"}}))

    specs = list(wl.load_specs(wf_dir))
    ids = {s["workflow_id"] for s in specs}
    assert ids == {"parent", "child", "orphan"}

    child_spec = next(s for s in specs if s["workflow_id"] == "child")
    parent_spec = next(s for s in specs if s["workflow_id"] == "parent")
    assert child_spec["roi_delta"] == pytest.approx(1.0)
    assert parent_spec["roi_delta"] is None

    graph = wl.build_graph(specs)
    if wl._HAS_NX:
        assert list(graph.successors("parent")) == ["child"]
        assert list(graph.successors("ghost")) == ["orphan"]
        assert graph.nodes["parent"]["summary"]["roi"] == 1.0
        assert graph.nodes["child"]["roi_delta"] == pytest.approx(1.0)
    else:
        edges = graph["edges"]
        assert edges["parent"] == ["child"]
        assert edges["ghost"] == ["orphan"]
        assert graph["nodes"]["parent"]["summary"]["roi"] == 1.0
        assert graph["nodes"]["child"]["roi_delta"] == pytest.approx(1.0)

    data = wl.to_json(graph)
    if wl._HAS_NX:
        node_map = {n["id"]: n for n in data["nodes"]}
        assert node_map["child"]["roi_delta"] == pytest.approx(1.0)
    else:
        assert data["nodes"]["child"]["roi_delta"] == pytest.approx(1.0)

    dot = wl.to_graphviz(graph)
    assert 'roi_delta="1.0"' in dot
