import json
import workflow_lineage as wl


def test_lineage_handles_bad_specs_and_missing_parents(tmp_path):
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir()

    # valid parent with summary
    (wf_dir / "parent.workflow.json").write_text(
        json.dumps({"metadata": {"workflow_id": "parent", "mutation_description": "root"}})
    )
    (wf_dir / "parent.summary.json").write_text(json.dumps({"roi": 1.23}))

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

    graph = wl.build_graph(specs)
    if wl._HAS_NX:
        assert list(graph.successors("parent")) == ["child"]
        assert list(graph.successors("ghost")) == ["orphan"]
        assert graph.nodes["parent"]["summary"]["roi"] == 1.23
    else:
        edges = graph["edges"]
        assert edges["parent"] == ["child"]
        assert edges["ghost"] == ["orphan"]
        assert graph["nodes"]["parent"]["summary"]["roi"] == 1.23
