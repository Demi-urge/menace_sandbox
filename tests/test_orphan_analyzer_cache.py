import networkx as nx

import orphan_analyzer
from module_graph_analyzer import DEFAULT_IGNORED_DIRS


def test_redundant_classifier_uses_cached_graph_and_ignores(tmp_path, monkeypatch):
    module_path = tmp_path / "orphan.py"
    module_path.write_text("")  # path-ignore
    metrics = {"functions": 0, "complexity": 0, "calls": 0, "docstring": False}
    calls: list[dict[str, object]] = []

    def fake_build_graph(root, *, ignore=None, max_file_size_bytes=None, parse_timeout_s=None):
        calls.append(
            {
                "root": root,
                "ignore": tuple(ignore) if ignore is not None else None,
                "max_file_size_bytes": max_file_size_bytes,
                "parse_timeout_s": parse_timeout_s,
            }
        )
        graph = nx.DiGraph()
        graph.graph["build_complete"] = True
        return graph

    monkeypatch.setattr(orphan_analyzer, "build_import_graph", fake_build_graph)
    orphan_analyzer._cached_import_graph_by_root.cache_clear()

    assert orphan_analyzer._redundant_classifier(module_path, metrics) == "redundant"
    assert orphan_analyzer._redundant_classifier(module_path, metrics) == "redundant"

    assert len(calls) == 1
    assert calls[0]["ignore"] is not None
    assert set(DEFAULT_IGNORED_DIRS).issubset(set(calls[0]["ignore"]))
    assert calls[0]["max_file_size_bytes"] == orphan_analyzer.DEFAULT_GRAPH_MAX_FILE_SIZE_BYTES
    assert calls[0]["parse_timeout_s"] == orphan_analyzer.DEFAULT_GRAPH_PARSE_TIMEOUT_S
