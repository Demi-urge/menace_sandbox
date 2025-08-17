import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

fake_qfe = types.ModuleType("quick_fix_engine")
fake_qfe.generate_patch = lambda path: 1
sys.modules["quick_fix_engine"] = fake_qfe

import menace_sandbox.module_retirement_service as module_retirement_service
import menace_sandbox.relevancy_radar as relevancy_radar
import menace_sandbox.relevancy_radar_service as relevancy_radar_service


def test_service_scan_updates_flags(monkeypatch, tmp_path):
    import menace_sandbox.metrics_exporter as metrics_exporter
    import menace_sandbox.module_graph_analyzer as module_graph_analyzer
    import menace_sandbox.relevancy_metrics_db as relevancy_metrics_db

    captured = {}

    monkeypatch.setattr(
        metrics_exporter, "update_relevancy_metrics", lambda flags: captured.update(flags=flags)
    )

    class DummyGraph:
        nodes = ["demo"]

    monkeypatch.setattr(module_graph_analyzer, "build_import_graph", lambda root: DummyGraph())
    monkeypatch.setattr(relevancy_radar, "load_usage_stats", lambda: {})

    class DummyDB:
        def __init__(self, path):
            pass

        def get_roi_deltas(self, modules):
            return {}

    monkeypatch.setattr(relevancy_metrics_db, "RelevancyMetricsDB", DummyDB)

    def fake_eval(self, compress, replace, *, graph, core_modules=None):
        return {"demo": "retire"}

    monkeypatch.setattr(
        relevancy_radar.RelevancyRadar,
        "evaluate_final_contribution",
        fake_eval,
    )

    class DummyRetirementService:
        flags = None

        def __init__(self, root):
            pass

        def process_flags(self, flags):
            DummyRetirementService.flags = flags
            return {k: "retired" for k in flags}

    monkeypatch.setattr(
        module_retirement_service, "ModuleRetirementService", DummyRetirementService
    )

    service = relevancy_radar_service.RelevancyRadarService(tmp_path, interval=0)

    service._scan_once()

    assert service.flags() == {"demo": "retire"}
    assert captured["flags"] == {"demo": "retire"}
    assert DummyRetirementService.flags == {"demo": "retire"}


def test_start_runs_initial_scan(monkeypatch, tmp_path):
    service = relevancy_radar_service.RelevancyRadarService(tmp_path)

    calls = {"count": 0}

    def fake_scan():
        calls["count"] += 1
        service.latest_flags = {"init": "flag"}

    monkeypatch.setattr(service, "_scan_once", fake_scan)

    service.start()
    service.stop()

    assert calls["count"] == 1
    assert service.flags() == {"init": "flag"}


def test_start_populates_latest_flags(monkeypatch, tmp_path):
    """start() triggers an immediate scan that updates latest_flags."""

    import menace_sandbox.module_graph_analyzer as module_graph_analyzer
    import menace_sandbox.relevancy_metrics_db as relevancy_metrics_db

    class DummyGraph:
        nodes = ["sample"]

    monkeypatch.setattr(module_graph_analyzer, "build_import_graph", lambda root: DummyGraph())
    monkeypatch.setattr(relevancy_radar, "load_usage_stats", lambda: {})

    class DummyDB:
        def __init__(self, path):
            pass

        def get_roi_deltas(self, modules):
            return {}

    monkeypatch.setattr(relevancy_metrics_db, "RelevancyMetricsDB", DummyDB)

    def fake_eval(self, compress, replace, *, graph, core_modules=None):
        return {"sample": "retire"}

    monkeypatch.setattr(
        relevancy_radar.RelevancyRadar,
        "evaluate_final_contribution",
        fake_eval,
    )

    class DummyRetirementService:
        def __init__(self, root):
            pass

        def process_flags(self, flags):
            return flags

    monkeypatch.setattr(
        module_retirement_service, "ModuleRetirementService", DummyRetirementService
    )

    service = relevancy_radar_service.RelevancyRadarService(tmp_path)

    service.start()
    service.stop()

    assert service.latest_flags == {"sample": "retire"}


def test_dependency_chain_not_flagged(monkeypatch, tmp_path):
    """Dependencies of used modules are ignored by evaluate_final_contribution."""

    # Create a small repository layout where ``core`` imports ``helper`` and an
    # additional module ``loner`` is unused.
    (tmp_path / "core.py").write_text("import helper\n")
    (tmp_path / "helper.py").write_text("\n")
    (tmp_path / "loner.py").write_text("\n")

    # Build a minimal import graph: core -> helper, loner isolated.
    import networkx as nx
    import menace_sandbox.module_graph_analyzer as module_graph_analyzer
    import menace_sandbox.relevancy_metrics_db as relevancy_metrics_db
    import menace_sandbox.metrics_exporter as metrics_exporter

    def fake_build_graph(root):
        g = nx.DiGraph()
        g.add_edge("core", "helper")
        g.add_node("loner")
        return g

    monkeypatch.setattr(module_graph_analyzer, "build_import_graph", fake_build_graph)

    # Only ``core`` has recorded usage.
    monkeypatch.setattr(relevancy_radar, "load_usage_stats", lambda: {"core": 1})

    class DummyDB:
        def __init__(self, path):
            pass

        def get_roi_deltas(self, modules):
            return {}

    monkeypatch.setattr(relevancy_metrics_db, "RelevancyMetricsDB", DummyDB)
    monkeypatch.setattr(metrics_exporter, "update_relevancy_metrics", lambda flags: None)

    class DummyRetirementService:
        def __init__(self, root):
            pass

        def process_flags(self, flags):
            return flags

    monkeypatch.setattr(
        module_retirement_service, "ModuleRetirementService", DummyRetirementService
    )

    # Propagate usage to dependencies via evaluate_final_contribution.
    def fake_eval_final(self, compress, replace, *, graph, core_modules=None):
        used = {m for m, d in self._metrics.items() if d.get("executions", 0) > 0}
        reachable = set()
        for mod in used:
            node = mod.replace(".", "/")
            if node in graph:
                reachable.add(mod)
                reachable.update(n.replace("/", ".") for n in nx.descendants(graph, node))
        for mod in reachable:
            stats = self._metrics.setdefault(mod, {"imports": 0, "executions": 0, "impact": 0.0})
            if stats["imports"] == 0 and stats["executions"] == 0:
                stats["imports"] = 1
        return self.evaluate_relevance(
            compress, replace, dep_graph=graph, core_modules=core_modules
        )

    monkeypatch.setattr(
        relevancy_radar.RelevancyRadar, "evaluate_final_contribution", fake_eval_final
    )

    service = relevancy_radar_service.RelevancyRadarService(tmp_path)
    service._scan_once()

    # ``helper`` is imported by ``core`` and thus not flagged. ``loner`` is
    # isolated and should be marked for retirement.
    assert service.flags() == {"loner": "retire"}
    assert "helper" not in service.flags()


def test_metrics_increment_on_flags(monkeypatch, tmp_path):
    import menace_sandbox.metrics_exporter as metrics_exporter
    import menace_sandbox.module_graph_analyzer as module_graph_analyzer
    import menace_sandbox.relevancy_metrics_db as relevancy_metrics_db

    # Reset gauge values
    for gauge in (
        metrics_exporter.relevancy_flags_retire_total,
        metrics_exporter.relevancy_flags_compress_total,
        metrics_exporter.relevancy_flags_replace_total,
    ):
        gauge.set(0)

    class DummyGraph:
        nodes = ["a", "b", "c"]

    monkeypatch.setattr(module_graph_analyzer, "build_import_graph", lambda root: DummyGraph())
    monkeypatch.setattr(relevancy_radar, "load_usage_stats", lambda: {})

    class DummyDB:
        def __init__(self, path):
            pass

        def get_roi_deltas(self, modules):
            return {}

    monkeypatch.setattr(relevancy_metrics_db, "RelevancyMetricsDB", DummyDB)
    monkeypatch.setattr(metrics_exporter, "update_relevancy_metrics", lambda flags: None)

    def fake_eval(self, compress, replace, *, graph, core_modules=None):
        return {"a": "retire", "b": "compress", "c": "replace"}

    monkeypatch.setattr(
        relevancy_radar.RelevancyRadar,
        "evaluate_final_contribution",
        fake_eval,
    )

    class DummyRetirementService:
        def __init__(self, root):
            pass

        def process_flags(self, flags):
            return flags

    monkeypatch.setattr(
        module_retirement_service, "ModuleRetirementService", DummyRetirementService
    )

    service = relevancy_radar_service.RelevancyRadarService(tmp_path)
    service._scan_once()

    def gauge_value(child) -> float:
        getter = getattr(child, "get", None)
        if getter:
            return getter()
        return child._value.get()  # type: ignore[attr-defined]

    assert gauge_value(metrics_exporter.relevancy_flags_retire_total) == 1.0
    assert gauge_value(metrics_exporter.relevancy_flags_compress_total) == 1.0
    assert gauge_value(metrics_exporter.relevancy_flags_replace_total) == 1.0
