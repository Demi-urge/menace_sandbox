import json
import importlib.util
from pathlib import Path
import types
import sys
import pytest

db_router = types.ModuleType("db_router")
db_router.DBRouter = object
db_router.GLOBAL_ROUTER = None
db_router.LOCAL_TABLES = set()
db_router.init_db_router = lambda *a, **k: None
sys.modules.setdefault("db_router", db_router)

metrics_exporter = types.ModuleType("metrics_exporter")
metrics_exporter.update_relevancy_metrics = lambda **k: None
sys.modules.setdefault("metrics_exporter", metrics_exporter)

relevancy_metrics_db = types.ModuleType("relevancy_metrics_db")
class _RelevancyMetricsDB:  # pragma: no cover - stub
    pass
relevancy_metrics_db.RelevancyMetricsDB = _RelevancyMetricsDB
sys.modules.setdefault("relevancy_metrics_db", relevancy_metrics_db)

dynamic_path_router = types.ModuleType("dynamic_path_router")
dynamic_path_router.resolve_path = lambda p: Path(p)
sys.modules.setdefault("dynamic_path_router", dynamic_path_router)

spec = importlib.util.spec_from_file_location(
    "relevancy_radar", Path(__file__).resolve().parents[1] / "relevancy_radar.py"  # path-ignore
)
relevancy_radar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relevancy_radar)


def test_call_graph_complexity(tmp_path, monkeypatch):
    data = {"a": ["b", "c"], "b": ["c"]}
    path = tmp_path / "cg.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(relevancy_radar, "_RELEVANCY_CALL_GRAPH_FILE", path)
    assert relevancy_radar.call_graph_complexity() == pytest.approx(1.0)

