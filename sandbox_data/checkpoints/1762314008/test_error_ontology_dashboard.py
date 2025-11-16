from pathlib import Path
import importlib.util
import json
import sqlite3 as _sqlite3
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("menace")
pkg.__path__ = [str(ROOT)]
sys.modules.setdefault("menace", pkg)


class DummyErrorDB:
    def __init__(self, path):
        sqlite3_connect = _sqlite3.connect
        self.conn = sqlite3_connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS telemetry(" "category TEXT, module TEXT, cause TEXT, "
            "resolution_status TEXT, source_menace_id TEXT NOT NULL DEFAULT '')"
        )
        self.conn.commit()

    def _menace_id(self, override=None):
        return override or ""


spec = importlib.util.spec_from_file_location(
    "menace.error_ontology_dashboard",
    ROOT / "error_ontology_dashboard.py",  # path-ignore
    submodule_search_locations=[str(ROOT)],
)
mod = importlib.util.module_from_spec(spec)
sys.modules["menace.error_ontology_dashboard"] = mod
spec.loader.exec_module(mod)

ErrorOntologyDashboard = mod.ErrorOntologyDashboard
ErrorDB = DummyErrorDB


class DummyGraph:
    def update_error_stats(self, db):
        pass


class DummyPredictor:
    def predict_high_risk_modules(self, *, min_cluster_size: int = 2, top_n: int = 5):
        return ["mod1", "mod2"]


def test_generate_report_includes_cause(tmp_path):
    db = ErrorDB(tmp_path / "e.db")
    db.conn.execute(
        "INSERT INTO telemetry(category,module,cause) VALUES (?,?,?)",
        ("cat1", "mod1", "cause1"),
    )
    db.conn.execute(
        "INSERT INTO telemetry(category,module,cause) VALUES (?,?,?)",
        ("cat1", "mod2", "cause2"),
    )
    db.conn.execute(
        "INSERT INTO telemetry(category,module,cause) VALUES (?,?,?)",
        ("cat2", "mod1", "cause1"),
    )
    db.conn.commit()

    dash = ErrorOntologyDashboard(error_db=db, graph=DummyGraph())
    dest = dash.generate_report(tmp_path / "rep.json")
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert any(c["cause"] == "cause1" and c["count"] == 2 for c in data["cause_stats"])
    assert any(c["cause"] == "cause2" and c["count"] == 1 for c in data["cause_stats"])


def test_category_success(tmp_path):
    db = ErrorDB(tmp_path / "e.db")
    db.conn.executemany(
        "INSERT INTO telemetry(category,resolution_status) VALUES (?,?)",
        [("cat1", "successful"), ("cat1", "fatal"), ("cat2", "successful")],
    )
    db.conn.commit()
    dash = ErrorOntologyDashboard(error_db=db, graph=DummyGraph())
    with dash.app.test_request_context():
        resp, _ = dash.category_success()
    data = json.loads(resp.get_data())
    rates = dict(zip(data["labels"], data["rate"]))
    assert rates["cat1"] == 0.5
    assert rates["cat2"] == 1.0


def test_predicted_modules(tmp_path):
    db = ErrorDB(tmp_path / "e.db")
    dash = ErrorOntologyDashboard(error_db=db, graph=DummyGraph())
    dash.predictor = DummyPredictor()
    with dash.app.test_request_context():
        resp, _ = dash.predicted_modules()
    data = json.loads(resp.get_data())
    assert data["labels"] == ["mod1", "mod2"]
    assert data["rank"] == [2, 1]


def test_category_data_scope(tmp_path):
    db = ErrorDB(tmp_path / "e.db")
    db.conn.executemany(
        "INSERT INTO telemetry(category,source_menace_id) VALUES (?,?)",
        [("local_cat", ""), ("global_cat", "remote")],
    )
    db.conn.commit()
    dash = ErrorOntologyDashboard(error_db=db, graph=DummyGraph())
    with dash.app.test_request_context("/category_data?scope=local"):
        resp, _ = dash.category_data()
    data = json.loads(resp.get_data())
    assert data["labels"] == ["local_cat"]

    with dash.app.test_request_context("/category_data?scope=global"):
        resp, _ = dash.category_data()
    data = json.loads(resp.get_data())
    assert data["labels"] == ["global_cat"]
