from pathlib import Path
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "menace.error_ontology_dashboard",
    ROOT / "error_ontology_dashboard.py",
    submodule_search_locations=[str(ROOT)],
)
mod = importlib.util.module_from_spec(spec)
sys.modules["menace.error_ontology_dashboard"] = mod
spec.loader.exec_module(mod)

ErrorOntologyDashboard = mod.ErrorOntologyDashboard
ErrorDB = mod.ErrorDB


class DummyGraph:
    def update_error_stats(self, db):
        pass


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
