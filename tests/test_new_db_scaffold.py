from pathlib import Path

from scripts.new_db import create_db_scaffold


def test_scaffold(tmp_path):
    # Prepare minimal embedding_backfill file for registration updates
    vs = tmp_path / "vector_service"
    vs.mkdir()
    (vs / "embedding_backfill.py").write_text("modules = [\n]\n")

    create_db_scaffold("demo", root=tmp_path)

    assert (tmp_path / "demo_db.py").exists()
    assert (tmp_path / "tests" / "test_demo_db.py").exists()

    reg = (vs / "embedding_backfill.py").read_text()
    assert '"demo_db"' in reg
