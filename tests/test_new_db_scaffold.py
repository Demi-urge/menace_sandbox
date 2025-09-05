import importlib
import shutil
import sys
from pathlib import Path

from scripts.scaffold_db import create_db_scaffold


def test_scaffold(tmp_path):
    # Provide the SQL template expected by the scaffold helper
    repo_root = Path(__file__).resolve().parents[1]
    sql_dir = tmp_path / "sql_templates"
    sql_dir.mkdir()
    shutil.copy(repo_root / "sql_templates" / "create_fts.sql", sql_dir / "create_fts.sql")

    # Minimal package to allow __init__ updates
    (tmp_path / "__init__.py").write_text("__all__ = []\n")  # path-ignore

    create_db_scaffold("demo", root=tmp_path)

    sys.path.insert(0, str(tmp_path))
    mod = importlib.import_module("demo_db")
    assert hasattr(mod, "DemoDB")
