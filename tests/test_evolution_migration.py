import sqlite3
from pathlib import Path
from menace.migrations.upgrade_evolution_history import upgrade


def test_upgrade_script(tmp_path: Path):
    db_path = tmp_path / "e.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE evolution_history(action TEXT, before_metric REAL, after_metric REAL, roi REAL, ts TEXT)"
    )
    conn.commit()
    conn.close()
    upgrade(db_path)
    conn = sqlite3.connect(db_path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(evolution_history)").fetchall()]
    conn.close()
    assert "efficiency" in cols and "bottleneck" in cols
