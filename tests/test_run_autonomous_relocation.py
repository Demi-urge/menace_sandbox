import ast
import json
import sqlite3
import sys
import types
from pathlib import Path

RA_PATH = Path(__file__).resolve().parents[1] / "run_autonomous.py"  # path-ignore

sys.modules.pop("dynamic_path_router", None)
import dynamic_path_router as dpr  # noqa: E402

clear_cache = dpr.clear_cache


def _load_previous_synergy():
    src = RA_PATH.read_text()
    tree = ast.parse(src)
    func = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_previous_synergy"
    )
    module = ast.Module([func], type_ignores=[])
    ns: dict[str, object] = {}

    class _Conn:
        def __init__(self, p: Path):
            self.conn = sqlite3.connect(p)  # noqa: SQL001

        def __enter__(self):  # noqa: D401 - context manager protocol
            return self.conn

        def __exit__(self, *_exc):
            self.conn.close()
            return False

    cli = types.SimpleNamespace(_ema=lambda vals: (vals[-1], 0.0))
    exec(
        compile(module, "ra_subset", "exec"),
        {
            "Path": Path,
            "json": json,
            "connect_locked": lambda p: _Conn(Path(p)),
            "cli": cli,
            "resolve_path": dpr.resolve_path,
        },
        ns,
    )
    return ns["load_previous_synergy"]


def test_load_previous_synergy_uses_repo_path(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data_dir = repo / "data"
    data_dir.mkdir(parents=True)
    db_path = data_dir / "synergy_history.db"
    conn = sqlite3.connect(db_path)  # noqa: SQL001
    conn.execute("CREATE TABLE synergy_history (id INTEGER PRIMARY KEY, entry TEXT)")
    conn.execute("INSERT INTO synergy_history(entry) VALUES (?)", [json.dumps({"x": 1.0})])
    conn.commit()
    conn.close()

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    clear_cache()
    func = _load_previous_synergy()

    history, ma = func("data")
    assert history == [{"x": 1.0}]
    assert ma and isinstance(ma[0], dict)
