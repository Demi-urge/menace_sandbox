import sys
import types

loguru_mod = types.ModuleType("loguru")
loguru_mod.logger = type("Logger", (), {"add": lambda *a, **k: None})()
sys.modules.setdefault("loguru", loguru_mod)
git_mod = types.ModuleType("git")
exc_mod = types.ModuleType("git.exc")
exc_mod.GitCommandError = type("GitCommandError", (Exception,), {})
exc_mod.InvalidGitRepositoryError = type("InvalidGitRepositoryError", (Exception,), {})
exc_mod.NoSuchPathError = type("NoSuchPathError", (Exception,), {})
git_mod.Repo = type("Repo", (), {"init": lambda *a, **k: None})
git_mod.exc = exc_mod
sys.modules.setdefault("git", git_mod)
sys.modules.setdefault("git.exc", exc_mod)
fl_mod = types.ModuleType("filelock")
class _DummyLock:
    def __init__(self, *a, **k):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False

fl_mod.FileLock = _DummyLock
fl_mod.Timeout = type("Timeout", (Exception,), {})
sys.modules.setdefault("filelock", fl_mod)
sys.modules.pop("menace", None)
sys.modules.setdefault("networkx", types.ModuleType("networkx"))
sys.modules["networkx"].DiGraph = object
sys.modules.setdefault("pulp", types.ModuleType("pulp"))
pandas_mod = types.ModuleType("pandas")
pandas_mod.DataFrame = type("DF", (), {"__init__": lambda self, *a, **k: None, "empty": False})
pandas_mod.read_sql = lambda *a, **k: pandas_mod.DataFrame()
pandas_mod.read_csv = lambda *a, **k: pandas_mod.DataFrame()
sys.modules.setdefault("pandas", pandas_mod)
sys.modules.setdefault("psutil", types.ModuleType("psutil"))
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sqlalchemy_mod = types.ModuleType("sqlalchemy")
engine_mod = types.ModuleType("sqlalchemy.engine")
engine_mod.Engine = object
sqlalchemy_mod.engine = engine_mod
sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
sys.modules.setdefault("sqlalchemy.engine", engine_mod)
sys.modules.setdefault("prometheus_client", types.ModuleType("prometheus_client"))
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives", types.ModuleType("primitives")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    types.ModuleType("ed25519"),
)
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault(
    "cryptography.hazmat.primitives.serialization",
    serialization,
)
import importlib.util
import os

package = types.ModuleType("menace")
sys.modules["menace"] = package
package.__path__ = [os.path.join(os.path.dirname(__file__), "..")]
spec = importlib.util.spec_from_file_location(
    "menace.communication_maintenance_bot",
    os.path.join(os.path.dirname(__file__), "..", "communication_maintenance_bot.py"),  # path-ignore
)
cmb = importlib.util.module_from_spec(spec)
sys.modules["menace.communication_maintenance_bot"] = cmb
spec.loader.exec_module(cmb)


def test_sqlite_maintenance_db_roundtrip(tmp_path):
    db = cmb.SQLiteMaintenanceDB(tmp_path / "m.db")
    rec = cmb.MaintenanceRecord(task="t", status="ok", details="d")
    db.log(rec)
    rows = db.fetch()
    assert rows and rows[0][0] == "t" and rows[0][1] == "ok"


def test_sqlite_maintenance_db_state(tmp_path):
    db = cmb.SQLiteMaintenanceDB(tmp_path / "m.db")
    db.set_state("k", "v")
    assert db.get_state("k") == "v"
