import logging
import os
import sys
import types

os.environ["MENACE_LIGHT_IMPORTS"] = "1"
for mod_name in ["jinja2", "yaml", "networkx"]:
    mod = types.ModuleType(mod_name)
    if mod_name == "jinja2":
        mod.Template = type("T", (), {"render": lambda self, *a, **k: ""})
    if mod_name == "yaml":
        mod.safe_load = lambda *a, **k: {}
    sys.modules.setdefault(mod_name, mod)
for name in ["env_config", "httpx", "requests", "numpy"]:
    mod = types.ModuleType(name)
    if name == "env_config":
        mod.DATABASE_URL = "sqlite:///:memory:"
    sys.modules.setdefault(name, mod)
sqlalchemy_mod = types.ModuleType("sqlalchemy")
engine_mod = types.ModuleType("sqlalchemy.engine")
engine_mod.Engine = object
sqlalchemy_mod.engine = engine_mod
sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
sys.modules.setdefault("sqlalchemy.engine", engine_mod)
crypto = types.ModuleType("cryptography")
sys.modules.setdefault("cryptography", crypto)
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("cryptography.hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("cryptography.hazmat.primitives"))
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", types.ModuleType("cryptography.hazmat.primitives.asymmetric"))
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric.ed25519", types.ModuleType("cryptography.hazmat.primitives.asymmetric.ed25519"))
sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"].Ed25519PrivateKey = object
sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"].Ed25519PublicKey = object
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", types.ModuleType("cryptography.hazmat.primitives.serialization"))
sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"].Ed25519PrivateKey = object
sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"].Ed25519PublicKey = object

from menace.neuroplasticity import PathwayDB
from menace.menace_memory_manager import MenaceMemoryManager
from menace.code_database import CodeDB
from menace.unified_learning_engine import UnifiedLearningEngine

class BadROI:
    def history(self, *a, **k):
        raise RuntimeError("boom")


def test_unified_learning_engine_logs(tmp_path, caplog):
    pdb = PathwayDB(tmp_path / "p.db")
    mm = MenaceMemoryManager(tmp_path / "m.db", embedder=None)
    mm._embed = lambda t: [1.0]  # type: ignore
    code_db = CodeDB(tmp_path / "c.db")
    engine = UnifiedLearningEngine(pdb, mm, code_db, BadROI())
    caplog.set_level(logging.ERROR)
    engine._roi_for_action("A")
    assert "roi lookup failed" in caplog.text
