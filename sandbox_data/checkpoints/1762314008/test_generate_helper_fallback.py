from pathlib import Path
import os
import sys
import types

# stub heavy optional deps
t = types.ModuleType("jinja2")
t.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", t)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519", types.ModuleType("ed25519")
)
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///:memory:"))
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
sys.modules.setdefault("menace.metrics_exporter", types.SimpleNamespace(error_bot_exceptions=None))
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import menace.self_coding_engine as sce  # noqa: E402
import menace.code_database as cd  # noqa: E402
import menace.menace_memory_manager as mm  # noqa: E402


def _engine(tmp_path: Path) -> sce.SelfCodingEngine:
    return sce.SelfCodingEngine(
        cd.CodeDB(tmp_path / "c.db"),
        mm.MenaceMemoryManager(tmp_path / "m.db"),
        context_builder=types.SimpleNamespace(
            build_context=lambda *a, **k: {},
            refresh_db_weights=lambda *a, **k: None,
        ),
    )


def test_read_file_fallback(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    code = eng.generate_helper("read file")
    assert "return fh.read()" in code


def test_unknown_fallback(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    code = eng.generate_helper("do something mysterious")
    assert "return {" in code
    assert "description" in code
