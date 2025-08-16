import sys
import types

sys.modules.setdefault("bot_database", types.SimpleNamespace(BotDB=object))
sys.modules.setdefault("task_handoff_bot", types.SimpleNamespace(WorkflowDB=object))
sys.modules.setdefault("error_bot", types.SimpleNamespace(ErrorDB=object))
sys.modules.setdefault("failure_learning_system", types.SimpleNamespace(DiscrepancyDB=object))
sys.modules.setdefault("code_database", types.SimpleNamespace(CodeDB=object))

jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
stub = types.ModuleType("numpy")
sys.modules.setdefault("numpy", stub)
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric"))
ed = types.ModuleType("ed25519")
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric.ed25519", ed)
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///:memory:"))
sys.modules.setdefault("httpx", types.ModuleType("httpx"))

import sqlite3
import menace.self_coding_engine as sce
import menace.code_database as cd
import menace.menace_memory_manager as mm
import menace.data_bot as db


def test_rollback_logs_negative_outcome(tmp_path, monkeypatch):
    mdb_path = tmp_path / "m.db"
    mdb = db.MetricsDB(mdb_path)
    patch_db = cd.PatchHistoryDB(tmp_path / "p.db")
    data_bot = db.DataBot(mdb, patch_db=patch_db)
    mem = mm.MenaceMemoryManager(tmp_path / "mem.db")
    engine = sce.SelfCodingEngine(
        cd.CodeDB(tmp_path / "c.db"),
        mem,
        data_bot=data_bot,
        patch_db=patch_db,
    )
    monkeypatch.setattr(engine, "_run_ci", lambda *a, **k: True)
    monkeypatch.setattr(engine, "generate_helper", lambda d: "def auto_x():\n    pass\n")
    roi_vals = iter([0.0, 1.0])
    monkeypatch.setattr(engine.data_bot, "roi", lambda *a, **k: next(roi_vals))
    monkeypatch.setattr(engine.data_bot, "complexity_score", lambda *a, **k: 0.0)
    monkeypatch.setattr(engine, "_current_errors", lambda: 0)
    path = tmp_path / "bot.py"
    path.write_text("def x():\n    pass\n")
    context = {"retrieval_session_id": "s1", "retrieval_vectors": [("db", "v1")]} 
    patch_id, reverted, _ = engine.apply_patch(path, "test", context_meta=context)
    assert patch_id is not None and not reverted
    engine.rollback_patch(str(patch_id))
    with sqlite3.connect(mdb_path) as conn:
        row = conn.execute(
            "SELECT success, reverted, origin_db, vector_id FROM patch_outcomes WHERE patch_id=? AND reverted=1",
            (str(patch_id),),
        ).fetchone()
    assert row == (0, 1, "db", "v1")


def test_failed_tests_log_negative_outcome(tmp_path, monkeypatch):
    mdb_path = tmp_path / "m.db"
    mdb = db.MetricsDB(mdb_path)
    patch_db = cd.PatchHistoryDB(tmp_path / "p.db")
    data_bot = db.DataBot(mdb, patch_db=patch_db)
    mem = mm.MenaceMemoryManager(tmp_path / "mem.db")
    engine = sce.SelfCodingEngine(
        cd.CodeDB(tmp_path / "c.db"),
        mem,
        data_bot=data_bot,
        patch_db=patch_db,
    )
    monkeypatch.setattr(engine, "_run_ci", lambda *a, **k: False)
    monkeypatch.setattr(engine, "generate_helper", lambda d: "def auto_y():\n    pass\n")
    monkeypatch.setattr(engine.data_bot, "roi", lambda *a, **k: 0.0)
    monkeypatch.setattr(engine.data_bot, "complexity_score", lambda *a, **k: 0.0)
    monkeypatch.setattr(engine, "_current_errors", lambda: 0)
    path = tmp_path / "bot.py"
    path.write_text("def y():\n    pass\n")
    context = {"retrieval_session_id": "s1", "retrieval_vectors": [("db", "v1")]} 
    patch_id, reverted, _ = engine.apply_patch(path, "test", context_meta=context)
    assert patch_id is None and not reverted
    with sqlite3.connect(mdb_path) as conn:
        row = conn.execute(
            "SELECT success, reverted, origin_db, vector_id FROM patch_outcomes WHERE patch_id=?",
            ("test",),
        ).fetchone()
    assert row == (0, 0, "db", "v1")
