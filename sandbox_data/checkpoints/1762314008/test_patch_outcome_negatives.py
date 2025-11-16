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
yaml_mod = types.ModuleType("yaml")
yaml_mod.safe_load = lambda *a, **k: {}
yaml_mod.dump = lambda *a, **k: ""
sys.modules.setdefault("yaml", yaml_mod)
stub = types.ModuleType("numpy")
sys.modules.setdefault("numpy", stub)
mpl = types.ModuleType("matplotlib")
mpl.pyplot = types.ModuleType("pyplot")  # path-ignore
sys.modules.setdefault("matplotlib", mpl)
sys.modules.setdefault("matplotlib.pyplot", mpl.pyplot)  # path-ignore
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
neuro = types.ModuleType("neurosales")
neuro.add_message = lambda *a, **k: None
neuro.get_recent_messages = lambda *a, **k: []
neuro.push_chain = lambda *a, **k: None
neuro.peek_chain = lambda *a, **k: None
neuro.MessageEntry = object
neuro.CTAChain = object
sys.modules.setdefault("neurosales", neuro)
sys.modules.setdefault("menace.chatgpt_idea_bot", types.SimpleNamespace(ChatGPTClient=object))
loguru_mod = types.ModuleType("loguru")

class DummyLogger:
    def __getattr__(self, name):
        def stub(*a, **k):
            return None

        return stub

    def add(self, *a, **k):
        pass

loguru_mod.logger = DummyLogger()
sys.modules.setdefault("loguru", loguru_mod)

git_mod = types.ModuleType("git")
git_mod.Repo = object
git_exc = types.ModuleType("git.exc")
class _GitErr(Exception):
    pass
git_exc.GitCommandError = _GitErr
git_exc.InvalidGitRepositoryError = _GitErr
git_exc.NoSuchPathError = _GitErr
git_mod.exc = git_exc
sys.modules.setdefault("git", git_mod)
sys.modules.setdefault("git.exc", git_exc)

filelock_mod = types.ModuleType("filelock")
class DummyLock:
    def __init__(self, *a, **k):
        self.is_locked = False
        self.lock_file = ""
    def acquire(self, *a, **k):
        self.is_locked = True
    def release(self):
        self.is_locked = False
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass
filelock_mod.FileLock = DummyLock
filelock_mod.Timeout = type("Timeout", (Exception,), {})
sys.modules.setdefault("filelock", filelock_mod)

dotenv_mod = types.ModuleType("dotenv")
dotenv_mod.load_dotenv = lambda *a, **k: None
sys.modules.setdefault("dotenv", dotenv_mod)

prom_mod = types.ModuleType("prometheus_client")
prom_mod.CollectorRegistry = object
prom_mod.Counter = prom_mod.Gauge = lambda *a, **k: object()
sys.modules.setdefault("prometheus_client", prom_mod)

joblib_mod = types.ModuleType("joblib")
joblib_mod.dump = joblib_mod.load = lambda *a, **k: None
sys.modules.setdefault("joblib", joblib_mod)

pyd_mod = types.ModuleType("pydantic")
pyd_mod.__path__ = []  # type: ignore
pyd_dc = types.ModuleType("dataclasses")
pyd_dc.dataclass = lambda *a, **k: (lambda f: f)
pyd_mod.dataclasses = pyd_dc
pyd_mod.Field = lambda default=None, **k: default
pyd_mod.ConfigDict = dict
pyd_mod.field_validator = lambda *a, **k: (lambda f: f)
pyd_mod.model_validator = lambda *a, **k: (lambda f: f)
class _VE(Exception):
    pass
pyd_mod.ValidationError = _VE
pyd_mod.BaseModel = object
sys.modules.setdefault("pydantic", pyd_mod)
sys.modules.setdefault("pydantic.dataclasses", pyd_dc)
pyd_settings_mod = types.ModuleType("pydantic_settings")
pyd_settings_mod.BaseSettings = object
pyd_settings_mod.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", pyd_settings_mod)

sk_mod = types.ModuleType("sklearn")
sk_mod.__path__ = []  # type: ignore
fe_mod = types.ModuleType("sklearn.feature_extraction")
fe_mod.__path__ = []  # type: ignore
text_mod = types.ModuleType("sklearn.feature_extraction.text")
text_mod.__path__ = []  # type: ignore
text_mod.TfidfVectorizer = object
fe_mod.text = text_mod
sk_mod.feature_extraction = fe_mod
sk_mod.feature_extraction.text = text_mod
cluster_mod = types.ModuleType("sklearn.cluster")
cluster_mod.__path__ = []  # type: ignore
cluster_mod.KMeans = object
lm_mod = types.ModuleType("sklearn.linear_model")
lm_mod.__path__ = []  # type: ignore
lm_mod.LinearRegression = object
sk_mod.cluster = cluster_mod
sk_mod.linear_model = lm_mod
pre_mod = types.ModuleType("sklearn.preprocessing")
pre_mod.__path__ = []  # type: ignore
pre_mod.PolynomialFeatures = object
sk_mod.preprocessing = pre_mod
sys.modules.setdefault("sklearn", sk_mod)
sys.modules.setdefault("sklearn.feature_extraction", fe_mod)
sys.modules.setdefault("sklearn.feature_extraction.text", text_mod)
sys.modules.setdefault("sklearn.cluster", cluster_mod)
sys.modules.setdefault("sklearn.linear_model", lm_mod)
sys.modules.setdefault("sklearn.preprocessing", pre_mod)

sys.modules.setdefault("scipy", types.ModuleType("scipy"))

import sqlite3
import menace.self_coding_engine as sce
qfe = types.ModuleType("menace.quick_fix_engine")
qfe.generate_patch = lambda *a, **k: 0
sys.modules.setdefault("menace.quick_fix_engine", qfe)
sys.modules.setdefault("quick_fix_engine", qfe)
map_mod = types.ModuleType("menace.model_automation_pipeline")
class _Pipe:
    def __init__(self, *a, **k):
        pass
    def run(self, *a, **k):
        return types.SimpleNamespace(package=None, roi=types.SimpleNamespace(roi=0.0))
map_mod.ModelAutomationPipeline = _Pipe
map_mod.AutomationResult = object
sys.modules.setdefault("menace.model_automation_pipeline", map_mod)
sys.modules.setdefault("model_automation_pipeline", map_mod)
import menace.self_improvement as sie
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
        context_builder=types.SimpleNamespace(
            build_context=lambda *a, **k: {},
            refresh_db_weights=lambda *a, **k: None,
        ),
    )
    engine.formal_verifier = None
    monkeypatch.setattr(engine, "_run_ci", lambda *a, **k: True)
    monkeypatch.setattr(engine, "generate_helper", lambda d: "def auto_x():\n    pass\n")
    roi_vals = iter([0.0, 1.0])
    monkeypatch.setattr(engine.data_bot, "roi", lambda *a, **k: next(roi_vals))
    monkeypatch.setattr(engine.data_bot, "complexity_score", lambda *a, **k: 0.0)
    monkeypatch.setattr(engine, "_current_errors", lambda: 0)
    path = tmp_path / "bot.py"  # path-ignore
    path.write_text("def x():\n    pass\n")
    context = {"retrieval_session_id": "s1", "retrieval_vectors": [("db", "v1", 0.0)]}
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
        context_builder=types.SimpleNamespace(
            build_context=lambda *a, **k: {},
            refresh_db_weights=lambda *a, **k: None,
        ),
    )
    engine.formal_verifier = None
    monkeypatch.setattr(engine, "_run_ci", lambda *a, **k: False)
    monkeypatch.setattr(engine, "generate_helper", lambda d: "def auto_y():\n    pass\n")
    monkeypatch.setattr(engine.data_bot, "roi", lambda *a, **k: 0.0)
    monkeypatch.setattr(engine.data_bot, "complexity_score", lambda *a, **k: 0.0)
    monkeypatch.setattr(engine, "_current_errors", lambda: 0)
    path = tmp_path / "bot.py"  # path-ignore
    path.write_text("def y():\n    pass\n")
    context = {"retrieval_session_id": "s1", "retrieval_vectors": [("db", "v1", 0.0)]}
    patch_id, reverted, _ = engine.apply_patch(path, "test", context_meta=context)
    assert patch_id is None and not reverted
    with sqlite3.connect(mdb_path) as conn:
        row = conn.execute(
            "SELECT success, reverted, origin_db, vector_id FROM patch_outcomes WHERE patch_id=?",
            ("test",),
        ).fetchone()
    assert row == (0, 0, "db", "v1")


def test_skipped_enhancement_logs_negative_outcome(tmp_path):
    mdb_path = tmp_path / "m.db"
    mdb = db.MetricsDB(mdb_path)
    patch_db = cd.PatchHistoryDB(tmp_path / "p.db")
    data_bot = db.DataBot(mdb, patch_db=patch_db)
    engine = sie.SelfImprovementEngine.__new__(sie.SelfImprovementEngine)
    engine.roi_history = [0.1]

    class Pred:
        def predict(self, features, horizon=1):
            return [0.0], "marginal", [0.0], [0.0]

    engine.roi_predictor = Pred()
    engine.allow_marginal_candidates = False
    engine.data_bot = data_bot
    engine._log_action = lambda *a, **k: None
    import logging

    engine.logger = logging.getLogger("test")
    engine._current_context = {
        "retrieval_session_id": "s1",
        "retrieval_vectors": [("db", "v1", 0.0)],
    }
    feats = engine._collect_action_features()
    assert feats == []
    with sqlite3.connect(mdb_path) as conn:
        row = conn.execute(
            "SELECT success, reverted, origin_db, vector_id FROM patch_outcomes WHERE patch_id=?",
            ("roi_history_0",),
        ).fetchone()
    assert row == (0, 0, "db", "v1")
