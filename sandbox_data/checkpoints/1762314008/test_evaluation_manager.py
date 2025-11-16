import sys
import types

# Stub heavy dependencies from learning modules
sys.modules.setdefault("networkx", types.ModuleType("networkx"))
sys.modules.setdefault("pulp", types.ModuleType("pulp"))
sys.modules.setdefault("pandas", types.ModuleType("pandas"))
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sqlalchemy_mod = types.ModuleType("sqlalchemy")
engine_mod = types.ModuleType("sqlalchemy.engine")
class DummyEngineMod:
    pass
engine_mod.Engine = DummyEngineMod
sqlalchemy_mod.engine = engine_mod
sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
sys.modules.setdefault("sqlalchemy.engine", engine_mod)
sys.modules.setdefault("prometheus_client", types.ModuleType("prometheus_client"))

# Stub engine modules to avoid heavy imports
le_mod = types.ModuleType("learning_engine")
le_mod.LearningEngine = object
ue_mod = types.ModuleType("unified_learning_engine")
ue_mod.UnifiedLearningEngine = object
ae_mod = types.ModuleType("action_learning_engine")
ae_mod.ActionLearningEngine = object
sys.modules.setdefault("menace.learning_engine", le_mod)
sys.modules.setdefault("menace.unified_learning_engine", ue_mod)
sys.modules.setdefault("menace.action_learning_engine", ae_mod)

from menace.evaluation_manager import EvaluationManager

for mod in [
    "networkx",
    "pulp",
    "pandas",
    "jinja2",
    "yaml",
    "prometheus_client",
    "menace.learning_engine",
    "menace.unified_learning_engine",
    "menace.action_learning_engine",
]:
    sys.modules.pop(mod, None)


class DummyEngine:
    def __init__(self, score: float) -> None:
        self.score = score
        self.evaluations = 0
        self.persisted = []

    def evaluate(self):
        self.evaluations += 1
        return {"cv_score": self.score, "holdout_score": self.score + 0.1}

    def persist_evaluation(self, res):
        self.persisted.append(res)


def test_collect_and_select_best():
    e1 = DummyEngine(0.1)
    e2 = DummyEngine(0.5)
    mgr = EvaluationManager(learning_engine=e1, unified_engine=e2)
    res = mgr.evaluate_all()
    assert e1.evaluations == 1
    assert e2.evaluations == 1
    assert "learning_engine" in res and "unified_engine" in res
    assert mgr.best_engine() == e2
    # update score to make e1 best
    e1.score = 0.6
    mgr.evaluate_all()
    assert mgr.best_engine() == e1
    assert e1.persisted
    assert e2.persisted


def test_logs_db_and_persist_failures(caplog):
    class FailDB:
        def add(self, rec):
            raise RuntimeError("boom")

    class FailEngine(DummyEngine):
        def persist_evaluation(self, res):
            raise RuntimeError("disk full")

    e = FailEngine(0.1)
    mgr = EvaluationManager(learning_engine=e, history_db=FailDB())
    caplog.set_level("ERROR")
    mgr.evaluate_all()
    assert "Failed to add evaluation record" in caplog.text
    assert "Failed to persist evaluation" in caplog.text
