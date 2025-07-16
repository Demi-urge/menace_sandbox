from dataclasses import dataclass
from datetime import datetime

from menace.neuroplasticity import PathwayDB, PathwayRecord, Outcome
from menace.code_database import CodeDB, CodeRecord
from menace.menace_memory_manager import MenaceMemoryManager


@dataclass
class KPIRecord:
    bot: str
    revenue: float
    api_cost: float
    cpu_seconds: float
    success_rate: float
    ts: str = datetime.utcnow().isoformat()


class DummyROIDB:
    def __init__(self) -> None:
        self.records: list[KPIRecord] = []

    def add(self, rec: KPIRecord) -> int:
        self.records.append(rec)
        return len(self.records)

    class _Col(list):
        def mean(self) -> float:
            return sum(self) / len(self) if self else 0.0

    class _DF:
        def __init__(self, rows: list[dict]):
            self.rows = rows

        @property
        def empty(self) -> bool:
            return not self.rows

        def __getitem__(self, key: str) -> "DummyROIDB._Col":
            return DummyROIDB._Col([r[key] for r in self.rows])

    def history(self, bot: str | None = None, limit: int = 50) -> "DummyROIDB._DF":
        if bot:
            recs = [r.__dict__ for r in self.records if r.bot == bot]
        else:
            recs = [r.__dict__ for r in self.records]
        return DummyROIDB._DF(recs[-limit:])

    def future_roi(self, action: str, discount: float = 0.9) -> float:
        df = self.history(action, limit=5)
        if df.empty or len(df.rows) < 2:
            return 0.0
        rois = [
            (r["revenue"] - r["api_cost"]) / (r["cpu_seconds"] or 1.0) * r["success_rate"]
            for r in df.rows
        ]
        trend = rois[-1] - rois[0]
        return (rois[-1] + trend) * discount


def test_action_learning_next_action(tmp_path):
    import sys
    import types
    sys.modules.setdefault("networkx", types.ModuleType("networkx"))
    sys.modules.setdefault("pulp", types.ModuleType("pulp"))
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))
    jinja_mod = types.ModuleType("jinja2")
    class DummyTemplate:
        def __init__(self, *a, **k):
            pass

        def render(self, *a, **k):
            return ""

    jinja_mod.Template = DummyTemplate
    sys.modules.setdefault("jinja2", jinja_mod)
    sys.modules.setdefault("yaml", types.ModuleType("yaml"))
    sqlalchemy_mod = types.ModuleType("sqlalchemy")
    engine_mod = types.ModuleType("sqlalchemy.engine")
    class DummyEngine:
        pass

    engine_mod.Engine = DummyEngine
    sqlalchemy_mod.engine = engine_mod
    sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
    sys.modules.setdefault("sqlalchemy.engine", engine_mod)

    import menace.action_learning_engine as ale
    from menace.unified_learning_engine import UnifiedLearningEngine
    pdb = PathwayDB(tmp_path / "p.db")
    roi = DummyROIDB()
    roi.add(KPIRecord(bot="B", revenue=12.0, api_cost=1.0, cpu_seconds=1.0, success_rate=1.0))
    roi.add(KPIRecord(bot="B", revenue=8.0, api_cost=1.0, cpu_seconds=1.0, success_rate=1.0))
    roi.add(KPIRecord(bot="C", revenue=2.0, api_cost=1.0, cpu_seconds=1.0, success_rate=1.0))
    roi.add(KPIRecord(bot="C", revenue=10.0, api_cost=1.0, cpu_seconds=1.0, success_rate=1.0))
    code_db = CodeDB(tmp_path / "c.db")
    code_db.add(CodeRecord(code="b", complexity_score=1.0, summary="B"))
    code_db.add(CodeRecord(code="c", complexity_score=5.0, summary="C"))
    mm = MenaceMemoryManager(tmp_path / "m.db")
    ule = UnifiedLearningEngine(pdb, mm, code_db, roi)
    engine = ale.ActionLearningEngine(pdb, roi, code_db, ule, epsilon=0.0)

    pdb.log(PathwayRecord(actions="A->B", inputs="", outputs="", exec_time=1.0, resources="", outcome=Outcome.SUCCESS, roi=1.0))
    pdb.log(PathwayRecord(actions="A->C", inputs="", outputs="", exec_time=1.0, resources="", outcome=Outcome.FAILURE, roi=-1.0))

    engine.train()
    assert engine.next_action("A") == "C"


def test_action_learning_next_action_nn(tmp_path):
    import sys
    import types
    sys.modules.setdefault("networkx", types.ModuleType("networkx"))
    sys.modules.setdefault("pulp", types.ModuleType("pulp"))
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))
    jinja_mod = types.ModuleType("jinja2")
    class DummyTemplate:
        def __init__(self, *a, **k):
            pass

        def render(self, *a, **k):
            return ""

    jinja_mod.Template = DummyTemplate
    sys.modules.setdefault("jinja2", jinja_mod)
    sys.modules.setdefault("yaml", types.ModuleType("yaml"))
    sqlalchemy_mod = types.ModuleType("sqlalchemy")
    engine_mod = types.ModuleType("sqlalchemy.engine")
    class DummyEngine:
        pass

    engine_mod.Engine = DummyEngine
    sqlalchemy_mod.engine = engine_mod
    sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
    sys.modules.setdefault("sqlalchemy.engine", engine_mod)

    import menace.action_learning_engine as ale
    from menace.unified_learning_engine import UnifiedLearningEngine
    pdb = PathwayDB(tmp_path / "p.db")
    roi = DummyROIDB()
    roi.add(KPIRecord(bot="B", revenue=10.0, api_cost=1.0, cpu_seconds=1.0, success_rate=1.0))
    roi.add(KPIRecord(bot="C", revenue=5.0, api_cost=1.0, cpu_seconds=1.0, success_rate=0.0))
    code_db = CodeDB(tmp_path / "c.db")
    code_db.add(CodeRecord(code="b", complexity_score=1.0, summary="B"))
    code_db.add(CodeRecord(code="c", complexity_score=5.0, summary="C"))
    mm = MenaceMemoryManager(tmp_path / "m.db")
    ule = UnifiedLearningEngine(pdb, mm, code_db, roi, model="nn")
    engine = ale.ActionLearningEngine(pdb, roi, code_db, ule, epsilon=0.0)

    pdb.log(PathwayRecord(actions="A->B", inputs="", outputs="", exec_time=1.0, resources="", outcome=Outcome.SUCCESS, roi=1.0))
    pdb.log(PathwayRecord(actions="A->C", inputs="", outputs="", exec_time=1.0, resources="", outcome=Outcome.FAILURE, roi=-1.0))

    engine.train()
    assert engine.next_action("A") == "B"


def test_action_learning_save_and_load(tmp_path):
    import sys
    import types
    sys.modules.setdefault("networkx", types.ModuleType("networkx"))
    sys.modules.setdefault("pulp", types.ModuleType("pulp"))
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))
    jinja_mod = types.ModuleType("jinja2")
    class DummyTemplate:
        def __init__(self, *a, **k):
            pass

        def render(self, *a, **k):
            return ""

    jinja_mod.Template = DummyTemplate
    sys.modules.setdefault("jinja2", jinja_mod)
    sys.modules.setdefault("yaml", types.ModuleType("yaml"))
    sqlalchemy_mod = types.ModuleType("sqlalchemy")
    engine_mod = types.ModuleType("sqlalchemy.engine")
    class DummyEngine:
        pass

    engine_mod.Engine = DummyEngine
    sqlalchemy_mod.engine = engine_mod
    sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
    sys.modules.setdefault("sqlalchemy.engine", engine_mod)

    import menace.action_learning_engine as ale
    from menace.unified_learning_engine import UnifiedLearningEngine
    pdb = PathwayDB(tmp_path / "p.db")
    roi = DummyROIDB()
    roi.add(KPIRecord(bot="B", revenue=10.0, api_cost=1.0, cpu_seconds=1.0, success_rate=1.0))
    roi.add(KPIRecord(bot="C", revenue=5.0, api_cost=1.0, cpu_seconds=1.0, success_rate=0.0))
    code_db = CodeDB(tmp_path / "c.db")
    code_db.add(CodeRecord(code="b", complexity_score=1.0, summary="B"))
    code_db.add(CodeRecord(code="c", complexity_score=5.0, summary="C"))
    mm = MenaceMemoryManager(tmp_path / "m.db")
    ule = UnifiedLearningEngine(pdb, mm, code_db, roi)
    engine = ale.ActionLearningEngine(pdb, roi, code_db, ule, epsilon=0.0)

    pdb.log(PathwayRecord(actions="A->B", inputs="", outputs="", exec_time=1.0, resources="", outcome=Outcome.SUCCESS, roi=1.0))
    pdb.log(PathwayRecord(actions="A->C", inputs="", outputs="", exec_time=1.0, resources="", outcome=Outcome.FAILURE, roi=-1.0))

    engine.train()
    policy = tmp_path / "policy.bin"
    assert engine.save_policy(policy)

    engine2 = ale.ActionLearningEngine(pdb, roi, code_db, ule, epsilon=0.0)
    assert engine2.load_policy(policy)
    assert engine2.next_action("A") == "B"


def test_action_learning_partial_training(tmp_path):
    import sys
    import types
    sys.modules.setdefault("networkx", types.ModuleType("networkx"))
    sys.modules.setdefault("pulp", types.ModuleType("pulp"))
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))
    jinja_mod = types.ModuleType("jinja2")
    class DummyTemplate:
        def __init__(self, *a, **k):
            pass

        def render(self, *a, **k):
            return ""

    jinja_mod.Template = DummyTemplate
    sys.modules.setdefault("jinja2", jinja_mod)
    sys.modules.setdefault("yaml", types.ModuleType("yaml"))
    sqlalchemy_mod = types.ModuleType("sqlalchemy")
    engine_mod = types.ModuleType("sqlalchemy.engine")
    class DummyEngine:
        pass

    engine_mod.Engine = DummyEngine
    sqlalchemy_mod.engine = engine_mod
    sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
    sys.modules.setdefault("sqlalchemy.engine", engine_mod)

    import menace.action_learning_engine as ale
    from menace.unified_learning_engine import UnifiedLearningEngine
    pdb = PathwayDB(tmp_path / "p.db")
    roi = DummyROIDB()
    roi.add(KPIRecord(bot="B", revenue=10.0, api_cost=1.0, cpu_seconds=1.0, success_rate=1.0))
    roi.add(KPIRecord(bot="C", revenue=5.0, api_cost=1.0, cpu_seconds=1.0, success_rate=0.0))
    code_db = CodeDB(tmp_path / "c.db")
    code_db.add(CodeRecord(code="b", complexity_score=1.0, summary="B"))
    code_db.add(CodeRecord(code="c", complexity_score=5.0, summary="C"))
    mm = MenaceMemoryManager(tmp_path / "m.db")
    ule = UnifiedLearningEngine(pdb, mm, code_db, roi)
    engine = ale.ActionLearningEngine(pdb, roi, code_db, ule, epsilon=0.0)

    rec1 = PathwayRecord(actions="A->B", inputs="", outputs="", exec_time=1.0, resources="", outcome=Outcome.SUCCESS, roi=1.0)
    rec2 = PathwayRecord(actions="A->C", inputs="", outputs="", exec_time=1.0, resources="", outcome=Outcome.FAILURE, roi=-1.0)
    engine.partial_train(rec1)
    engine.partial_train(rec2)
    assert engine.next_action("A") == "B"


def test_reward_logic(tmp_path):
    import sys
    import types

    sys.modules.setdefault("networkx", types.ModuleType("networkx"))
    sys.modules.setdefault("pulp", types.ModuleType("pulp"))
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))
    jinja_mod = types.ModuleType("jinja2")

    class DummyTemplate:
        def __init__(self, *a, **k):
            pass

        def render(self, *a, **k):
            return ""

    jinja_mod.Template = DummyTemplate
    sys.modules.setdefault("jinja2", jinja_mod)
    sys.modules.setdefault("yaml", types.ModuleType("yaml"))
    sqlalchemy_mod = types.ModuleType("sqlalchemy")
    engine_mod = types.ModuleType("sqlalchemy.engine")

    class DummyEngine:
        pass

    engine_mod.Engine = DummyEngine
    sqlalchemy_mod.engine = engine_mod
    sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
    sys.modules.setdefault("sqlalchemy.engine", engine_mod)

    import menace.action_learning_engine as ale

    class DummyULE:
        def predict_success(self, *a, **k):
            return 0.5

    pdb = PathwayDB(tmp_path / "p.db")
    roi = DummyROIDB()
    roi.add(KPIRecord(bot="B", revenue=10.0, api_cost=1.0, cpu_seconds=1.0, success_rate=1.0))
    roi.add(KPIRecord(bot="B", revenue=20.0, api_cost=1.0, cpu_seconds=1.0, success_rate=1.0))
    code_db = CodeDB(tmp_path / "c.db")
    code_db.add(CodeRecord(code="b", complexity_score=2.0, summary="B"))
    engine = ale.ActionLearningEngine(
        pdb,
        roi,
        code_db,
        DummyULE(),
        reward_weights={"roi": 1.0, "future_roi": 1.0, "trend": 1.0, "prediction": 1.0, "accuracy": 1.0, "complexity": 1.0},
        epsilon=0.0,
    )

    rec = PathwayRecord(actions="A->B", inputs="", outputs="", exec_time=1.0, resources="", outcome=Outcome.SUCCESS, roi=5.0)

    reward = engine._reward("B", rec)
    features = engine._observation_features("B", rec)
    assert features == (10.0, 0.5, 2.0)
    assert round(reward, 1) == 49.1


def test_action_learning_sb3_algorithms(tmp_path):
    import sys
    import types

    sys.modules.setdefault("networkx", types.ModuleType("networkx"))
    sys.modules.setdefault("pulp", types.ModuleType("pulp"))
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))
    jinja_mod = types.ModuleType("jinja2")
    class DummyTemplate:
        def __init__(self, *a, **k):
            pass

        def render(self, *a, **k):
            return ""

    jinja_mod.Template = DummyTemplate
    sys.modules.setdefault("jinja2", jinja_mod)
    sys.modules.setdefault("yaml", types.ModuleType("yaml"))
    sqlalchemy_mod = types.ModuleType("sqlalchemy")
    engine_mod = types.ModuleType("sqlalchemy.engine")
    class DummyEngine:
        pass

    engine_mod.Engine = DummyEngine
    sqlalchemy_mod.engine = engine_mod
    sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
    sys.modules.setdefault("sqlalchemy.engine", engine_mod)

    sb3 = types.ModuleType("stable_baselines3")

    last = {}

    class DummyAlgo:
        def __init__(self, *a, **k):
            last["algo"] = self.__class__.__name__
            last["kwargs"] = k

        def learn(self, *a, **k):
            last["learn"] = a, k

        def predict(self, s):
            return 0, None

        def save(self, p):
            pass

        @classmethod
        def load(cls, p):
            return cls()

    class SAC(DummyAlgo):
        pass

    sb3.DQN = sb3.PPO = sb3.A2C = DummyAlgo
    sb3.SAC = SAC
    sb3.TD3 = DummyAlgo
    sys.modules["stable_baselines3"] = sb3

    gym_mod = types.ModuleType("gym")
    spaces_mod = types.ModuleType("gym.spaces")

    class Discrete:
        def __init__(self, n):
            self.n = n

    spaces_mod.Discrete = Discrete
    gym_mod.Env = object
    sys.modules["gym"] = gym_mod
    sys.modules["gym.spaces"] = spaces_mod
    gym_mod.spaces = spaces_mod

    import importlib
    if "menace.action_learning_engine" in sys.modules:
        del sys.modules["menace.action_learning_engine"]
    import menace.action_learning_engine as ale
    importlib.reload(ale)
    from menace.unified_learning_engine import UnifiedLearningEngine
    pdb = PathwayDB(tmp_path / "p.db")
    roi = DummyROIDB()
    code_db = CodeDB(tmp_path / "c.db")
    mm = MenaceMemoryManager(tmp_path / "m.db")
    ule = UnifiedLearningEngine(pdb, mm, code_db, roi)
    engine = ale.ActionLearningEngine(
        pdb,
        roi,
        code_db,
        ule,
        algo="sac",
        algo_kwargs={"learning_rate": 0.01},
        train_steps=2,
        epsilon=0.0,
    )

    pdb.log(
        PathwayRecord(
            actions="A->B",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.SUCCESS,
            roi=1.0,
        )
    )
    pdb.log(
        PathwayRecord(
            actions="A->C",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.FAILURE,
            roi=-1.0,
        )
    )

    engine.train()

    assert last["algo"] == "SAC"
    assert last["kwargs"]["learning_rate"] == 0.01


def test_reward_weights_from_env(tmp_path, monkeypatch):
    import sys
    import types

    sys.modules.setdefault("networkx", types.ModuleType("networkx"))
    sys.modules.setdefault("pulp", types.ModuleType("pulp"))
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))
    jinja_mod = types.ModuleType("jinja2")

    class DummyTemplate:
        def __init__(self, *a, **k):
            pass

        def render(self, *a, **k):
            return ""

    jinja_mod.Template = DummyTemplate
    sys.modules.setdefault("jinja2", jinja_mod)
    sys.modules.setdefault("yaml", types.ModuleType("yaml"))
    sqlalchemy_mod = types.ModuleType("sqlalchemy")
    engine_mod = types.ModuleType("sqlalchemy.engine")

    class DummyEngine:
        pass

    engine_mod.Engine = DummyEngine
    sqlalchemy_mod.engine = engine_mod
    sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
    sys.modules.setdefault("sqlalchemy.engine", engine_mod)

    import menace.action_learning_engine as ale

    monkeypatch.setenv("ACTION_LEARNING_REWARD_WEIGHTS", "{\"roi\": 2}")
    pdb = PathwayDB(tmp_path / "p.db")
    roi = DummyROIDB()
    code_db = CodeDB(tmp_path / "c.db")
    engine = ale.ActionLearningEngine(pdb, roi, code_db)

    assert engine.reward_weights["roi"] == 2


def test_action_filter_fn(tmp_path):
    import sys
    import types

    sys.modules.setdefault("networkx", types.ModuleType("networkx"))
    sys.modules.setdefault("pulp", types.ModuleType("pulp"))
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))
    jinja_mod = types.ModuleType("jinja2")

    class DummyTemplate:
        def __init__(self, *a, **k):
            pass

        def render(self, *a, **k):
            return ""

    jinja_mod.Template = DummyTemplate
    sys.modules.setdefault("jinja2", jinja_mod)
    sys.modules.setdefault("yaml", types.ModuleType("yaml"))
    sqlalchemy_mod = types.ModuleType("sqlalchemy")
    engine_mod = types.ModuleType("sqlalchemy.engine")

    class DummyEngine:
        pass

    engine_mod.Engine = DummyEngine
    sqlalchemy_mod.engine = engine_mod
    sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
    sys.modules.setdefault("sqlalchemy.engine", engine_mod)

    import menace.action_learning_engine as ale
    from menace.unified_learning_engine import UnifiedLearningEngine
    pdb = PathwayDB(tmp_path / "p.db")
    roi = DummyROIDB()
    roi.add(KPIRecord(bot="B", revenue=10.0, api_cost=1.0, cpu_seconds=1.0, success_rate=1.0))
    code_db = CodeDB(tmp_path / "c.db")
    mm = MenaceMemoryManager(tmp_path / "m.db")
    ule = UnifiedLearningEngine(pdb, mm, code_db, roi)
    engine = ale.ActionLearningEngine(pdb, roi, code_db, ule, epsilon=0.0, action_filter_fn=lambda a: a != "B")

    pdb.log(PathwayRecord(actions="A->B", inputs="", outputs="", exec_time=1.0, resources="", outcome=Outcome.SUCCESS, roi=1.0))
    engine.train()
    assert engine.next_action("A") is None


def test_save_policy_meta(tmp_path):
    import sys
    import types

    sys.modules.setdefault("networkx", types.ModuleType("networkx"))
    sys.modules.setdefault("pulp", types.ModuleType("pulp"))
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))
    jinja_mod = types.ModuleType("jinja2")
    jinja_mod.Template = type("T", (), {"render": lambda *a, **k: ""})
    sys.modules.setdefault("jinja2", jinja_mod)
    sys.modules.setdefault("yaml", types.ModuleType("yaml"))
    sqlalchemy_mod = types.ModuleType("sqlalchemy")
    engine_mod = types.ModuleType("sqlalchemy.engine")
    engine_mod.Engine = type("E", (), {})
    sqlalchemy_mod.engine = engine_mod
    sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
    sys.modules.setdefault("sqlalchemy.engine", engine_mod)

    import menace.action_learning_engine as ale
    pdb = PathwayDB(tmp_path / "p.db")
    roi = DummyROIDB()
    code_db = CodeDB(tmp_path / "c.db")
    engine = ale.ActionLearningEngine(pdb, roi, code_db, epsilon=0.0)
    pdb.log(PathwayRecord(actions="A->B", inputs="", outputs="", exec_time=1.0, resources="", outcome=Outcome.SUCCESS, roi=1.0))
    engine.train()
    policy = tmp_path / "p.bin"
    assert engine.save_policy(policy)
    meta = policy.parent / (policy.name + ".meta.json")
    assert meta.exists()

