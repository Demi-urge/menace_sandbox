from dataclasses import dataclass
from datetime import datetime

import sys
import types
import pytest

sys.modules.setdefault("networkx", types.ModuleType("networkx"))
sys.modules["networkx"].DiGraph = object
sys.modules.setdefault("pulp", types.ModuleType("pulp"))
sys.modules.setdefault("pandas", types.ModuleType("pandas"))
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("prometheus_client", types.ModuleType("prometheus_client"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric"))
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric.ed25519", types.ModuleType("ed25519"))
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)
sqlalchemy_mod = types.ModuleType("sqlalchemy")
engine_mod = types.ModuleType("sqlalchemy.engine")
class DummyEngine:
    pass

engine_mod.Engine = DummyEngine
sqlalchemy_mod.engine = engine_mod
sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
sys.modules.setdefault("sqlalchemy.engine", engine_mod)

from menace.neuroplasticity import PathwayDB, PathwayRecord, Outcome
from menace.code_database import CodeDB
from menace.menace_memory_manager import MenaceMemoryManager
from menace import unified_learning_engine as ule


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


class DummyModel:
    def __init__(self) -> None:
        self.calls = 0

    def pretrain(self, X):
        self.calls += 1

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        return [[0.5, 0.5] for _ in X]


def test_pretrain_called_once(tmp_path, monkeypatch):
    pdb = PathwayDB(tmp_path / "p.db")
    mm = MenaceMemoryManager(tmp_path / "m.db", embedder=None)
    mm._embed = lambda text: [1.0]  # type: ignore
    roi = DummyROIDB()
    code_db = CodeDB(tmp_path / "c.db")

    pdb.log(
        PathwayRecord(
            actions="A",
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
            actions="B",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.FAILURE,
            roi=0.5,
        )
    )

    dummy = DummyModel()
    monkeypatch.setattr(ule, "SequenceModel", DummyModel)
    engine = ule.UnifiedLearningEngine(pdb, mm, code_db, roi, model=dummy)
    engine.evaluate()
    assert dummy.calls == 1
    with pytest.raises(ValueError):
        engine.tune_hyperparameters()
