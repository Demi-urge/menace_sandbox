import sys
import types
import pytest
import time

# Stub optional dependencies required during import
stub = types.ModuleType("stub")
stub.CollectorRegistry = object
stub.Counter = object
stub.Gauge = object
stub.TfidfVectorizer = object
stub.KMeans = object
stub.LinearRegression = object
stub.train_test_split = lambda *a, **k: ([], [])
stub.accuracy_score = lambda *a, **k: 0.0
stub.LogisticRegression = object
stub.RandomForestClassifier = object
sys.modules.setdefault("prometheus_client", stub)

# common libs
sys.modules.setdefault("numpy", stub)
sys.modules.setdefault("pandas", stub)
sys.modules.setdefault("networkx", stub)
sys.modules.setdefault("pulp", stub)
crypto = types.ModuleType("cryptography")
hazmat = types.ModuleType("hazmat")
primitives = types.ModuleType("primitives")
asym = types.ModuleType("asymmetric")
ed25519 = types.ModuleType("ed25519")
ed25519.Ed25519PrivateKey = object
ed25519.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
asym.ed25519 = ed25519
primitives.asymmetric = asym
primitives.serialization = serialization
hazmat.primitives = primitives
crypto.hazmat = hazmat
sys.modules.setdefault("cryptography", crypto)
sys.modules.setdefault("cryptography.hazmat", hazmat)
sys.modules.setdefault("cryptography.hazmat.primitives", primitives)
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", asym)
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric.ed25519", ed25519)
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)

# deap stubs
deap = types.ModuleType("deap")
deap.base = types.ModuleType("base")
deap.creator = types.ModuleType("creator")
deap.tools = types.ModuleType("tools")
sys.modules.setdefault("deap", deap)
sys.modules.setdefault("deap.base", deap.base)
sys.modules.setdefault("deap.creator", deap.creator)
sys.modules.setdefault("deap.tools", deap.tools)

# sklearn stubs
sys.modules.setdefault("sklearn", stub)
sys.modules.setdefault("sklearn.feature_extraction", stub)
sys.modules.setdefault("sklearn.feature_extraction.text", stub)
sys.modules.setdefault("sklearn.cluster", stub)
sys.modules.setdefault("sklearn.linear_model", stub)
sys.modules.setdefault("sklearn.model_selection", stub)
sys.modules.setdefault("sklearn.metrics", stub)
sys.modules.setdefault("sklearn.ensemble", stub)

from menace.workflow_cloner import WorkflowCloner
from menace.neuroplasticity import PathwayDB, PathwayRecord, Outcome
from menace.evolution_history_db import EvolutionHistoryDB


class DummyGA:
    def run_cycle(self, bot: str, generations: int = 1):
        return types.SimpleNamespace(roi=2.5)


class DummyCreator:
    async def create_bots(self, tasks, **kwargs):
        return [1]


def test_clone_logs_history(tmp_path):
    pdb = PathwayDB(tmp_path / "p.db")
    pid = pdb.log(
        PathwayRecord(
            actions="a->b",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.SUCCESS,
            roi=2.0,
        )
    )
    hist = EvolutionHistoryDB(tmp_path / "e.db")
    cloner = WorkflowCloner(
        pdb,
        ga_manager=DummyGA(),
        bot_creator=DummyCreator(),
        history_db=hist,
    )
    cloner.clone_top_workflows(limit=1)
    rows = hist.fetch()
    assert rows and rows[0][8] == pid


class FailingGA:
    def run_cycle(self, bot: str, generations: int = 1):
        raise RuntimeError("boom")


class FailingCreator:
    async def create_bots(self, tasks, **kwargs):
        raise RuntimeError("boom")


def test_clone_logs_error(tmp_path, caplog):
    pdb = PathwayDB(tmp_path / "p.db")
    pid = pdb.log(
        PathwayRecord(
            actions="a->b",
            inputs="",
            outputs="",
            exec_time=1.0,
            resources="",
            outcome=Outcome.SUCCESS,
            roi=2.0,
        )
    )
    hist = EvolutionHistoryDB(tmp_path / "e.db")
    cloner = WorkflowCloner(
        pdb,
        ga_manager=FailingGA(),
        bot_creator=FailingCreator(),
        history_db=hist,
    )
    caplog.set_level("ERROR")
    cloner._clone(pid)
    assert "ga run failed" in caplog.text
    assert "bot creation failed" in caplog.text


def test_loop_logs_exception(monkeypatch, caplog):
    cloner = WorkflowCloner(PathwayDB(":memory:"), interval=0)

    def boom():
        raise RuntimeError("fail")

    monkeypatch.setattr(cloner, "clone_top_workflows", boom)

    def stop(_):
        cloner.running = False

    monkeypatch.setattr(time, "sleep", stop)
    cloner.running = True
    caplog.set_level("ERROR")
    cloner._loop()
    assert "clone cycle failed" in caplog.text
