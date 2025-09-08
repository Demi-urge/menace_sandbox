import pytest
pytest.importorskip("jinja2")
pytest.importorskip("git")
import os  # noqa: E402
import sys
import types
os.environ.setdefault("MENACE_DB_PATH", "/tmp")
os.environ.setdefault("MENACE_SHARED_DB_PATH", "/tmp")
pytest.skip("event integration dependencies unavailable", allow_module_level=True)

os.environ.setdefault("STRIPE_SECRET_KEY", "sk_live_dummy")
os.environ.setdefault("STRIPE_PUBLIC_KEY", "pk_live_dummy")

class DummyBuilder:
    def __init__(self):
        self.queried = []

    def refresh_db_weights(self):
        pass

    def query(self, term, **kwargs):  # pragma: no cover - stub
        self.queried.append(term)
        return {"snippets": [term], "metadata": {"q": term}}

# Provide lightweight vector_service and loguru stubs for imports
vector_service_stub = types.SimpleNamespace(
    ContextBuilder=DummyBuilder,
    FallbackResult=object,
    ErrorResult=object,
    Retriever=object,
    EmbeddingBackfill=object,
    CognitionLayer=object,
    EmbeddableDBMixin=object,
    SharedVectorService=object,
)
sys.modules.setdefault("vector_service", vector_service_stub)

# Stub db_router to prevent filesystem access during imports
class DummyConn:
    def execute(self, *a, **k):
        return None

    def commit(self):
        return None

class DummyRouter:
    def __init__(self, *a, **k):
        pass

    def query_all(self, term):
        return {}

    def get_connection(self, name):
        return DummyConn()

db_router_stub = types.SimpleNamespace(
    DBRouter=DummyRouter,
    GLOBAL_ROUTER=DummyRouter(),
    init_db_router=lambda name: DummyRouter(),
)
sys.modules.setdefault("menace.db_router", db_router_stub)

loguru_mod = types.ModuleType("loguru")
class DummyLogger:
    def add(self, *a, **k):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None

loguru_mod.logger = DummyLogger()
sys.modules.setdefault("loguru", loguru_mod)

import menace.data_bot as db  # noqa: E402
from menace.unified_event_bus import UnifiedEventBus  # noqa: E402
from menace.menace_memory_manager import MenaceMemoryManager, MemoryEntry  # noqa: E402


def test_error_bot_event_subscription(tmp_path):
    bus = UnifiedEventBus()
    try:
        import menace.error_bot as eb
    except Exception:
        pytest.skip("error bot unavailable")
    err_db = eb.ErrorDB(tmp_path / "e.db", event_bus=bus)
    metrics = db.MetricsDB(tmp_path / "m.db")
    bot = eb.ErrorBot(err_db, metrics, event_bus=bus, context_builder=DummyBuilder())
    bus.publish("errors:new", {"message": "boom"})
    assert bot.last_error_event


def test_offer_bot_event_subscription(tmp_path):
    bus = UnifiedEventBus()
    try:
        import menace.offer_testing_bot as ot
    except Exception:
        pytest.skip("offer bot unavailable")
    odb = ot.OfferDB(tmp_path / "o.db", event_bus=bus)
    bot = ot.OfferTestingBot(odb, event_bus=bus)
    bus.publish("variants:new", {"product": "p"})
    assert bot.last_variant_event


def test_finance_bot_memory_subscription(tmp_path):
    try:
        import menace.finance_router_bot as frb
    except Exception:
        pytest.skip("finance bot unavailable")
    mm = MenaceMemoryManager(tmp_path / "m.db")
    log = tmp_path / "p.json"
    bot = frb.FinanceRouterBot(payout_log_path=log, memory_mgr=mm)
    mm.log(MemoryEntry("k", "d", 1, "finance"))
    assert bot.last_finance_memory


def test_communication_bot_event_subscription(tmp_path):
    bus = UnifiedEventBus()
    mm = MenaceMemoryManager(tmp_path / "m.db")
    repo_path = tmp_path / "repo"
    try:
        import menace.communication_maintenance_bot as cmb
    except Exception:
        pytest.skip("communication bot unavailable")
    cmb.Repo.init(repo_path)
    builder = DummyBuilder()
    bot = cmb.CommunicationMaintenanceBot(
        cmb.MaintenanceDB(tmp_path / "c.db"),
        repo_path=repo_path,
        event_bus=bus,
        memory_mgr=mm,
        context_builder=builder,
    )
    bus.publish("deployments:new", {"id": 1})
    assert bot.last_deployment_event
    mm.log(MemoryEntry("x", "d", 1, "maintenance"))
    assert bot.last_memory_entry


def test_deployment_bot_event_subscription(tmp_path):
    bus = UnifiedEventBus()
    mm = MenaceMemoryManager(tmp_path / "m.db")
    try:
        import menace.deployment_bot as dep
    except Exception:
        pytest.skip("deployment bot unavailable")
    ddb = dep.DeploymentDB(tmp_path / "d.db", event_bus=bus)
    bot = dep.DeploymentBot(ddb, event_bus=bus, memory_mgr=mm)
    ddb.add("n", "ok", "{}")
    assert bot.last_deployment_event
    mm.log(MemoryEntry("z", "d", 1, "deploy"))
    assert bot.last_memory_entry


def test_optimizer_memory_subscription(tmp_path):
    bus = UnifiedEventBus()
    mm = MenaceMemoryManager(tmp_path / "m.db")
    try:
        import menace.resource_allocation_optimizer as rao
    except Exception:
        pytest.skip("optimizer unavailable")
    opt = rao.ResourceAllocationOptimizer(
        rao.ROIDB(tmp_path / "r.db"), event_bus=bus, memory_mgr=mm
    )
    bus.publish("errors:new", {"message": "e"})
    assert opt.last_error_event
    mm.log(MemoryEntry("r", "d", 1, "roi"))
    assert opt.last_memory_entry
