import sys
import os
import sys
import types
from pathlib import Path

menace_sandbox_pkg = types.ModuleType("menace_sandbox")
menace_sandbox_pkg.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules.setdefault("menace_sandbox", menace_sandbox_pkg)


class _StubBotRegistry:
    def _verify_signed_provenance(self, *args, **kwargs):
        return True


sandbox_bot_registry = types.ModuleType("menace_sandbox.bot_registry")
sandbox_bot_registry.BotRegistry = _StubBotRegistry
sys.modules.setdefault("menace_sandbox.bot_registry", sandbox_bot_registry)

root_pkg = types.ModuleType("menace")
root_pkg.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules.setdefault("menace", root_pkg)

root_bot_registry = types.ModuleType("menace.bot_registry")
root_bot_registry.BotRegistry = _StubBotRegistry
sys.modules.setdefault("menace.bot_registry", root_bot_registry)

sys.path.insert(0, str(Path(__file__).resolve().parent))

sandbox_runner_pkg = types.ModuleType("sandbox_runner")
sandbox_runner_pkg.__path__ = [str(Path(__file__).resolve().parent / "sandbox_runner")]
sys.modules.setdefault("sandbox_runner", sandbox_runner_pkg)

# Stub menace package to avoid heavy imports during tests
root_pkg = types.ModuleType("menace")
root_pkg.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules.setdefault("menace", root_pkg)
sub_pkg = types.ModuleType("menace.self_improvement")
sub_pkg.__path__ = [str(Path(__file__).resolve().parent / "self_improvement")]
sys.modules.setdefault("menace.self_improvement", sub_pkg)
sys.modules.setdefault("self_improvement", sub_pkg)
metric_stub = types.SimpleNamespace(
    labels=lambda **k: types.SimpleNamespace(inc=lambda: None)
)
metrics_stub = types.SimpleNamespace(
    Gauge=lambda *a, **k: metric_stub,
    CollectorRegistry=object,
    self_improvement_failure_total=metric_stub,
    environment_failure_total=metric_stub,
)
sys.modules.setdefault("metrics_exporter", metrics_stub)
sys.modules.setdefault("menace.metrics_exporter", metrics_stub)
sys.modules.setdefault(
    "menace.sandbox_settings",
    types.SimpleNamespace(
        SandboxSettings=lambda: types.SimpleNamespace(),
        normalize_workflow_tests=lambda value=None: [],
    ),
)
sys.modules.setdefault(
    "sandbox_settings",
    types.SimpleNamespace(
        SandboxSettings=lambda: types.SimpleNamespace(),
        normalize_workflow_tests=lambda value=None: [],
    ),
)

# Stub neurosales package and optional billing dependencies
neuro_pkg = types.ModuleType("neurosales")
neuro_pkg.__path__ = [
    str(Path(__file__).resolve().parent / "neurosales" / "neurosales"),
    str(Path(__file__).resolve().parent / "neurosales" / "scripts"),
]
sys.modules.setdefault("neurosales", neuro_pkg)
scripts_pkg = types.ModuleType("neurosales.scripts")
scripts_pkg.__path__ = [str(Path(__file__).resolve().parent / "neurosales" / "scripts")]
sys.modules.setdefault("neurosales.scripts", scripts_pkg)
sys.modules.setdefault(
    "stripe_billing_router", types.ModuleType("stripe_billing_router")
)
sqlalchemy_pkg = types.ModuleType("sqlalchemy")
sqlalchemy_orm_pkg = types.ModuleType("sqlalchemy.orm")
sqlalchemy_orm_pkg.declarative_base = lambda *a, **k: None
sqlalchemy_orm_pkg.sessionmaker = lambda *a, **k: None
sqlalchemy_orm_pkg.relationship = lambda *a, **k: None
sys.modules.setdefault("sqlalchemy", sqlalchemy_pkg)
sys.modules.setdefault("sqlalchemy.orm", sqlalchemy_orm_pkg)
neo4j_stub = types.ModuleType("neo4j")
neo4j_stub.GraphDatabase = type(
    "GraphDatabase", (), {"driver": staticmethod(lambda *a, **k: None)}
)
sys.modules.setdefault("neo4j", neo4j_stub)


class _StubDiGraph:
    def __init__(self):
        self.nodes = {}
        self._edges = {}

    def add_node(self, name):
        self.nodes.setdefault(name, {})
        self._edges.setdefault(name, [])

    def add_edge(self, u, v, **data):
        self.add_node(u)
        self.add_node(v)
        self._edges[u].append((v, dict(data)))

    def has_edge(self, u, v):
        return any(neigh == v for neigh, _ in self._edges.get(u, []))

    def successors(self, node):
        for neigh, _ in self._edges.get(node, []):
            yield neigh

    def edges(self, data=False):
        for u, neighbours in self._edges.items():
            for v, edge_data in neighbours:
                if data:
                    yield (u, v, edge_data)
                else:
                    yield (u, v)

    def clear(self):
        self.nodes.clear()
        self._edges.clear()


networkx_stub = types.ModuleType("networkx")
networkx_stub.DiGraph = _StubDiGraph
sys.modules.setdefault("networkx", networkx_stub)

if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")

    class _Array(list):
        def reshape(self, *_shape):
            return self

        def tolist(self):
            return list(self)

    numpy_stub.array = lambda data, dtype=None: _Array(list(data))
    numpy_stub.arange = lambda n, *a, **k: _Array(range(n))
    numpy_stub.isscalar = lambda value: isinstance(value, (int, float, complex))
    numpy_stub.bool_ = bool
    numpy_stub.percentile = lambda data, percentile: 0.0
    numpy_stub.std = lambda data, ddof=0: 0.0
    numpy_stub.dot = lambda a, b: sum(float(x) * float(y) for x, y in zip(a, b))
    sys.modules.setdefault("numpy", numpy_stub)

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_a, **_k: {}
    yaml_stub.safe_dump = lambda *_a, **_k: ""
    yaml_stub.dump = lambda *_a, **_k: ""
    sys.modules.setdefault("yaml", yaml_stub)

# Stub unified_event_bus to avoid heavy dependencies
bus_stub = types.ModuleType("unified_event_bus")
bus_stub.UnifiedEventBus = lambda *a, **k: None
sys.modules.setdefault("unified_event_bus", bus_stub)

# Provide a lightweight dynamic_path_router to satisfy imports during tests
sys.modules.setdefault(
    "dynamic_path_router",
    types.SimpleNamespace(
        resolve_path=lambda p: Path(p),
        resolve_dir=lambda p: Path(p),
        path_for_prompt=lambda p: Path(p).as_posix(),
        repo_root=lambda: Path("."),
        get_project_root=lambda: Path("."),
    ),
)

threshold_stub = types.ModuleType("menace_sandbox.threshold_service")
threshold_stub.threshold_service = types.SimpleNamespace(load=lambda *_a, **_k: None)
sys.modules.setdefault("menace_sandbox.threshold_service", threshold_stub)
sys.modules.setdefault("threshold_service", threshold_stub)

import pytest
from menace_sandbox.bot_registry import BotRegistry as SandboxBotRegistry
from menace.bot_registry import BotRegistry as RootBotRegistry


@pytest.fixture(autouse=True)
def _auto_verify_signed_provenance(monkeypatch):
    """Default to accepting signed provenance during tests.

    Individual tests can override this behaviour by monkeypatching the
    :meth:`BotRegistry._verify_signed_provenance` method.
    """

    for cls in (SandboxBotRegistry, RootBotRegistry):
        monkeypatch.setattr(cls, "_verify_signed_provenance", lambda self, p, c: True)
