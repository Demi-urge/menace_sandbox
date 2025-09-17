import sys
import types
from pathlib import Path
import os

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
        get_project_root=lambda: Path(")."),
    ),
)

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
