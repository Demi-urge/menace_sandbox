import sys
import types
import importlib.util
from pathlib import Path

import pytest

# Set up a lightweight package structure to import metrics without executing
# the heavy package __init__ from the repository root.
ROOT = Path(__file__).resolve().parents[2]
PKG = "pkg"

root_pkg = types.ModuleType(PKG)
root_pkg.__path__ = [str(ROOT)]
sys.modules.setdefault(PKG, root_pkg)

sub_pkg = types.ModuleType(f"{PKG}.self_improvement")
sub_pkg.__path__ = [str(ROOT / "self_improvement")]
sys.modules.setdefault(f"{PKG}.self_improvement", sub_pkg)
sys.modules.setdefault("self_improvement", sub_pkg)

# Stub out dynamic_path_router to simple path operations
stub = types.ModuleType("dynamic_path_router")
stub.resolve_path = lambda p: Path(p)
stub.resolve_dir = lambda p: Path(p)
stub.repo_root = lambda: ROOT
sys.modules.setdefault("dynamic_path_router", stub)
sys.modules.setdefault(f"{PKG}.dynamic_path_router", stub)

from dynamic_path_router import resolve_path

# Stub out sandbox_settings and pydantic to avoid heavy dependencies
sandbox_stub = types.ModuleType("sandbox_settings")


class SandboxSettings:  # minimal stub
    def __init__(self, sandbox_repo_path: str = ""):
        self.sandbox_repo_path = sandbox_repo_path
        self.metrics_skip_dirs = []


sandbox_stub.SandboxSettings = SandboxSettings
sys.modules.setdefault("sandbox_settings", sandbox_stub)
sys.modules.setdefault(f"{PKG}.sandbox_settings", sandbox_stub)

# Resolve the real sandbox_settings path for completeness
_ = resolve_path("sandbox_settings.py")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    sys.modules[name] = mod
    return mod

metrics_mod = _load(
    f"{PKG}.self_improvement.metrics",
    resolve_path("self_improvement/metrics.py"),
)

SandboxSettings = sandbox_stub.SandboxSettings
collect_snapshot_metrics = metrics_mod.collect_snapshot_metrics
_collect_metrics = metrics_mod._collect_metrics
compute_call_graph_complexity = metrics_mod.compute_call_graph_complexity


def test_collect_snapshot_metrics_matches_internal(tmp_path):
    file1 = tmp_path / "a.py"  # path-ignore
    file1.write_text("a=1\n")
    file2 = tmp_path / "b.py"  # path-ignore
    file2.write_text("b=2\n")
    settings = SandboxSettings(sandbox_repo_path=str(tmp_path))
    files = [file1, file2]
    _, _, _, _, exp_entropy, exp_div = _collect_metrics(files, tmp_path, settings=settings)
    entropy, diversity = collect_snapshot_metrics(files, settings=settings)
    assert entropy == pytest.approx(exp_entropy)
    assert diversity == pytest.approx(exp_div)


def test_compute_call_graph_complexity(monkeypatch):
    import networkx as nx

    g = nx.DiGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")

    monkeypatch.setattr(metrics_mod, "build_import_graph", lambda root: g)
    assert compute_call_graph_complexity(Path(".")) == pytest.approx(1.0)
