import sys
import types
from pathlib import Path
import os

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
sys.modules.setdefault(
    "menace.metrics_exporter",
    types.SimpleNamespace(self_improvement_failure_total=metric_stub),
)
sys.modules.setdefault(
    "menace.sandbox_settings",
    types.SimpleNamespace(SandboxSettings=lambda: types.SimpleNamespace()),
)

# Provide a lightweight dynamic_path_router to satisfy imports during tests
sys.modules.setdefault(
    "dynamic_path_router",
    types.SimpleNamespace(resolve_path=lambda p: Path(p), resolve_dir=lambda p: Path(p)),
)

os.environ.setdefault("VISUAL_AGENT_URLS", "http://127.0.0.1:8001")
os.environ.setdefault("VISUAL_AGENT_TOKEN", "test-token")
