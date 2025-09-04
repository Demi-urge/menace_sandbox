import ast
import sys
import types
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

router = types.ModuleType("dynamic_path_router")
router.resolve_path = lambda p: p
router.repo_root = lambda: ROOT
sys.modules.setdefault("dynamic_path_router", router)

boot = types.ModuleType("sandbox_runner.bootstrap")
boot.initialize_autonomous_sandbox = lambda *a, **k: None
sys.modules.setdefault("sandbox_runner.bootstrap", boot)

pkg = types.ModuleType("menace_sandbox.self_improvement")
pkg.__path__ = [str(ROOT / "self_improvement")]
sys.modules["menace_sandbox.self_improvement"] = pkg

from menace_sandbox.sandbox_settings import SandboxSettings
from filelock import FileLock

prompt_memory = importlib.import_module("menace_sandbox.self_improvement.prompt_memory")


def _load_select_prompt_strategy(namespace):
    eng_path = ROOT / "self_improvement" / "engine.py"
    mod = ast.parse(eng_path.read_text())
    class_def = next(
        n for n in mod.body if isinstance(n, ast.ClassDef) and n.name == "SelfImprovementEngine"
    )
    func_node = next(
        n for n in class_def.body if isinstance(n, ast.FunctionDef) and n.name == "_select_prompt_strategy"
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias("annotations", None)], level=0)
    module = ast.Module([future, func_node], type_ignores=[])
    module = ast.fix_missing_locations(module)
    exec(compile(module, str(eng_path), "exec"), namespace)
    return namespace["_select_prompt_strategy"]


def test_select_prompt_strategy_prefers_high_roi(tmp_path, monkeypatch):
    stats_path = tmp_path / "stats.json"
    monkeypatch.setattr(prompt_memory, "_strategy_stats_path", stats_path)
    monkeypatch.setattr(prompt_memory, "_strategy_lock", FileLock(str(stats_path) + ".lock"))

    prompt_memory.update_strategy_roi("s1", 0.5)
    prompt_memory.update_strategy_roi("s2", 1.5)

    namespace = {
        "load_prompt_penalties": lambda: {},
        "load_strategy_roi_stats": prompt_memory.load_strategy_roi_stats,
        "SandboxSettings": SandboxSettings,
    }
    select = _load_select_prompt_strategy(namespace)

    stub = types.SimpleNamespace(deprioritized_strategies=set(), pending_strategy=None)
    choice = select(stub, ["s1", "s2"])
    assert choice == "s2"
