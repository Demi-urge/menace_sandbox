import ast
import sys
import types
import json
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

from menace_sandbox.sandbox_settings import SandboxSettings  # noqa: E402
import menace_sandbox.prompt_optimizer as prompt_optimizer  # noqa: E402
from dynamic_path_router import resolve_path  # noqa: E402


def _load_select_prompt_strategy(namespace):
    eng_path = Path(resolve_path(str(ROOT / "self_improvement" / "engine.py")))
    mod = ast.parse(eng_path.read_text())
    class_def = next(
        n
        for n in mod.body
        if isinstance(n, ast.ClassDef) and n.name == "SelfImprovementEngine"
    )
    func_node = next(
        n
        for n in class_def.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_select_prompt_strategy"
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias("annotations", None)], level=0)
    module = ast.Module([future, func_node], type_ignores=[])
    module = ast.fix_missing_locations(module)
    exec(compile(module, str(eng_path), "exec"), namespace)
    return namespace["_select_prompt_strategy"]


def test_select_prompt_strategy_prefers_high_roi(tmp_path, monkeypatch):
    stats_path = tmp_path / "stats.json"
    data = {
        "s1": {
            "success": 1,
            "total": 1,
            "roi_sum": 0.5,
            "weighted_roi_sum": 0.5,
            "weight_sum": 1.0,
        },
        "s2": {
            "success": 1,
            "total": 1,
            "roi_sum": 1.5,
            "weighted_roi_sum": 1.5,
            "weight_sum": 1.0,
        },
    }
    stats_path.write_text(json.dumps(data))

    namespace = {
        "load_prompt_penalties": lambda: {},
        "load_strategy_stats": lambda: prompt_optimizer.load_strategy_stats(stats_path),
        "SandboxSettings": SandboxSettings,
    }
    select = _load_select_prompt_strategy(namespace)

    stub = types.SimpleNamespace(deprioritized_strategies=set(), pending_strategy=None)
    choice = select(stub, ["s1", "s2"])
    assert choice == "s2"
