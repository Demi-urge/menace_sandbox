import ast
import types
from pathlib import Path
from typing import Any, Dict

ENG_PATH = Path(__file__).resolve().parents[1] / "self_improvement" / "engine.py"
src = ENG_PATH.read_text()
tree = ast.parse(src)
future = ast.ImportFrom(module="__future__", names=[ast.alias("annotations", None)], level=0)
sie_cls = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SelfImprovementEngine"
)
methods = [
    n
    for n in sie_cls.body
    if isinstance(n, ast.FunctionDef) and n.name in ("_check_momentum", "momentum_coefficient")
]
engine_module = ast.Module(
    [future, ast.ClassDef("SelfImprovementEngine", [], [], methods, [])],
    type_ignores=[],
)
engine_module = ast.fix_missing_locations(engine_module)
ns: Dict[str, Any] = {"log_record": lambda **k: k}
exec(compile(engine_module, "<engine>", "exec"), ns)
SelfImprovementEngine = ns["SelfImprovementEngine"]


def _make_engine():
    eng = SelfImprovementEngine.__new__(SelfImprovementEngine)
    eng.success_history = [False, False, True, False]
    eng.momentum_window = 4
    eng.urgency_tier = 0
    eng.logger = types.SimpleNamespace(warning=lambda *a, **k: None)
    eng.baseline_tracker = types.SimpleNamespace(
        get=lambda m: 0.75, std=lambda m: 0.05
    )
    eng.momentum_dev_multiplier = 1.0
    eng.stagnation_cycles = 2
    eng._momentum_streak = 0
    return eng


def test_momentum_stagnation_escalates():
    eng = _make_engine()
    eng._check_momentum()
    assert eng.urgency_tier == 0
    eng._check_momentum()
    assert eng.urgency_tier == 1
