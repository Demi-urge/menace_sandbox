import ast
import types
from pathlib import Path
from typing import Any, Dict

ENG_PATH = Path(__file__).resolve().parents[1] / "self_improvement" / "engine.py"  # path-ignore
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
ns: Dict[str, Any] = {
    "log_record": lambda **k: k,
    "settings": types.SimpleNamespace(
        momentum_stagnation_dev_multiplier=1.0
    ),
}
exec(compile(engine_module, "<engine>", "exec"), ns)
SelfImprovementEngine = ns["SelfImprovementEngine"]


def _make_engine(deltas: list[float] | None = None):
    class Tracker:
        def __init__(self, deltas: list[float], std: float = 0.05):
            self._deltas = iter(deltas)
            self._std = std

        def delta(self, metric: str) -> float:
            return next(self._deltas)

        def std(self, metric: str) -> float:
            return self._std

    eng = SelfImprovementEngine.__new__(SelfImprovementEngine)
    eng.urgency_tier = 0
    eng.logger = types.SimpleNamespace(warning=lambda *a, **k: None)
    eng.baseline_tracker = Tracker(deltas or [-0.1, -0.1])
    eng.stagnation_cycles = 2
    eng._momentum_streak = 0
    eng.urgency_recovery_threshold = 0.05
    return eng


def test_momentum_stagnation_escalates():
    eng = _make_engine()
    eng._check_momentum()
    assert eng.urgency_tier == 0
    eng._check_momentum()
    assert eng.urgency_tier == 1


def test_momentum_recovery_resets():
    eng = _make_engine([-0.1, -0.1, 0.1])
    eng._check_momentum()
    eng._check_momentum()
    assert eng.urgency_tier == 1
    eng._check_momentum()
    assert eng.urgency_tier == 0
