import ast
import types
import importlib.util
from pathlib import Path

from dynamic_path_router import resolve_path


def _load_baseline_tracker():
    spec = importlib.util.spec_from_file_location(
        "baseline", resolve_path("self_improvement/baseline_tracker.py")  # path-ignore
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BaselineTracker


def _load_eval_method():
    src = resolve_path("self_improvement/engine.py").read_text()  # path-ignore
    tree = ast.parse(src)
    func = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SelfImprovementEngine":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_evaluate_module_relevance":
                    func = item
                    break
    assert func is not None
    mod = types.ModuleType("dummy_engine_mod")
    class Settings:
        relevancy_metrics_db_path = "metrics.db"
        auto_process_relevancy_flags = False
        relevancy_deviation_multiplier = 1.0
        relevancy_history_min_length = 1
    mod.SandboxSettings = lambda: Settings
    mod.Path = Path
    mod.radar_scan = lambda *a, **k: {}
    mod._repo_path = lambda: Path(".")
    exec(compile(ast.Module(body=[func], type_ignores=[]), "dummy", "exec"), mod.__dict__)
    return mod._evaluate_module_relevance


class DummyLogger:
    def exception(self, *a, **k):
        pass


def test_thresholds_adjust_with_history():
    BaselineTracker = _load_baseline_tracker()
    eval_method = _load_eval_method()
    captured: list[tuple[float, float]] = []

    class Radar:
        def __init__(self, score: float):
            self._metrics = {
                "mod": {
                    "imports": score / 2,
                    "executions": score / 2,
                    "impact": 0.0,
                    "output_impact": 0.0,
                }
            }

        def evaluate_final_contribution(self, compress: float, replace: float):
            captured.append((compress, replace))
            return {}

    class Engine:
        def __init__(self, score: float):
            self.baseline_tracker = BaselineTracker(window=5)
            self.relevancy_radar = Radar(score)
            self.relevancy_flags: dict[str, str] = {}
            self.event_bus = None
            self.logger = DummyLogger()

        _evaluate_module_relevance = eval_method

    eng = Engine(50.0)
    for val in [50.0, 50.0, 50.0]:
        eng.baseline_tracker.update(relevancy=val)
    eng._evaluate_module_relevance()
    eng.relevancy_radar._metrics["mod"]["imports"] = 0.0
    eng.relevancy_radar._metrics["mod"]["executions"] = 0.0
    eng._evaluate_module_relevance()

    assert captured[1][1] < captured[0][1]
    assert captured[1][0] <= captured[0][0]
