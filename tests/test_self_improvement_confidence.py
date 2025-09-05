import ast
import types
from pathlib import Path
from typing import Iterable

import pytest
from dynamic_path_router import resolve_path


source = resolve_path("self_improvement/engine.py").read_text()  # path-ignore
module_ast = ast.parse(source)
cls = next(
    n for n in module_ast.body if isinstance(n, ast.ClassDef) and n.name == "SelfImprovementEngine"
)
func_node = next(
    n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_score_modifications"
)
func_code = ast.get_source_segment(source, func_node)
globals_ns = {
    "Path": Path,
    "sts": types.SimpleNamespace(get_failed_critical_tests=lambda: []),
    "Iterable": Iterable,
    "log_record": lambda **k: k,
}
locals_ns: dict[str, object] = {}
exec(func_code, globals_ns, locals_ns)
score_mods = locals_ns["_score_modifications"]


class _DummyPredictor:
    def predict(self, features, horizon=None):
        roi = float(features[-1])
        category = "exponential" if roi == 10 else "marginal"
        return [roi], category, None, None


class _DummyTracker:
    def calculate_raroi(self, roi_est, workflow_type="standard", metrics=None, failing_tests=None):
        return float(roi_est), float(roi_est) * 0.8, []

    def workflow_confidence(self, workflow_id):
        return {"mod1": 0.9, "mod2": 0.3}.get(workflow_id, 0.0)

    def score_workflow(self, workflow_id, raroi, tau=None):
        conf = self.workflow_confidence(workflow_id)
        return raroi * conf, conf < (tau or 0.5), conf


class _DummyEngine:
    def __init__(self) -> None:
        self.entropy_ceiling_modules = set()
        self.roi_predictor = _DummyPredictor()
        self.roi_tracker = _DummyTracker()
        self.use_adaptive_roi = True
        self.tau = 0.5
        self.growth_multipliers = {"exponential": 2.0, "marginal": 0.5}
        self.growth_weighting = True
        self.logger = types.SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None)
        self._log_action = lambda *a, **k: None
        self._candidate_features = lambda mod: [10] if mod == "mod1" else [5]
        self.borderline_raroi_threshold = 0.0


def test_final_weighting_and_low_confidence_demotes():
    engine = _DummyEngine()
    scored = score_mods(engine, ["mod1", "mod2"])
    assert scored == [("mod1", 10.0, "exponential", pytest.approx(14.4))]
