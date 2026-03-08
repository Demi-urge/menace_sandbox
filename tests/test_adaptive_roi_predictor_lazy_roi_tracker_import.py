from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


def test_script_mode_import_does_not_eagerly_require_roi_tracker_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Importing adaptive_roi_predictor should not require roi_tracker.ROITracker eagerly."""

    # Mimic service_supervisor startup style: top-level script-mode import.
    (tmp_path / "roi_tracker.py").write_text(
        "from __future__ import annotations\n"
        "__all__ = []\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    roi_tracker_stub = types.ModuleType("roi_tracker")
    monkeypatch.setitem(sys.modules, "roi_tracker", roi_tracker_stub)
    monkeypatch.setitem(sys.modules, "menace_sandbox.roi_tracker", roi_tracker_stub)

    logging_stub = types.ModuleType("logging_utils")
    logging_stub.get_logger = lambda *_a, **_k: types.SimpleNamespace(
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "logging_utils", logging_stub)
    monkeypatch.setitem(sys.modules, "menace_sandbox.logging_utils", logging_stub)

    eval_stub = types.ModuleType("evaluation_history_db")
    eval_stub.EvaluationHistoryDB = object
    monkeypatch.setitem(sys.modules, "evaluation_history_db", eval_stub)
    monkeypatch.setitem(sys.modules, "menace_sandbox.evaluation_history_db", eval_stub)

    evo_stub = types.ModuleType("evolution_history_db")
    evo_stub.EvolutionHistoryDB = object
    monkeypatch.setitem(sys.modules, "evolution_history_db", evo_stub)
    monkeypatch.setitem(sys.modules, "menace_sandbox.evolution_history_db", evo_stub)

    truth_stub = types.ModuleType("truth_adapter")
    truth_stub.TruthAdapter = object
    monkeypatch.setitem(sys.modules, "truth_adapter", truth_stub)
    monkeypatch.setitem(sys.modules, "menace_sandbox.truth_adapter", truth_stub)

    dataset_stub = types.ModuleType("adaptive_roi_dataset")
    dataset_stub.build_dataset = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("unused"))
    dataset_stub._label_growth = lambda *_a, **_k: "marginal"
    monkeypatch.setitem(sys.modules, "adaptive_roi_dataset", dataset_stub)
    monkeypatch.setitem(sys.modules, "menace_sandbox.adaptive_roi_dataset", dataset_stub)

    for name in (
        "adaptive_roi_predictor",
        "menace_sandbox.adaptive_roi_predictor",
    ):
        sys.modules.pop(name, None)

    mod = importlib.import_module("adaptive_roi_predictor")
    assert mod is not None

    with pytest.raises(ImportError, match="ROITracker"):
        mod._get_roi_tracker_cls()
